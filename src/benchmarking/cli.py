"""Benchmarking CLI -- eval a checkpoint to the store, then compare / summarise.
    eval      score a trained checkpoint over footprint chips -> sharded store
    eval-dir  eval EVERY checkpoint under a dir (model_name/seed/θ* from each
              run's train_meta.json/sweep.json) into one store, then report
    compare   paired bootstrap CI + Wilcoxon signed-rank between two models
    variance  cross-seed mean +/- std + 95% CI for one model (training instability)
    report    per-model mean +/- std + all pairwise comparisons (multi-metric,
              optional markdown/CSV export)

Metrics resolve against the chips table first, then the tiles table (plugin
metrics like APLS pair on tile_id). ``compare``/``report`` guard against
comparing models that were scored against DIFFERENT ground truth (resolution /
label source): pixel metrics are only comparable within one GT; graph metrics
are the cross-GT route.
"""
import itertools
import json
from pathlib import Path
from typing import Annotated, List, Optional

import typer
import yaml

app = typer.Typer(help="Road-segmentation benchmarking: eval + statistical comparison")

# Runs-table columns that define "same ground truth / same protocol". Models
# whose runs disagree on any of these are not pixel-comparable.
_GT_KEYS = ("gt_res_m", "label_source", "mask_source", "mask_dirname",
            "dataset_split", "cell_m")


@app.callback()
def _main(
    ctx: typer.Context,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            help="YAML of per-command option defaults; top-level keys are command "
            "names (`eval`, `compare`, `variance`, `report`). Explicit flags override.",
        ),
    ] = None,
):
    """Benchmarking CLI. ``--config FILE`` pre-fills each command's options."""
    if config is not None:
        loaded = yaml.safe_load(config.read_text()) or {}
        ctx.default_map = {**(ctx.default_map or {}), **loaded}


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _load_metric_table(store_dir, metric):
    """(df, unit) for ``metric``: the chips table if the column lives there,
    else the tiles table with ``tile_id`` renamed to ``chip_id`` — the stats
    functions pair on ``chip_id``, and at tile granularity the tile IS the unit."""
    from benchmarking.store import load_chips, load_tiles

    chips = load_chips(store_dir)
    if metric in chips.columns:
        return chips, "chip"
    try:
        tiles = load_tiles(store_dir)
    except FileNotFoundError:
        tiles = None
    if tiles is not None and metric in tiles.columns:
        return tiles.rename(columns={"tile_id": "chip_id"}), "tile"
    raise typer.BadParameter(
        f"metric {metric!r} found in neither chips nor tiles columns"
    )


def _comparability_issues(store_dir, models) -> list[str]:
    """Human-readable list of GT/protocol keys the models disagree on."""
    from benchmarking.store import load_runs

    try:
        runs = load_runs(store_dir)
    except FileNotFoundError:
        return []  # legacy store without runs metadata — nothing to check
    if "model_name" not in runs.columns:
        return []
    sub = runs[runs["model_name"].isin(list(models))]
    issues = []
    for key in _GT_KEYS:
        if key not in sub.columns:
            continue
        vals = sub.groupby("model_name")[key].agg(
            lambda s: ", ".join(sorted({str(v) for v in s.dropna()}))
        )
        if len(set(vals)) > 1:
            issues.append(f"{key}: " + "; ".join(f"{m}=[{v}]" for m, v in vals.items()))
    return issues


def _guard_comparable(store_dir, models, force: bool):
    issues = _comparability_issues(store_dir, models)
    if not issues:
        return
    typer.secho("WARNING: these models were scored against different ground truth "
                "or protocol:", fg=typer.colors.RED, err=True)
    for line in issues:
        typer.secho(f"  {line}", fg=typer.colors.RED, err=True)
    if not force:
        typer.secho(
            "Pixel metrics are not comparable across different GT (compare within "
            "one label source/resolution; use graph metrics across). "
            "Pass --force to override.",
            fg=typer.colors.RED, err=True,
        )
        raise typer.Exit(2)
    typer.secho("--force given: comparing anyway.", fg=typer.colors.YELLOW, err=True)


def _md_table(headers, rows) -> str:
    """CommonMark table without the tabulate dependency."""
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(" --- " for _ in headers) + "|"
    body = ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _bench_summary(store_dir, run_id: str, split: str) -> dict:
    """Flat ``bench_<split>/…`` summary of one benchmark run: chip-level pixel
    means (at the run's θ), chip- and tile-level APLS means (NaN-skipping),
    plus θ and the store run_id for cross-reference. Pure — no wandb here, so
    it tests without one."""
    from benchmarking.store import load_chips, load_runs, load_tiles

    prefix = f"bench_{split}/"
    chips = load_chips(store_dir)
    chips = chips[chips["run_id"] == run_id]
    summary = {}
    for m in ("iou", "f1", "precision", "recall"):
        if m in chips.columns:
            summary[prefix + m] = float(chips[m].mean())   # pandas mean skips NaN
    for m in ("apls", "cldice"):
        if m in chips.columns:
            summary[prefix + m + "_chip"] = float(chips[m].mean())
    try:
        tiles = load_tiles(store_dir)
        tiles = tiles[tiles["run_id"] == run_id]
        for col in ("apls", "apls_gt_to_prop", "apls_prop_to_gt", "cldice"):
            if col in tiles.columns:
                summary[prefix + col] = float(tiles[col].mean())
    except FileNotFoundError:
        pass  # no tile-metric plugins ran
    runs = load_runs(store_dir)
    row = runs[runs["run_id"] == run_id].iloc[0]
    summary[prefix + "threshold"] = float(row["threshold"])
    summary[prefix + "n_chips"] = int(row["n_chips"])
    summary[prefix + "store_run_id"] = run_id
    return summary


def _push_bench_to_wandb(meta_path: Path, run_id: str, store_dir, split: str):
    """Resume the FIT stage's wandb run (id recorded in train_meta.json by
    unet.train_ablation) and update its summary with the benchmark metrics —
    this is how val APLS reaches the wandb table alongside train/val curves."""
    import json
    import os

    meta = yaml.safe_load(Path(meta_path).read_text()) if str(meta_path).endswith(
        (".yaml", ".yml")) else json.loads(Path(meta_path).read_text())
    info = meta.get("wandb") or {}
    if not info.get("id"):
        # Legacy runs: fits from before train_meta.json carried a wandb block.
        # WandbLogger(save_dir=run_dir) leaves wandb/run-<ts>-<id>/ next to the
        # meta — recover the id from the newest one.
        run_dirs = sorted(Path(meta_path).parent.glob("wandb/run-*"))
        if run_dirs:
            import os as _os
            info = {"id": run_dirs[-1].name.rsplit("-", 1)[-1],
                    "project": _os.environ.get("WANDB_PROJECT"),
                    "entity": None, "name": meta.get("run_name")}
            typer.echo(f"wandb: no block in train_meta.json; recovered legacy "
                       f"run id {info['id']} from {run_dirs[-1].name}")
    if not info.get("id"):
        typer.secho("WARN: no wandb run id found (fit ran with wandb disabled?) "
                    "— skipping wandb push", fg=typer.colors.YELLOW, err=True)
        return
    summary = _bench_summary(store_dir, run_id, split)

    import wandb

    run = wandb.init(project=info.get("project"), entity=info.get("entity"),
                     id=info["id"], resume="allow",
                     mode=os.environ.get("WANDB_MODE") or None)
    run.summary.update(summary)
    run.finish()
    typer.echo(f"wandb: pushed {len(summary)} bench metrics onto run "
               f"{info.get('name') or info['id']}")


def _find_checkpoint(run_dir: Path) -> Optional[Path]:
    """The checkpoint inside one run dir: ``checkpoints/best_f1.ckpt`` if present,
    else the first ``checkpoints/*.ckpt``, else the first ``*.ckpt`` in the dir.
    ``None`` when the dir has no checkpoint (e.g. an HPC run whose weights were
    never synced back — only its logs/config are here)."""
    best = run_dir / "checkpoints" / "best_f1.ckpt"
    if best.exists():
        return best
    for cand in sorted(run_dir.glob("checkpoints/*.ckpt")) + sorted(run_dir.glob("*.ckpt")):
        return cand
    return None


def _discover_ckpt_runs(ckpt_dir: Path) -> list[dict]:
    """One spec per model dir under ``ckpt_dir``. Each spec carries the checkpoint
    path and the ``model_name`` / ``seed`` / ``threshold`` (θ*) recovered from the
    sibling ``train_meta.json`` (falling back to ``sweep.json`` for θ* and to the
    dir name for ``model_name``). ``checkpoint=None`` marks a dir we must skip."""
    specs = []
    for run_dir in sorted(p for p in ckpt_dir.iterdir() if p.is_dir()):
        meta = {}
        meta_path = run_dir / "train_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        threshold = meta.get("best_threshold")
        if threshold is None:
            sweep_path = run_dir / "sweep.json"
            if sweep_path.exists():
                threshold = json.loads(sweep_path.read_text()).get("best_threshold")
        cfg = run_dir / "config.yaml"
        specs.append({
            "run_dir": run_dir,
            "checkpoint": _find_checkpoint(run_dir),
            "model_name": meta.get("arm") or run_dir.name,
            "seed": int(meta.get("seed", 0)),
            "threshold": threshold,
            "config_yaml": cfg if cfg.exists() else None,
        })
    return specs


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command(name="eval")
def run_eval(
    dataset_dir: Annotated[Path, typer.Option(help="ROSA dataset root (has splits/<split>.csv)")],
    checkpoint: Annotated[Path, typer.Option(help="Trained .ckpt to evaluate")],
    model_name: Annotated[str, typer.Option(help="Config identifier the stats pair/group on, e.g. unet_cdngi")],
    seed: Annotated[int, typer.Option(help="Training seed (for cross-seed CIs)")],
    store_dir: Annotated[Path, typer.Option(help="Sharded store dir (runs/ chips/ tiles/)")],
    split: Annotated[str, typer.Option(help="Split to evaluate")] = "test",
    model: Annotated[str, typer.Option(help="Model family loader: unet | sr")] = "unet",
    cell_m: Annotated[float, typer.Option(help="Footprint cell edge in metres (chip unit)")] = 2560.0,
    chip_px: Annotated[Optional[int], typer.Option(help="Override: cell edge in native px (bypasses the transform)")] = None,
    batch_size: Annotated[int, typer.Option(help="Chips per forward pass (unet family)")] = 8,
    mask_source: Annotated[Optional[str], typer.Option(help="sr GT: graph (masks_graph parquet) | raster (<split>/<mask-dirname>)")] = None,
    mask_dirname: Annotated[Optional[str], typer.Option(help="unet: remap the CSV mask dir; sr raster: HR mask dir")] = None,
    sen2sr_dir: Annotated[Optional[Path], typer.Option(help="Override the checkpoint's baked-in SR weights dir")] = None,
    config_yaml: Annotated[Optional[Path], typer.Option(help="Training config (e.g. best_params.yaml) -> config_hash")] = None,
    exp_tag: Annotated[str, typer.Option(help="Experiment tag (e.g. cdngi, r2a_cdngi)")] = "",
    label_source: Annotated[str, typer.Option(help="GT label source (cdngi | osm | overture)")] = "",
    tile_metric: Annotated[List[str], typer.Option(help="Tile-metric plugin(s), repeatable (see benchmarking.tile_metrics)")] = None,
    check: Annotated[str, typer.Option(help="tp+fn-vs-mask invariant: first | all | off")] = "first",
    device: Annotated[Optional[str], typer.Option(help="cuda | cpu (default: auto)")] = None,
    threshold: Annotated[Optional[float], typer.Option(help="Override the checkpoint's binarisation threshold (e.g. a tuned θ*)")] = None,
    max_tiles: Annotated[Optional[int], typer.Option(help="Score only the first N tiles of the split (quick local smoke)")] = None,
    wandb_meta: Annotated[Optional[Path], typer.Option(help="train_meta.json with a `wandb` block: resume that run and push the bench metrics (incl. APLS) to its summary")] = None,
):
    """Score a checkpoint over the split's footprint chips -> the sharded store."""
    from benchmarking.runner import evaluate

    run_id = evaluate(
        dataset_dir=dataset_dir, checkpoint=checkpoint, model_name=model_name,
        seed=seed, store_dir=store_dir, split=split, model=model, cell_m=cell_m,
        chip_px=chip_px, batch_size=batch_size, mask_source=mask_source,
        mask_dirname=mask_dirname, sen2sr_dir=sen2sr_dir, config_yaml_path=config_yaml,
        exp_tag=exp_tag, label_source=label_source,
        tile_metrics=tuple(tile_metric or ()), check=check, device=device,
        threshold=threshold, max_tiles=max_tiles,
    )
    if wandb_meta is not None:
        _push_bench_to_wandb(wandb_meta, run_id, store_dir, split)


@app.command(name="eval-dir")
def run_eval_dir(
    dataset_dir: Annotated[Path, typer.Option(help="ROSA dataset root (has splits/<split>.csv)")],
    ckpt_dir: Annotated[Path, typer.Option(help="Directory of model run dirs; each holds a .ckpt + train_meta.json/sweep.json")],
    store_dir: Annotated[Path, typer.Option(help="Sharded store dir to write (runs/ chips/ tiles/)")],
    split: Annotated[str, typer.Option(help="Split to evaluate")] = "test",
    model: Annotated[str, typer.Option(help="Model family loader: unet | sr")] = "unet",
    tile_metric: Annotated[List[str], typer.Option(help="Tile-metric plugin(s), repeatable; pass 'none' for pixel metrics only")] = None,
    metric: Annotated[List[str], typer.Option(help="Report metric(s), repeatable")] = None,
    aggregation: Annotated[str, typer.Option(help="micro or macro cross-seed aggregation for the report")] = "micro",
    batch_size: Annotated[int, typer.Option(help="Chips per forward pass (unet family)")] = 8,
    cell_m: Annotated[float, typer.Option(help="Footprint cell edge in metres (chip unit)")] = 2560.0,
    chip_px: Annotated[Optional[int], typer.Option(help="Override: cell edge in native px")] = None,
    label_source: Annotated[str, typer.Option(help="GT label source stamped on every run (comparability key)")] = "",
    device: Annotated[Optional[str], typer.Option(help="cuda | cpu (default: auto)")] = None,
    max_tiles: Annotated[Optional[int], typer.Option(help="Score only the first N tiles per model (quick local smoke)")] = None,
    skip_existing: Annotated[bool, typer.Option(help="Skip a (model_name, seed) already in the store")] = True,
    report: Annotated[bool, typer.Option(help="Run `report` over the store when all evals finish")] = True,
    out: Annotated[Optional[Path], typer.Option(help="Write the final report to .md or .csv")] = None,
    dry_run: Annotated[bool, typer.Option(help="List what would be evaluated, then exit")] = False,
):
    """Eval every checkpoint under CKPT_DIR into one store, then report.

    Each model's ``model_name``/``seed``/θ* are read from its ``train_meta.json``
    (θ* falls back to ``sweep.json``), so one command turns a directory of trained
    runs into a full cross-model comparison. Dirs with no checkpoint are skipped.
    """
    from benchmarking.runner import evaluate

    tile_metrics = () if (tile_metric and tile_metric[0].lower() in ("none", "off")) \
        else tuple(tile_metric if tile_metric is not None else ["apls"])
    report_metrics = list(metric or (["f1", "iou", "apls"] if tile_metrics else ["f1", "iou"]))

    specs = _discover_ckpt_runs(Path(ckpt_dir))
    if not specs:
        raise typer.BadParameter(f"no model dirs found under {ckpt_dir}")

    try:
        from benchmarking.store import load_runs
        existing = load_runs(store_dir)
        seen = set(zip(existing["model_name"], existing["seed"])) if not existing.empty else set()
    except FileNotFoundError:
        seen = set()

    typer.echo(f"discovered {len(specs)} model dir(s) under {ckpt_dir}:")
    todo = []
    for s in specs:
        key = (s["model_name"], s["seed"])
        if s["checkpoint"] is None:
            typer.secho(f"  SKIP {s['model_name']:24} (no checkpoint in {s['run_dir'].name})",
                        fg=typer.colors.YELLOW)
        elif skip_existing and key in seen:
            typer.secho(f"  SKIP {s['model_name']:24} (already in store, seed={s['seed']})",
                        fg=typer.colors.BLUE)
        else:
            thr = s["threshold"]
            typer.echo(f"  EVAL {s['model_name']:24} seed={s['seed']} "
                       f"θ={thr if thr is not None else 'ckpt-default'}  {s['checkpoint']}")
            todo.append(s)

    if dry_run:
        typer.echo("\n--dry-run: nothing evaluated.")
        return
    if not todo:
        typer.echo("\nnothing to evaluate.")
    for i, s in enumerate(todo, 1):
        typer.secho(f"\n[{i}/{len(todo)}] {s['model_name']}", fg=typer.colors.GREEN)
        evaluate(
            dataset_dir=dataset_dir, checkpoint=s["checkpoint"],
            model_name=s["model_name"], seed=s["seed"], store_dir=store_dir,
            split=split, model=model, cell_m=cell_m, chip_px=chip_px,
            batch_size=batch_size, label_source=label_source,
            exp_tag=f"loss_{s['model_name']}",
            config_yaml_path=s["config_yaml"], tile_metrics=tile_metrics,
            device=device, threshold=s["threshold"], max_tiles=max_tiles,
        )

    if report:
        typer.echo("\n" + "=" * 70)
        _run_report(store_dir, report_metrics, aggregation, n_boot=2000, out=out)


@app.command()
def compare(
    store_dir: Annotated[Path, typer.Option(help="Sharded store dir")],
    model_a: Annotated[str, typer.Option(help="First model_name")],
    model_b: Annotated[str, typer.Option(help="Second model_name")],
    metric: Annotated[str, typer.Option(help="Chip metric (iou/f1/...) or tile metric (e.g. apls)")] = "f1",
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples")] = 2000,
    seed: Annotated[int, typer.Option(help="Bootstrap RNG seed")] = 0,
    force: Annotated[bool, typer.Option(help="Compare even across different GT")] = False,
):
    """Paired bootstrap 95% CI + Wilcoxon signed-rank between two models.

    Seed-averages each unit first (one value per model/unit), then pairs on it.
    """
    import numpy as np

    from benchmarking.stats import bootstrap_paired_diff, wilcoxon_paired

    _guard_comparable(store_dir, [model_a, model_b], force)
    df, unit = _load_metric_table(store_dir, metric)
    avg = df.groupby(["model_name", "chip_id"], as_index=False)[metric].mean()
    boot = bootstrap_paired_diff(
        avg, model_a, model_b, metric=metric, n_boot=n_boot, rng=np.random.default_rng(seed)
    )
    wil = wilcoxon_paired(avg, model_a, model_b, metric=metric)
    verdict = "significant" if wil["p_value"] < 0.05 else "not significant"
    typer.echo(f"{model_a} vs {model_b} on seed-averaged per-{unit} {metric}:")
    typer.echo(
        f"  bootstrap  diff = {boot['diff_mean']:+.4f}  "
        f"95% CI [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]  n_pairs = {boot['n_pairs']}"
    )
    typer.echo(
        f"  wilcoxon   W = {wil['statistic']:.0f}  p = {wil['p_value']:.4g}  n = {wil['n_pairs']}"
    )
    typer.echo(f"  -> difference is {verdict} at alpha = 0.05")


@app.command()
def variance(
    store_dir: Annotated[Path, typer.Option(help="Sharded store dir")],
    model_name: Annotated[str, typer.Option(help="Model to summarise across seeds")],
    metric: Annotated[str, typer.Option(help="Chip metric or tile metric")] = "iou",
    aggregation: Annotated[str, typer.Option(help="micro (pool counts) or macro (mean per unit)")] = "micro",
):
    """Cross-seed mean +/- std and 95% CI for one model (training instability)."""
    from benchmarking.stats import _MICRO_DERIVABLE, cross_seed_ci

    df, unit = _load_metric_table(store_dir, metric)
    agg = aggregation if metric in _MICRO_DERIVABLE and "tp" in df.columns else "macro"
    out = cross_seed_ci(df, {"model_name": model_name}, metric=metric, aggregation=agg)
    typer.echo(f"{model_name}: {metric} across {out['n_seeds']} seed(s) [{agg}, per-{unit}]")
    typer.echo(
        f"  mean {out['mean']:.4f} +/- {out['std']:.4f}  "
        f"95% CI [{out['ci_lo']:.4f}, {out['ci_hi']:.4f}]"
    )
    typer.echo(f"  per-seed: {[round(v, 4) for v in out['per_seed_values']]}")


@app.command()
def report(
    store_dir: Annotated[Path, typer.Option(help="Sharded store dir")],
    metric: Annotated[List[str], typer.Option(help="Metric(s), repeatable; chip or tile level")] = None,
    aggregation: Annotated[str, typer.Option(help="micro or macro cross-seed aggregation")] = "micro",
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples for pairwise")] = 2000,
    out: Annotated[Optional[Path], typer.Option(help="Write the report to .md or .csv as well")] = None,
):
    """Per-model cross-seed mean +/- std + all pairwise comparisons, per metric."""
    _run_report(store_dir, list(metric or ("iou", "f1")), aggregation, n_boot, out)


def _run_report(store_dir, metrics, aggregation, n_boot, out):
    """Shared body of the ``report`` command; also chained from ``eval-dir``."""
    import numpy as np
    import pandas as pd

    from benchmarking.stats import (
        _MICRO_DERIVABLE,
        bootstrap_paired_diff,
        cross_seed_ci,
        wilcoxon_paired,
    )

    md_parts, csv_rows = [], []

    for met in metrics:
        df, unit = _load_metric_table(store_dir, met)
        models = sorted(df["model_name"].unique())
        if len(models) > 1:
            # Report only warns (force=True): it summarises whatever exists.
            _guard_comparable(store_dir, models, force=True)

        agg = aggregation if met in _MICRO_DERIVABLE and "tp" in df.columns else "macro"
        typer.echo(f"\n== per-model {met} (mean +/- std across seeds, {agg}, per-{unit}) ==")
        summary_rows = []
        for m in models:
            try:
                o = cross_seed_ci(df, {"model_name": m}, metric=met, aggregation=agg)
                typer.echo(f"  {m:24} {o['mean']:.4f} +/- {o['std']:.4f}  (n_seeds={o['n_seeds']})")
                summary_rows.append([m, f"{o['mean']:.4f}", f"{o['std']:.4f}", o["n_seeds"]])
                csv_rows.append({"metric": met, "model": m, "mean": o["mean"],
                                 "std": o["std"], "n_seeds": o["n_seeds"]})
            except ValueError as e:
                typer.echo(f"  {m:24} <{e}>")

        pair_rows = []
        if len(models) > 1:
            avg = df.groupby(["model_name", "chip_id"], as_index=False)[met].mean()
            typer.echo(f"== pairwise (seed-averaged per-{unit} {met}; * = p<0.05) ==")
            for a, b in itertools.combinations(models, 2):
                boot = bootstrap_paired_diff(
                    avg, a, b, metric=met, n_boot=n_boot, rng=np.random.default_rng(0)
                )
                wil = wilcoxon_paired(avg, a, b, metric=met)
                sig = "*" if wil["p_value"] < 0.05 else " "
                typer.echo(
                    f"  {a} - {b}: diff {boot['diff_mean']:+.4f} "
                    f"CI[{boot['ci_lo']:+.4f},{boot['ci_hi']:+.4f}] p={wil['p_value']:.3g} {sig}"
                )
                pair_rows.append([a, b, f"{boot['diff_mean']:+.4f}",
                                  f"[{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]",
                                  f"{wil['p_value']:.3g}", sig.strip() or ""])

        md_parts.append(f"## {met} ({agg}, per-{unit})\n\n"
                        + _md_table(["model", "mean", "std", "n_seeds"], summary_rows))
        if pair_rows:
            md_parts.append(_md_table(["model A", "model B", "diff", "95% CI", "p", "sig"],
                                      pair_rows))

    if out is not None:
        out = Path(out)
        if out.suffix == ".csv":
            pd.DataFrame(csv_rows).to_csv(out, index=False)
        else:
            out.write_text("# Benchmark report\n\n" + "\n\n".join(md_parts) + "\n")
        typer.echo(f"\nwrote {out}")


def cli_main():
    app()


if __name__ == "__main__":
    cli_main()
