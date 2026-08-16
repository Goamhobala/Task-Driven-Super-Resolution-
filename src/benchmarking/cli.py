"""Benchmarking CLI -- eval a checkpoint to the store, then compare / summarise.
    eval      score a trained checkpoint over footprint chips -> sharded store
    sweep     score a checkpoint at every θ in a grid off ONE inference pass ->
              sweep.json (θ* selection / sensitivity curve; store never written)
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
# whose runs disagree on any of these are not pixel-comparable. ``stratum`` is
# in here because an Urban-only run and a whole-split run cover different
# geography — their pixel metrics are not the same quantity.
_GT_KEYS = ("gt_res_m", "label_source", "mask_source", "mask_dirname",
            "dataset_split", "cell_m", "stratum")


def _with_stratum(df, store_dir, stratum_col: Optional[str]):
    """Ensure every row has a ``stratum``, filling only the ones that lack it.

    A store can be MIXED: shards written before runner stamped a per-chip
    stratum have none, newer shards do. Treating that all-or-nothing would
    silently drop the older arms from a stratified report, so fill per-row --
    the tiles table never carries the column and always takes the join path.
    """
    from benchmarking import strata as _s
    from benchmarking.store import load_runs

    col = stratum_col or _s.DEFAULT_COL
    have = df["stratum"] if "stratum" in df.columns else None
    missing = have.isna() | have.astype(str).eq("") if have is not None else None
    if have is not None and not missing.any():
        return df

    runs = load_runs(store_dir)
    pairs = {(r["dataset_dir"], r["dataset_split"]) for _, r in runs.iterrows()}
    if len(pairs) != 1:
        raise typer.BadParameter(
            "--stratum needs one dataset/split in the store to join against, "
            f"found {sorted(pairs)}"
        )
    dsdir, split = pairs.pop()
    joined = _s.annotate_chips(df.drop(columns=["stratum"], errors="ignore"),
                               dsdir, split, col)["stratum"]
    out = df.copy()
    out["stratum"] = joined if have is None else have.where(~missing, joined)
    return out


def _apply_stratum(df, store_dir, stratum: Optional[str], stratum_col: Optional[str]):
    """Restrict an already-loaded metric table to one stratum.

    Prefers the per-chip ``stratum`` column written by newer runs; falls back to
    a tile_id -> stratum join against the split CSV named in the runs table, so
    stores written before that column existed can still be sliced. Returns
    (df, resolved_stratum_or_None).
    """
    if not stratum:
        return df, None
    df = _with_stratum(df, store_dir, stratum_col)
    from benchmarking import strata as _s

    choices = sorted(c for c in df["stratum"].dropna().astype(str).unique() if c)
    try:
        resolved = _s.resolve(stratum, choices)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    out = df[df["stratum"].astype(str) == resolved]
    if out.empty:
        raise typer.BadParameter(f"stratum {resolved!r} matched no rows in the store")
    return out, resolved


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
def _parse_radii(spec):
    """'3' -> 3.0;  '1,2,3,4,5' -> [1.0..5.0];  None -> None."""
    if spec is None:
        return None
    vals = [float(x) for x in str(spec).replace(" ", ",").split(",") if x]
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


def _parse_theta_grid(spec: str) -> list[float]:
    """'0.05:0.95:0.05' -> [0.05, 0.10, ..., 0.95] (stop-inclusive)."""
    try:
        lo, hi, step = (float(x) for x in spec.split(":"))
    except ValueError as e:
        raise typer.BadParameter(
            f"--thresholds must be start:stop:step, got {spec!r}") from e
    if not (0.0 < lo <= hi < 1.0) or step <= 0:
        raise typer.BadParameter(
            f"--thresholds needs 0 < start <= stop < 1 and step > 0, got {spec!r}")
    n = int(round((hi - lo) / step))
    # round() kills float-accumulation dust so the JSON keys read "0.15",
    # not "0.15000000000000002".
    return [round(lo + i * step, 10) for i in range(n + 1)]


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
    stratum: Annotated[Optional[str], typer.Option(help="Score only this stratum, e.g. Urban | PeriUrban | Rural (case/dash-insensitive)")] = None,
    stratum_col: Annotated[Optional[str], typer.Option(help="Split-CSV column the stratum comes from")] = None,
    ap_bins: Annotated[Optional[int], typer.Option(help="Add per-chip Average Precision (AUPRC) scored from the probability map over this many thresholds spanning [0,1] (101 = 0.01 resolution). Unlike an AP derived from a theta sweep, coverage is complete by construction. Step-wise sum, per Davis & Goadrich.")] = None,
    buffer_px: Annotated[Optional[str], typer.Option(help="Buffered precision/recall/F1 tolerance(s) in px: '3', or a comma list '1,2,3,4,5' for a tolerance sweep (columns gain an _r<N> suffix). Several radii share one distance transform, so the sweep is nearly free. 3 px = 7.5 m at 2.5 m GSD.")] = None,
    wandb_meta: Annotated[Optional[Path], typer.Option(help="train_meta.json with a `wandb` block: resume that run and push the bench metrics (incl. APLS) to its summary")] = None,
):
    """Score a checkpoint over the split's footprint chips -> the sharded store.

    ``--stratum Urban`` restricts scoring to that stratum. To break down a store
    you have ALREADY scored over the whole split, prefer ``report --stratum`` --
    same arithmetic, no re-inference.
    """
    from benchmarking.runner import evaluate

    run_id = evaluate(
        dataset_dir=dataset_dir, checkpoint=checkpoint, model_name=model_name,
        seed=seed, store_dir=store_dir, split=split, model=model, cell_m=cell_m,
        chip_px=chip_px, batch_size=batch_size, mask_source=mask_source,
        mask_dirname=mask_dirname, sen2sr_dir=sen2sr_dir, config_yaml_path=config_yaml,
        exp_tag=exp_tag, label_source=label_source,
        tile_metrics=tuple(tile_metric or ()), check=check, device=device,
        threshold=threshold, max_tiles=max_tiles,
        stratum=stratum, stratum_col=stratum_col,
        buffer_px=_parse_radii(buffer_px), ap_bins=ap_bins,
    )
    if wandb_meta is not None:
        _push_bench_to_wandb(wandb_meta, run_id, store_dir, split)


@app.command(name="sweep")
def run_sweep(
    dataset_dir: Annotated[Path, typer.Option(help="ROSA dataset root (has splits/<split>.csv)")],
    checkpoint: Annotated[Path, typer.Option(help="Trained .ckpt to sweep")],
    model_name: Annotated[str, typer.Option(help="Config identifier, recorded as `run` in the JSON")],
    split: Annotated[str, typer.Option(help="Split to sweep. Training-side splits stamp purpose=selection; test stamps purpose=sensitivity (reporting only — NEVER selection)")] = "val",
    model: Annotated[str, typer.Option(help="Model family loader: unet | sr")] = "unet",
    seed: Annotated[int, typer.Option(help="Training seed (stamped on the per-chip rows)")] = 0,
    cell_m: Annotated[float, typer.Option(help="Footprint cell edge in metres (chip unit)")] = 2560.0,
    chip_px: Annotated[Optional[int], typer.Option(help="Override: cell edge in native px (bypasses the transform)")] = None,
    batch_size: Annotated[int, typer.Option(help="Chips per forward pass (unet family)")] = 8,
    mask_source: Annotated[Optional[str], typer.Option(help="sr GT: graph (masks_graph parquet) | raster (<split>/<mask-dirname>)")] = None,
    mask_dirname: Annotated[Optional[str], typer.Option(help="unet: remap the CSV mask dir; sr raster: HR mask dir")] = None,
    sen2sr_dir: Annotated[Optional[Path], typer.Option(help="Override the checkpoint's baked-in SR weights dir")] = None,
    config_yaml: Annotated[Optional[Path], typer.Option(help="Training config (e.g. best_params.yaml), recorded only")] = None,
    check: Annotated[str, typer.Option(help="tp+fn-vs-mask invariant: first | all | off")] = "first",
    device: Annotated[Optional[str], typer.Option(help="cuda | cpu (default: auto)")] = None,
    max_tiles: Annotated[Optional[int], typer.Option(help="Score only the first N tiles of the split (quick local smoke)")] = None,
    thresholds: Annotated[str, typer.Option(help="θ grid as start:stop:step, stop-inclusive")] = "0.05:0.95:0.05",
    criterion: Annotated[str, typer.Option(help="Argmax criterion for best_threshold: iou | f1 (global pooled counts)")] = "iou",
    buffer_px: Annotated[Optional[str], typer.Option(help="Buffered-F1 tolerance(s) in px, e.g. '1,2,3,4,5' — adds buffered_* columns to every θ entry")] = None,
    out: Annotated[Optional[Path], typer.Option(help="Output JSON (default: <ckpt run dir>/sweep.json)")] = None,
):
    """Score one checkpoint at every θ in the grid off ONE inference pass -> sweep.json.

    Writes NOTHING to the store (that is ``eval``'s job, once, at θ*). Per θ the
    JSON records GLOBAL pooled-count IoU/F1 (tp/fp/fn summed over chips — the
    same accumulation semantics as the training-time BinaryJaccardIndex), plus
    micro-pooled buffered metrics when --buffer-px is given. ``best_threshold``
    is the --criterion argmax; ``purpose`` derives from the split so a test
    sweep can never be mistaken for a selection artifact.
    """
    from benchmarking.runner import evaluate
    from benchmarking.stats import _micro_metric_from_counts

    if criterion not in ("iou", "f1"):
        raise typer.BadParameter(f"--criterion must be iou or f1, got {criterion!r}")
    grid = _parse_theta_grid(thresholds)

    per_theta = evaluate(
        dataset_dir=dataset_dir, checkpoint=checkpoint, model_name=model_name,
        seed=seed, store_dir=None, split=split, model=model, cell_m=cell_m,
        chip_px=chip_px, batch_size=batch_size, mask_source=mask_source,
        mask_dirname=mask_dirname, sen2sr_dir=sen2sr_dir, config_yaml_path=config_yaml,
        tile_metrics=(), check=check, device=device, max_tiles=max_tiles,
        sweep_thresholds=grid, buffer_px=_parse_radii(buffer_px),
    )

    curve = {}
    for t, chips in sorted(per_theta.items()):
        entry = {"iou": _micro_metric_from_counts(chips, "iou"),
                 "f1": _micro_metric_from_counts(chips, "f1")}
        for col in sorted(c for c in chips.columns if c.startswith("buffered_")):
            entry[col] = _micro_metric_from_counts(chips, col)
        curve[str(t)] = entry

    scored = {t: v[criterion] for t, v in curve.items() if v[criterion] == v[criterion]}
    if not scored:
        raise typer.BadParameter(
            f"{criterion} is NaN at every θ — empty split or degenerate GT?")
    # Ascending-θ iteration + max => ties resolve to the LOWEST θ.
    best_key = max(scored, key=lambda t: scored[t])
    best_t = float(best_key)
    if best_key in (str(grid[0]), str(grid[-1])):
        print(f"WARN: best_threshold={best_t} sits on the grid edge — widen --thresholds.")

    purpose = "sensitivity" if split == "test" else "selection"
    if out is None:
        ck = checkpoint.resolve()
        run_dir = ck.parent.parent if ck.parent.name == "checkpoints" else ck.parent
        out = run_dir / "sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"run": model_name, "split": split, "criterion": criterion,
         "purpose": purpose, "checkpoint": str(Path(checkpoint).resolve()),
         "best_threshold": best_t, "sweep": curve}, indent=1))
    print(f"θ* = {best_t}  ({criterion}={scored[best_key]:.4f} global, split={split}, "
          f"purpose={purpose})\nwrote {out}")


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
    stratum: Annotated[Optional[str], typer.Option(help="Score only this stratum, e.g. Urban | PeriUrban | Rural")] = None,
    stratum_col: Annotated[Optional[str], typer.Option(help="Split-CSV column the stratum comes from")] = None,
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
        # Keyed on stratum too: the same (model, seed) may legitimately appear
        # once per stratum, and those are different runs, not duplicates.
        if existing.empty:
            seen = set()
        else:
            strat = (existing["stratum"].fillna("") if "stratum" in existing.columns
                     else [""] * len(existing))
            seen = set(zip(existing["model_name"], existing["seed"], strat))
    except FileNotFoundError:
        seen = set()

    typer.echo(f"discovered {len(specs)} model dir(s) under {ckpt_dir}:"
               + (f"  [stratum={stratum}]" if stratum else ""))
    todo = []
    for s in specs:
        key = (s["model_name"], s["seed"], stratum or "")
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
            stratum=stratum, stratum_col=stratum_col,
        )

    if report:
        typer.echo("\n" + "=" * 70)
        _run_report(store_dir, report_metrics, aggregation, n_boot=2000, out=out,
                    stratum=stratum, stratum_col=stratum_col)


@app.command()
def compare(
    store_dir: Annotated[Path, typer.Option(help="Sharded store dir")],
    model_a: Annotated[str, typer.Option(help="First model_name")],
    model_b: Annotated[str, typer.Option(help="Second model_name")],
    metric: Annotated[str, typer.Option(help="Chip metric (iou/f1/...) or tile metric (e.g. apls)")] = "f1",
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples")] = 2000,
    seed: Annotated[int, typer.Option(help="Bootstrap RNG seed")] = 0,
    force: Annotated[bool, typer.Option(help="Compare even across different GT")] = False,
    stratum: Annotated[Optional[str], typer.Option(help="Compare only within this stratum, e.g. Urban | PeriUrban | Rural")] = None,
    stratum_col: Annotated[Optional[str], typer.Option(help="Split-CSV column the stratum comes from")] = None,
):
    """Paired bootstrap 95% CI + Wilcoxon signed-rank between two models.

    Seed-averages each unit first (one value per model/unit), then pairs on it.
    ``--stratum`` restricts the pairing to that stratum's chips.
    """
    import numpy as np

    from benchmarking.stats import bootstrap_paired_diff, wilcoxon_paired

    _guard_comparable(store_dir, [model_a, model_b], force)
    df, unit = _load_metric_table(store_dir, metric)
    df, resolved = _apply_stratum(df, store_dir, stratum, stratum_col)
    avg = df.groupby(["model_name", "chip_id"], as_index=False)[metric].mean()
    boot = bootstrap_paired_diff(
        avg, model_a, model_b, metric=metric, n_boot=n_boot, rng=np.random.default_rng(seed)
    )
    wil = wilcoxon_paired(avg, model_a, model_b, metric=metric)
    verdict = "significant" if wil["p_value"] < 0.05 else "not significant"
    typer.echo(f"{model_a} vs {model_b} on seed-averaged per-{unit} {metric}"
               + (f" [stratum={resolved}]" if resolved else "") + ":")
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
    need = {"tp", "fp", "fn", metric} if metric.startswith("buffered_") else {"tp"}
    agg = (aggregation if metric in _MICRO_DERIVABLE and need <= set(df.columns)
           else "macro")
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
    stratum: Annotated[Optional[str], typer.Option(help="Report only this stratum, e.g. Urban | PeriUrban | Rural. Slices the existing store — no re-inference.")] = None,
    stratum_col: Annotated[Optional[str], typer.Option(help="Split-CSV column the stratum comes from")] = None,
    by_stratum: Annotated[bool, typer.Option(help="Report every stratum in turn (overrides --stratum)")] = False,
):
    """Per-model cross-seed mean +/- std + all pairwise comparisons, per metric.

    ``--stratum Urban`` restricts the report to that stratum; ``--by-stratum``
    loops over all of them. Both slice the store you already have -- the chips
    are per-tile, so this is the same arithmetic as a stratified eval without
    re-running inference.
    """
    metrics = list(metric or ("iou", "f1"))
    if by_stratum:
        for s in _store_strata(store_dir, stratum_col):
            typer.secho(f"\n{'#' * 70}\n# stratum: {s}\n{'#' * 70}", fg=typer.colors.CYAN)
            _run_report(store_dir, metrics, aggregation, n_boot,
                        _stratum_out(out, s), stratum=s, stratum_col=stratum_col)
        return
    _run_report(store_dir, metrics, aggregation, n_boot, out,
                stratum=stratum, stratum_col=stratum_col)


def _store_strata(store_dir, stratum_col: Optional[str]) -> list[str]:
    """Strata present in a store, from the chips column or the split CSV join."""
    from benchmarking.store import load_chips

    chips = _with_stratum(load_chips(store_dir), store_dir, stratum_col)
    found = sorted(c for c in chips["stratum"].dropna().astype(str).unique() if c)
    if not found:
        raise typer.BadParameter("no strata found in this store")
    return found


def _stratum_out(out: Optional[Path], stratum: str) -> Optional[Path]:
    """Per-stratum output path, so --by-stratum doesn't overwrite one file."""
    if out is None:
        return None
    out = Path(out)
    return out.with_name(f"{out.stem}_{stratum}{out.suffix}")


def _run_report(store_dir, metrics, aggregation, n_boot, out,
                stratum=None, stratum_col=None):
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
    label = ""

    for met in metrics:
        df, unit = _load_metric_table(store_dir, met)
        df, resolved = _apply_stratum(df, store_dir, stratum, stratum_col)
        if resolved:
            label = f" [stratum={resolved}]"
        models = sorted(df["model_name"].unique())
        if len(models) > 1:
            # Report only warns (force=True): it summarises whatever exists.
            _guard_comparable(store_dir, models, force=True)

        # A buffered metric is micro-derivable only when its own ratio column is
        # present alongside the counts its denominator comes from; "tp" alone is
        # not enough. Falling back to macro is right for apls/cldice, which have
        # no per-chip denominator at all.
        from benchmarking.stats import is_micro_derivable

        if met.startswith("buffered_"):
            need = {"tp", "fp", "fn", met,
                    met.replace("buffered_f1", "buffered_precision"),
                    met.replace("buffered_f1", "buffered_recall")}
        else:
            need = {"tp"}
        agg = (aggregation if is_micro_derivable(met) and need <= set(df.columns)
               else "macro")
        n_units = df["chip_id"].nunique() if "chip_id" in df.columns else len(df)
        typer.echo(f"\n== per-model {met} (mean +/- std across seeds, {agg}, "
                   f"per-{unit}){label}  n_{unit}s={n_units} ==")
        summary_rows = []
        for m in models:
            try:
                o = cross_seed_ci(df, {"model_name": m}, metric=met, aggregation=agg)
                typer.echo(f"  {m:24} {o['mean']:.4f} +/- {o['std']:.4f}  (n_seeds={o['n_seeds']})")
                summary_rows.append([m, f"{o['mean']:.4f}", f"{o['std']:.4f}", o["n_seeds"]])
                csv_rows.append({"metric": met, "model": m, "mean": o["mean"],
                                 "std": o["std"], "n_seeds": o["n_seeds"],
                                 "stratum": resolved or "all"})
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

        md_parts.append(f"## {met} ({agg}, per-{unit}){label}\n\n"
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
