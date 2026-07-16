"""Benchmarking CLI -- eval a checkpoint to the store, then compare / summarise.
    eval      score a trained checkpoint over footprint chips -> sharded store
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
):
    """Score a checkpoint over the split's footprint chips -> the sharded store."""
    from benchmarking.runner import evaluate

    evaluate(
        dataset_dir=dataset_dir, checkpoint=checkpoint, model_name=model_name,
        seed=seed, store_dir=store_dir, split=split, model=model, cell_m=cell_m,
        chip_px=chip_px, batch_size=batch_size, mask_source=mask_source,
        mask_dirname=mask_dirname, sen2sr_dir=sen2sr_dir, config_yaml_path=config_yaml,
        exp_tag=exp_tag, label_source=label_source,
        tile_metrics=tuple(tile_metric or ()), check=check, device=device,
        threshold=threshold,
    )


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
    import numpy as np
    import pandas as pd

    from benchmarking.stats import (
        _MICRO_DERIVABLE,
        bootstrap_paired_diff,
        cross_seed_ci,
        wilcoxon_paired,
    )

    metrics = list(metric or ("iou", "f1"))
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
