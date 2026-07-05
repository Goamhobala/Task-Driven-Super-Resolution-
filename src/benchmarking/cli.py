"""Benchmarking CLI -- eval a checkpoint to the store, then compare / summarise.
    eval      score a trained checkpoint over non-overlapping chips -> store
    compare   paired bootstrap CI + Wilcoxon signed-rank between two models
    variance  cross-seed mean +/- std + 95% CI for one model (training instability)
    report    per-model mean +/- std + all pairwise comparisons
"""
import itertools
from pathlib import Path
from typing import Annotated, Optional
import typer
import yaml

app = typer.Typer(help="Road-segmentation benchmarking: eval + statistical comparison")


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


@app.command(name="eval")
def run_eval(
    dataset_dir: Annotated[Path, typer.Option(help="ROSA dataset root (has splits/<split>.csv)")],
    checkpoint: Annotated[Path, typer.Option(help="Trained UNet .ckpt to evaluate")],
    model_name: Annotated[str, typer.Option(help="Config identifier the stats pair/group on")],
    seed: Annotated[int, typer.Option(help="Training seed (for cross-seed CIs)")],
    store_dir: Annotated[Path, typer.Option(help="Output store dir (runs + chip_metrics parquet)")],
    split: Annotated[str, typer.Option(help="Split to evaluate")] = "test",
    chip_size: Annotated[int, typer.Option(help="Non-overlapping chip edge (px)")] = 256,
    model: Annotated[str, typer.Option(help="Model family to load")] = "unet",
):
    """Score a checkpoint over the split's non-overlapping chips -> the store."""
    from benchmarking.runner import evaluate

    evaluate(
        dataset_dir=dataset_dir, checkpoint=checkpoint, model_name=model_name,
        seed=seed, store_dir=store_dir, split=split, chip_size=chip_size, model=model,
    )


@app.command()
def compare(
    store_dir: Annotated[Path, typer.Option(help="Store dir with chip_metrics.parquet")],
    model_a: Annotated[str, typer.Option(help="First model_name")],
    model_b: Annotated[str, typer.Option(help="Second model_name")],
    metric: Annotated[str, typer.Option(help="Metric to compare (iou/f1/precision/recall)")] = "f1",
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples")] = 2000,
    seed: Annotated[int, typer.Option(help="Bootstrap RNG seed")] = 0,
):
    """Paired bootstrap 95% CI + Wilcoxon signed-rank between two models.

    Seed-averages each chip first (one value per model/chip), then pairs on chip_id.
    """
    import numpy as np

    from benchmarking.stats import bootstrap_paired_diff, wilcoxon_paired
    from benchmarking.store import load_chips

    df = load_chips(store_dir)
    avg = df.groupby(["model_name", "chip_id"], as_index=False)[metric].mean()
    boot = bootstrap_paired_diff(
        avg, model_a, model_b, metric=metric, n_boot=n_boot, rng=np.random.default_rng(seed)
    )
    wil = wilcoxon_paired(avg, model_a, model_b, metric=metric)
    verdict = "significant" if wil["p_value"] < 0.05 else "not significant"
    typer.echo(f"{model_a} vs {model_b} on seed-averaged per-chip {metric}:")
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
    store_dir: Annotated[Path, typer.Option(help="Store dir with chip_metrics.parquet")],
    model_name: Annotated[str, typer.Option(help="Model to summarise across seeds")],
    metric: Annotated[str, typer.Option(help="Metric (iou/f1/precision/recall/accuracy)")] = "iou",
    aggregation: Annotated[str, typer.Option(help="micro (pool counts) or macro (mean per-chip)")] = "micro",
):
    """Cross-seed mean +/- std and 95% CI for one model (training instability)."""
    from benchmarking.stats import cross_seed_ci
    from benchmarking.store import load_chips

    df = load_chips(store_dir)
    out = cross_seed_ci(df, {"model_name": model_name}, metric=metric, aggregation=aggregation)
    typer.echo(f"{model_name}: {metric} across {out['n_seeds']} seed(s) [{aggregation}]")
    typer.echo(
        f"  mean {out['mean']:.4f} +/- {out['std']:.4f}  "
        f"95% CI [{out['ci_lo']:.4f}, {out['ci_hi']:.4f}]"
    )
    typer.echo(f"  per-seed: {[round(v, 4) for v in out['per_seed_values']]}")


@app.command()
def report(
    store_dir: Annotated[Path, typer.Option(help="Store dir with chip_metrics.parquet")],
    metric: Annotated[str, typer.Option(help="Metric for the summary")] = "iou",
    aggregation: Annotated[str, typer.Option(help="micro or macro cross-seed aggregation")] = "micro",
    n_boot: Annotated[int, typer.Option(help="Bootstrap resamples for pairwise")] = 2000,
):
    """Per-model cross-seed mean +/- std + all pairwise Wilcoxon/bootstrap comparisons."""
    import numpy as np

    from benchmarking.stats import bootstrap_paired_diff, cross_seed_ci, wilcoxon_paired
    from benchmarking.store import load_chips

    df = load_chips(store_dir)
    models = sorted(df["model_name"].unique())

    typer.echo(f"== per-model {metric} (mean +/- std across seeds, {aggregation}) ==")
    for m in models:
        try:
            o = cross_seed_ci(df, {"model_name": m}, metric=metric, aggregation=aggregation)
            typer.echo(f"  {m:24} {o['mean']:.4f} +/- {o['std']:.4f}  (n_seeds={o['n_seeds']})")
        except ValueError as e:
            typer.echo(f"  {m:24} <{e}>")

    if len(models) > 1:
        avg = df.groupby(["model_name", "chip_id"], as_index=False)[metric].mean()
        typer.echo(f"\n== pairwise (seed-averaged per-chip {metric}; * = p<0.05) ==")
        for a, b in itertools.combinations(models, 2):
            boot = bootstrap_paired_diff(
                avg, a, b, metric=metric, n_boot=n_boot, rng=np.random.default_rng(0)
            )
            wil = wilcoxon_paired(avg, a, b, metric=metric)
            sig = "*" if wil["p_value"] < 0.05 else " "
            typer.echo(
                f"  {a} - {b}: diff {boot['diff_mean']:+.4f} "
                f"CI[{boot['ci_lo']:+.4f},{boot['ci_hi']:+.4f}] p={wil['p_value']:.3g} {sig}"
            )


def cli_main():
    app()


if __name__ == "__main__":
    cli_main()
