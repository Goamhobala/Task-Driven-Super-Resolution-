from enum import Enum
from pathlib import Path
from typing import Annotated, Optional, Tuple
import typer
import yaml

from sentinel2data.generator import (
    RoadVectorExtractor,
    make_v1rosa_pipeline,
    make_v2rosa_pipeline,
)
from sentinel2data.viz import visualize_classification, visualize_rosav2


class Backdrop(str, Enum):
    satellite = "satellite"
    mask = "mask"
    none = "none"


class Variant(str, Enum):
    """The dataset format to generate."""

    V1ROSA = "V1ROSA"  # patch-window index (masks generated in place, one row / block)
    V2ROSA = "V2ROSA"  # cut tiles (physical NxN image+mask COGs, one row / tile)


app = typer.Typer(help="S2-ROSA Dataset Pipeline")

DATASET_HELP = "Base path to the S2-ROSA dataset directory"
DatasetDir = Annotated[Path, typer.Option(help=DATASET_HELP)]


@app.callback()
def _main(
    ctx: typer.Context,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            help="YAML of per-command option defaults; top-level keys are command "
            "names (`generate`, `roads`, `norm-stats`, `visualize`, `visualize-v2`). "
            "Explicit CLI flags still override. See src/sentinel2data/configs/.",
        ),
    ] = None,
):
    """S2-ROSA Dataset Pipeline. ``--config FILE`` pre-fills each command's options."""
    if config is not None:
        # Click resolves each subcommand's defaults from ctx.default_map[<command>].
        loaded = yaml.safe_load(config.read_text()) or {}
        ctx.default_map = {**(ctx.default_map or {}), **loaded}


@app.command()
def generate(
    variant: Annotated[
        Variant,
        typer.Option(help="Dataset variant: V1ROSA (patch-window index) or V2ROSA (cut tiles)"),
    ],
    roads: Annotated[Path, typer.Option(help="Combined roads GeoParquet from the `roads` command")],
    dataset_dir: Annotated[
        Optional[Path],
        typer.Option(help="[V1ROSA] dataset root; scans <dir>/imagery, writes masks + metadata in place"),
    ] = None,
    imagery_dir: Annotated[
        Optional[Path], typer.Option(help="[V2ROSA] directory of source satellite COGs")
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="[V2ROSA] output tiled-dataset directory")
    ] = None,
    biome_parquet: Annotated[
        Optional[Path],
        typer.Option(help="[V2ROSA] NVM2024 biome GeoParquet (scripts/biome.py convert); tiles tagged 'Unknown' if omitted"),
    ] = None,
    tile_size: Annotated[int, typer.Option(help="[V2ROSA] tile edge in pixels (kept tiles are exactly this)")] = 512,
    patch_size: Annotated[int, typer.Option(help="[V2ROSA] COG block size + dataloader crop edge")] = 256,
    buffer_m: Annotated[
        Optional[int],
        typer.Option(help="Fallback road buffer (metres); default 10 for V1ROSA, 5 for V2ROSA"),
    ] = None,
    empty_keep_ratio: Annotated[
        float,
        typer.Option(help="[V2ROSA] fraction of empty (no-road) tiles kept, same for all splits (1.0=keep all, 0.0=only road tiles)"),
    ] = 1.0,
    tile_seed: Annotated[int, typer.Option(help="[V2ROSA] seed for empty-tile subsampling")] = 42,
):
    """Generate a dataset variant: --variant V1ROSA (in-place patch index) or V2ROSA (cut tiles)."""
    if variant is Variant.V1ROSA:
        if dataset_dir is None:
            raise typer.BadParameter("V1ROSA requires --dataset-dir.")
        make_v1rosa_pipeline(
            dataset_dir=dataset_dir,
            roads_parquet_path=roads,
            buffer_m=10 if buffer_m is None else buffer_m,
        ).run()
    else:  # V2ROSA
        if imagery_dir is None or output_dir is None:
            raise typer.BadParameter("V2ROSA requires --imagery-dir and --output-dir.")
        make_v2rosa_pipeline(
            imagery_dir=imagery_dir,
            output_dir=output_dir,
            roads_parquet_path=roads,
            biome_parquet_path=biome_parquet,
            tile_size=tile_size,
            patch_size=patch_size,
            buffer_m=5 if buffer_m is None else buffer_m,
            empty_keep_ratio=empty_keep_ratio,
            tile_seed=tile_seed,
        ).run()


@app.command()
def roads(
    out: Annotated[Path, typer.Option(help="Output path (.parquet GeoParquet or .gpkg)")],
    cdngi: Annotated[
        Optional[Path],
        typer.Option(help="CDNGI GeoPackage file, or a directory of province *.gpkg"),
    ] = None,
    overture: Annotated[
        Optional[Path], typer.Option(help="Overture roads GeoParquet")
    ] = None,
):
    """Extract major + medium scale roads from CDNGI or Overture (exactly one) into one layer."""
    if (cdngi is None) == (overture is None):
        raise typer.BadParameter("Provide exactly one of --cdngi or --overture.")

    RoadVectorExtractor.from_paths(
        out_path=out, cdngi_path=cdngi, overture_path=overture
    ).build()


@app.command()
def visualize(
    dataset_dir: DatasetDir,
    zone_name: Annotated[Optional[str], typer.Option(help="Zone (COG stem) to plot")] = None,
    tile_id: Annotated[Optional[int], typer.Option(help="Tile id to plot")] = None,
    backdrop: Annotated[
        Backdrop, typer.Option(help="Backdrop under the classification overlay")
    ] = Backdrop.satellite,
):
    """Plot one tile's patches coloured by urbanisation classification."""
    if (zone_name is None) == (tile_id is None):
        raise typer.BadParameter("Provide exactly one of --zone-name or --tile-id.")

    metadata_path = dataset_dir / "metadata.parquet"
    label = zone_name if zone_name is not None else f"tile{tile_id}"
    out_plot_path = (
        dataset_dir / "classification_plots" / f"{label}_{backdrop.value}_classification.png"
    )

    visualize_classification(
        metadata_path=metadata_path,
        out_plot_path=out_plot_path,
        zone_name=zone_name,
        tile_id=tile_id,
        dataset_dir=dataset_dir,
        backdrop=backdrop.value,
    )


@app.command()
def visualize_v2(
    dataset_dir: DatasetDir,
    out_dir: Annotated[
        Optional[Path],
        typer.Option(help="Output folder (default <dataset_dir>/visualisation)"),
    ] = None,
    limit: Annotated[
        Optional[int], typer.Option(help="Max images rendered per split (default: all)")
    ] = None,
):
    """Render side-by-side RGB | RGB-enhanced | road-mask PNGs for a ROSAV2 dataset
    into <dataset_dir>/visualisation/{train,val,test}/."""
    visualize_rosav2(dataset_dir=dataset_dir, out_dir=out_dir, limit=limit)


@app.command()
def norm_stats(
    dataset_dir: DatasetDir,
    out: Annotated[
        Optional[Path],
        typer.Option(help="Output config YAML (default <dataset_dir>/norm_stats.yaml)"),
    ] = None,
    exclude_zero: Annotated[
        bool,
        typer.Option(help="When a COG sets no nodata, drop 0-valued pixels from the stats"),
    ] = True,
    sar_clip: Annotated[
        Optional[Tuple[float, float]],
        typer.Option(help="Clip SAR bands (VV/VH asc+desc) to LO HI before stats"),
    ] = None,
):
    """Per-band mean/std over the TRAIN split (frozen, no leakage) -> a LightningCLI
    config (data.norm_mean / data.norm_std), merged into the unet config so train and
    val/test/predict z-score with the same frozen stats."""
    # Lazy import: pulls torch/lightning via the dataset stack, unwanted by other commands.
    from sentinel2data.dataset.compute_norm_stats import (
        compute,
        print_table,
        write_stats_yaml,
    )

    out = out or dataset_dir / "norm_stats.yaml"
    names, mean, std = compute(dataset_dir, exclude_zero=exclude_zero, sar_clip=sar_clip)
    saved = write_stats_yaml(out, mean, std)
    typer.echo(f"\nSaved {saved}")
    print_table(names, mean, std)


@app.command()
def summary(
    dataset_dir: DatasetDir,
    out: Annotated[
        Optional[Path],
        typer.Option(help="Output summary YAML (default <dataset_dir>/dataset_summary.yaml)"),
    ] = None,
):
    """Per-split biome + urbanisation ratios and road-density stats from metadata.parquet."""
    from sentinel2data.generator.summary import summarise_metadata

    metadata_path = dataset_dir / "metadata.parquet"
    if not metadata_path.exists():
        raise typer.BadParameter(f"metadata.parquet not found under {dataset_dir}")
    summarise_metadata(metadata_path, out)


if __name__ == "__main__":
    app()
