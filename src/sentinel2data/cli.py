from enum import Enum
from pathlib import Path
from typing import Annotated, Optional
import typer

from sentinel2data.processor.manager import DatasetManager
from sentinel2data.processor.mask_generator import RoadVectorExtractor
from sentinel2data.viz import visualize_classification


class Backdrop(str, Enum):
    satellite = "satellite"
    mask = "mask"
    none = "none"

app = typer.Typer(help="S2-ROSA Dataset Pipeline")

DATASET_HELP = "Base path to the S2-ROSA dataset directory"
DatasetDir = Annotated[Path, typer.Option(help=DATASET_HELP)]


@app.command()
def build(
    dataset_dir: DatasetDir,
    roads: Annotated[Path, typer.Option(help="Combined roads GeoParquet from the `roads` command")],
    buffer_m: Annotated[int, typer.Option(help="Fallback road buffer (metres) for classes without a per-tier width")] = 10,
):
    """Scan imagery/, generate masks_raster/, masks_graph/, metadata.parquet and splits/."""
    manager = DatasetManager(
        dataset_dir=dataset_dir,
        roads_parquet_path=roads,
        buffer_m=buffer_m,
    )
    manager.build_products()


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
    """Extract major + medium scale roads from CDNGI and/or Overture into one layer."""
    if cdngi is None and overture is None:
        raise typer.BadParameter("Provide at least one of --cdngi or --overture.")

    extractor = RoadVectorExtractor(
        out_path=out, cdngi_path=cdngi, overture_path=overture
    )
    extractor.build()


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


if __name__ == "__main__":
    app()
