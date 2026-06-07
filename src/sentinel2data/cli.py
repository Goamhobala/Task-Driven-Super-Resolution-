import os
from pathlib import Path
from typing import Annotated
import typer
from processor.mask_generator import RoadMaskGenerator
from processor.classifier import TileClassifier
from viz import visualize_classification

app = typer.Typer(help="S2-ROSA Dataset Pipeline")

DATASET_HELP = "Base path to the S2-ROSA dataset directory"
DatasetDir = Annotated[Path, typer.Option(help=DATASET_HELP)]

@app.command()
def mask(
    dataset_dir: DatasetDir,
    sat_img: Annotated[str, typer.Option(help="Filename of satellite COG inside imagery/")],
    parquet: Annotated[Path, typer.Option(help="Absolute path to Overture parquet file")],
):
    """Generate the COG road mask and road vector parquet."""
    sat_path = os.path.join(dataset_dir, "imagery", sat_img)
    stem = os.path.splitext(sat_img)[0]
    # Name the outputs similarly to the satellite image
    mask_out_path = os.path.join(dataset_dir, "masks_raster", f"{stem}_mask.tif")
    graph_out_path = os.path.join(dataset_dir, "masks_graph", f"{stem}_roads.parquet")

    builder = RoadMaskGenerator(
        sat_cog_path=sat_path,
        vector_parquet_path=str(parquet),
        out_mask_path=mask_out_path,
        out_graph_path=graph_out_path,
    )
    builder.generate()


@app.command()
def classify(
    dataset_dir: DatasetDir,
    mask_img: Annotated[str, typer.Option(help="Filename of mask COG inside masks_raster/")],
):
    """Classify tiles and generate metadata.parquet."""
    mask_path = os.path.join(dataset_dir, "masks_raster", mask_img)
    metadata_path = os.path.join(dataset_dir, "metadata", "metadata.parquet")

    classifier = TileClassifier(
        mask_cog_path=mask_path,
        out_metadata_path=metadata_path,
    )
    classifier.classify()


@app.command()
def visualize(
    dataset_dir: DatasetDir,
    sat_img: Annotated[str, typer.Option(help="Filename of satellite COG inside imagery/")],
):
    """Visualize classifications over satellite imagery."""
    sat_path = os.path.join(dataset_dir, "imagery", sat_img)
    meta_dir = os.path.join(dataset_dir, "metadata")
    metadata_path = os.path.join(meta_dir, "metadata.parquet")
    plot_out_name = f"{os.path.splitext(sat_img)[0]}_mask.tif"
    plot_out_path = os.path.join(meta_dir, plot_out_name)

    visualize_classification(
        sat_cog_path=sat_path,
        out_plot_path=plot_out_path,
        metadata_path=metadata_path,
    )


if __name__ == "__main__":
    app()
