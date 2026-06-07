import argparse
import os
from processor.mask_generator import RoadMaskGenerator
from processor.classifier import TileClassifier
from viz import visualize_classification

def main():
    parser = argparse.ArgumentParser(description="S2-ROSA Dataset Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Base dataset path argument
    dataset_help = "Base path to the S2-ROSA dataset directory"

    # Generate Masks
    parser_mask = subparsers.add_parser("mask", help="Generate the COG road mask")
    parser_mask.add_argument("--dataset-dir", required=True, help=dataset_help)
    parser_mask.add_argument("--sat-img", required=True, help="Filename of satellite COG inside imagery/")
    parser_mask.add_argument("--parquet", required=True, help="Absolute path to Overture parquet file")

    # Classify Tiles
    parser_classify = subparsers.add_parser("classify", help="Classify tiles and generate metadata.parquet")
    parser_classify.add_argument("--dataset-dir", required=True, help=dataset_help)
    parser_classify.add_argument("--mask-img", required=True, help="Filename of mask COG inside masks_raster/")

    # Visualize
    parser_visualize = subparsers.add_parser("visualize", help="Visualize classifications over satellite imagery")
    parser_visualize.add_argument("--dataset-dir", required=True, help=dataset_help)
    parser_visualize.add_argument("--sat-img", required=True, help="Filename of satellite COG inside imagery/")

    args = parser.parse_args()

    # Define standardized paths based on the dataset directory structure
    img_dir = os.path.join(args.dataset_dir, "imagery")
    mask_dir = os.path.join(args.dataset_dir, "masks_raster")
    meta_dir = os.path.join(args.dataset_dir, "metadata")
    
    metadata_path = os.path.join(meta_dir, "metadata.parquet")

    if args.command == "mask":
        sat_path = os.path.join(img_dir, args.sat_img)
        # Name the output mask similarly to the satellite image
        mask_out_name = f"{os.path.splitext(args.sat_img)[0]}_mask.tif"
        mask_out_path = os.path.join(mask_dir, mask_out_name)

        builder = RoadMaskGenerator(
            sat_cog_path=sat_path,
            vector_parquet_path=args.parquet,
            out_mask_path=mask_out_path
        )
        builder.generate()

    elif args.command == "classify":
        mask_path = os.path.join(mask_dir, args.mask_img)
        
        classifier = TileClassifier(
            mask_cog_path=mask_path,
            out_metadata_path=metadata_path
        )
        classifier.classify()

    elif args.command == "visualize":
        sat_path = os.path.join(img_dir, args.sat_img)
        plot_out_path = os.path.join(meta_dir, "classification_on_image.png")
        
        visualize_classification(sat_cog_path=sat_path, out_plot_path=plot_out_path, metadata_path=metadata_path)

if __name__ == "__main__":
    main()