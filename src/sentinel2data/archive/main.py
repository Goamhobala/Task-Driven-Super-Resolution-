from sentinel2data.preprocessor import convert_dataset_to_graphs

def main():
    # Update this to where your original 256x256 satellite images are stored
    IMG_DIR = "/Volumes/MacOSFiles/Sentinel2OriginalData/images_enhanced_png"
    MASK_DIR = "/Volumes/MacOSFiles/Sentinel2OriginalData/masks_png"

    # The new root directory for the 1024x1024 dataset
    OUTPUT_ROOT_DIR = "/Volumes/MacOSFiles/Sentinel2GraphData"

    convert_dataset_to_graphs(
        img_dir=IMG_DIR,
        mask_dir=MASK_DIR,
        output_dir=OUTPUT_ROOT_DIR,
        node_spacing=5,        # Baseline distance in 256x256 space (auto-scales to 20px)
        kernel_size=10,         # Morphological closing kernel (applied directly to upscaled image)
        scale_factor=4.0,      # Upscale factor (4x turns 256x256 into 1024x1024)
        min_spur_length=3      # Removes dead ends shorter than 3 pixels in 256x256 space (auto-scales to 12px)
    )

if "__main__" == __name__:
    main()