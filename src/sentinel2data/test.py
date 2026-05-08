import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Using your specific directory structure
image_path = "road_mask_local.tif"

with rasterio.open(image_path) as src:
    # 1. Read the pixel data into a NumPy array
    # Sentinel-2 often has many bands. .read() extracts them all.
    # To read specific bands (e.g., Band 1, 2, and 3 for RGB):
    image_array = src.read([1])

    # 2. Extract the Spatial Metadata
    crs = src.crs                 # Coordinate Reference System (e.g., UTM zone)
    transform = src.transform     # The mathematical mapping of pixels to physical meters
    bounds = src.bounds           # The bounding box (left, bottom, right, top)
    profile = src.profile         # A dictionary containing all file metadata

print(f"Image Array Shape: {image_array.shape}")

print(f"Coordinate Reference System: {crs}")

plt.figure(figsize=(8, 8))
plt.imshow(image_array[0], cmap='gray')  # Display the first band (road mask)
plt.title("Road Mask from Sentinel-2")
plt.axis('off')
plt.show()