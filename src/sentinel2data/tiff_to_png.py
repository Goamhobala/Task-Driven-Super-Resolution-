import numpy as np
import rasterio


def cumulative_stretch(input_tif, output_png):
    with rasterio.open(input_tif) as src:
        # Read the RGB bands (1, 2, 3)
        img = src.read([1, 2, 3]).astype(np.float32)

        # Create an empty array for the result
        stretched_img = np.zeros_like(img)

        for i in range(3):
            band = img[i]
            # Calculate the 2nd and 98th percentiles
            p2, p98 = np.percentile(band[band > 0], [2, 98])

            # Clip and Stretch to 0-1
            stretched = np.clip(band, p2, p98)
            stretched = (stretched - p2) / (p98 - p2)
            stretched_img[i] = stretched * 255

        # Convert to 8-bit
        img_8bit = stretched_img.astype(np.uint8)

        # Save using PIL or write back with Rasterio
        from PIL import Image

        final_img = np.transpose(img_8bit, (1, 2, 0))
        Image.fromarray(final_img).save(output_png)


cumulative_stretch("JohannesburgOutsideSouthWest.tif", "JohannesburgOutsideSouthWest.png")
