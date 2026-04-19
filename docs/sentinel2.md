# Sentinel 2

## Specifing Areas to obtain satellite data
Use https://geojson.io.


## Generation of Satellite Imagery
Have data_generator.py to generate the observations.csv for chip_generator script. Then we run:

```bash
uv run instageo/data/chip_creator.py --dataframe_path=/mnt/hhd/home/Projects/InstaRoadPrototype/dataset/sentinel2/sa_observations.csv --output_directory=/mnt/hhd/home/Projects/S2ImageData --data_source=S2 --data_format=csv --processing_method=cog --chip_size=256 --cloud_coverage=0 --temporal_tolerance=365 --num_steps=1 --nois_time_series_task --noshift_to_month_start --min_count=1
```

## Visualising S2 Images
Build Virtual Raster:

1. Go to the top menu: Raster > Miscellaneous > Build Virtual Raster...
2. Input Layers: Click the ... button. Instead of selecting files, click Add Directory and point it to your folder containing all the South African TIFFs.
3. Resolution: Set this to Average or Highest (Highest is usually safer for research).
4. Place each input file in a separate band: Uncheck this (you want them to stitch together, not stack on top of each other).
5. Run: Save the output as SouthAfrica_Mosaic.vrt.
6. Once it finishes, drag that one .vrt file into QGIS. It will look like one massive image.

## Only 10m bands
The native 10m Sentinel-2 bands are only 4:

B02 → "blue"
B03 → "green"
B04 → "red"
B08 → "nir broad"
To pull just those, change ASSET in settings.py:177:
ASSET: List[str] = ["blue", "green", "red", "nir broad"]
Then set spatial_resolution to 10m when running chip_creator. The default is ~30m in degrees; 10m ≈ 0.00008983 degrees:

```bash
uv run instageo/data/chip_creator.py --dataframe_path=/mnt/hhd/home/Projects/InstaRoadPrototype/dataset/sentinel2/sa_observations.csv --output_directory=/mnt/hhd/home/Projects/S2Image10m --data_source=S2 --data_format=csv --processing_method=cog --chip_size=256 --cloud_coverage=0 --temporal_tolerance=365 --num_steps=1 --nois_time_series_task --noshift_to_month_start --min_count=1 --spatial_resolution=0.00008983
```

Second script
```
uv run download_s2.py \
  --geojson /mnt/hhd/home/Projects/InstaRoadPrototype/dataset/sentinel2/sa_map.geojson \
  --output_dir /mnt/hhd/home/Projects/S2ImageOriginal \
  --start_date 2025-01-01 \
  --end_date 2026-04-01 \
  --cloud_coverage 0
```