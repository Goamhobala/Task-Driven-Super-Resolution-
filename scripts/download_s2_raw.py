"""
Download raw Sentinel-2 10m band GeoTIFFs for a GeoJSON region.

Outputs a single 4-band stacked TIF per scene (B02, B03, B04, B08).
No reprojection — output TIFs are in each tile's native UTM CRS.

Usage:
    uv run scripts/download_s2_raw.py \
        --geojson dataset/sentinel2/sa_map.geojson \
        --output_dir /mnt/hhd/home/Projects/S2Raw \
        --start_date 2024-01-01 \
        --end_date 2025-01-01 \
        --cloud_coverage 10
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import planetary_computer
import rasterio
import rasterio.mask
from pyproj import Transformer
from pystac_client import Client
from shapely.geometry import mapping, shape
from shapely.ops import transform

BANDS_10M = ["B02", "B03", "B04", "B08"]
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


def load_aoi(geojson_path: str):
    with open(geojson_path) as f:
        fc = json.load(f)
    return shape(fc["features"][0]["geometry"])


def reproject_geometry(geom, src_epsg: int, dst_epsg: int):
    transformer = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return transform(transformer.transform, geom)


def search_scenes(aoi, start_date: str, end_date: str, cloud_coverage: int) -> list[Any]:
    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        intersects=mapping(aoi),
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lte": cloud_coverage}},
    )
    items = list(search.item_collection())
    print(f"Found {len(items)} scenes.")
    return items


def download_scene_stacked(item: Any, aoi, output_dir: Path) -> None:
    item = planetary_computer.sign(item)
    scene_id = item.id
    out_path = output_dir / f"{scene_id}.tif"

    if out_path.exists():
        print(f"  [{scene_id}] already exists, skipping.")
        return

    band_arrays = []
    out_meta = None
    aoi_native = None

    for band in BANDS_10M:
        if band not in item.assets:
            print(f"  [{scene_id}] {band} not available, skipping scene.")
            return

        href = item.assets[band].href
        try:
            with rasterio.open(href) as src:
                if aoi_native is None:
                    native_epsg = src.crs.to_epsg()
                    aoi_native = (
                        reproject_geometry(aoi, 4326, native_epsg)
                        if native_epsg and native_epsg != 4326
                        else aoi
                    )

                out_image, out_transform = rasterio.mask.mask(
                    src,
                    [mapping(aoi_native)],
                    crop=True,
                    nodata=src.nodata or 0,
                )
                if out_meta is None:
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver":    "GTiff",
                        "count":     len(BANDS_10M),
                        "height":    out_image.shape[1],
                        "width":     out_image.shape[2],
                        "transform": out_transform,
                        "compress":  "deflate",
                    })
                band_arrays.append(out_image[0])

        except Exception as e:
            print(f"  [{scene_id}] {band} FAILED: {e}")
            return

    stacked = np.stack(band_arrays, axis=0)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(stacked)
        for i, name in enumerate(BANDS_10M, start=1):
            dst.update_tags(i, name=name)

    print(f"  [{scene_id}] saved ({len(BANDS_10M)} bands) -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson",        required=True,            help="Path to AOI GeoJSON")
    parser.add_argument("--output_dir",     required=True,            help="Output directory")
    parser.add_argument("--start_date",     required=True,            help="Start date YYYY-MM-DD")
    parser.add_argument("--end_date",       required=True,            help="End date YYYY-MM-DD")
    parser.add_argument("--cloud_coverage", type=int, default=10,     help="Max cloud cover %%")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aoi = load_aoi(args.geojson)
    items = search_scenes(aoi, args.start_date, args.end_date, args.cloud_coverage)

    for i, item in enumerate(items):
        print(f"\nScene {i + 1}/{len(items)}: {item.id}")
        download_scene_stacked(item, aoi, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
