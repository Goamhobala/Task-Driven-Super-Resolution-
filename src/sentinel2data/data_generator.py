"""Generate Sentinel-2 chips that cover the polygons defined in a GeoJSON file.

Drives ``instageo.data.chip_creator`` as a subprocess, which is not importable
directly. The workflow is:

    1. Read polygons from the GeoJSON file.
    2. Grid-sample observation points inside each polygon at a spacing of
       ``CHIP_SIZE * S2_SPATIAL_RESOLUTION`` so the produced chips tile the
       polygon with minimal overlap.
    3. Write the points to a CSV with the ``date,x,y,label`` schema expected
       by chip_creator.
    4. Invoke ``python -m instageo.data.chip_creator`` with flags configured
       for a single-timestep, (near-)cloud-free Sentinel-2 pull anywhere in
       the last ~2 years.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry


# Sentinel-2 L2A spatial resolution in EPSG:4326 degrees per pixel (~10 m).
# Matches instageo.data.settings.S2APISettings.
S2_SPATIAL_RESOLUTION = 8.983152841195215e-05
CHIP_SIZE = 256

# 0 = strictly cloud-free. Bump to e.g. 1-5 if no scenes are found.
CLOUD_COVERAGE_PCT = 0

# Search window: centre date + tolerance in days on either side.
# Today is ~2026-04-16; this window covers 2024-09 through 2026-09, i.e. "last
# year and this year".
SEARCH_CENTRE_DATE = "2025-09-01"
TEMPORAL_TOLERANCE_DAYS = 365

REPO_ROOT = Path(__file__).resolve().parents[2]
GEOJSON_PATH = REPO_ROOT / "dataset" / "sentinel2" / "sa_map.geojson"
POINTS_CSV_PATH = REPO_ROOT / "dataset" / "sentinel2" / "sa_observations.csv"
OUTPUT_DIR = Path("/Volumes/MacOSFiles/Custom_S2_Dataset")

def sample_polygon_points(polygon: BaseGeometry, spacing: float):
    """Yield (lon, lat) grid points inside ``polygon`` at ``spacing`` degrees."""
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.arange(minx + spacing / 2, maxx, spacing)
    ys = np.arange(miny + spacing / 2, maxy, spacing)
    for y in ys:
        for x in xs:
            if polygon.contains(Point(x, y)):
                yield float(x), float(y)


def build_observations_csv(geojson_path: Path, csv_path: Path) -> int:
    """Write grid-sampled polygon points to ``csv_path``. Returns point count."""
    with open(geojson_path) as f:
        features = json.load(f)["features"]

    spacing = S2_SPATIAL_RESOLUTION * CHIP_SIZE
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(csv_path, "w") as f:
        f.write("date,x,y,label\n")
        for feat in features:
            geom = shape(feat["geometry"])
            for x, y in sample_polygon_points(geom, spacing):
                f.write(f"{SEARCH_CENTRE_DATE},{x:.8f},{y:.8f},1\n")
                count += 1
    return count


def run_chip_creator() -> None:
    """Invoke the chip_creator CLI as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "instageo.data.chip_creator",
        f"--dataframe_path={POINTS_CSV_PATH}",
        f"--output_directory={OUTPUT_DIR}",
        "--data_source=S2",
        "--data_format=csv",
        "--processing_method=cog",
        f"--chip_size={CHIP_SIZE}",
        f"--cloud_coverage={CLOUD_COVERAGE_PCT}",
        f"--temporal_tolerance={TEMPORAL_TOLERANCE_DAYS}",
        "--num_steps=1",
        # Defaults to True; we want a single image on the observation date.
        "--nois_time_series_task",
        # Defaults to True; we pass explicit dates so leave them unshifted.
        "--noshift_to_month_start",
        # Keep every MGRS tile, even sparsely-sampled polygons.
        "--min_count=1",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    n_points = build_observations_csv(GEOJSON_PATH, POINTS_CSV_PATH)
    print(f"Wrote {n_points} observations to {POINTS_CSV_PATH}")
    if n_points == 0:
        raise SystemExit("No sample points generated; aborting.")
    # run_chip_creator()


if __name__ == "__main__":
    main()
