"""Shared config loader for the sentinel2data package.

Reads ``config.yaml`` from the same directory as this file.
Copy ``config.yaml.example`` to ``config.yaml`` and fill in your paths.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_DEFAULTS: dict = {
    "paths": {
        "geojson": "dataset/sentinel2/sa_map.geojson",
        "points_csv": "dataset/sentinel2/sa_observations.csv",
        "output_dir": "output/sentinel2_chips",
        "img_dir": "FILL_IN",
        "mask_dir": "FILL_IN",
        "graph_output_dir": "output/sentinel2_graphs",
    },
    "chip": {
        "size": 256,
        "cloud_coverage_pct": 0,
        "search_centre_date": "2025-09-01",
        "temporal_tolerance_days": 365,
    },
    "preprocessor": {
        "node_spacing": 5,
        "kernel_size": 10,
        "scale_factor": 4.0,
        "min_spur_length": 3,
    },
    "gee": {
        "project_id": "FILL_IN",
        "drive_folder": "SA_S1_S2_Road_Tiles_V2",
        "start_date": "2025-01-01",
        "end_date": "2026-01-01",
        "grid_step": 2,
        "max_cloud_cover": 1,
        "min_lon": 18,
        "max_lon": 20,
        "min_lat": -35,
        "max_lat": -33,
    },
}


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        print(f"[warning] No config.yaml found at {config_path}; using built-in defaults.")
        print(f"[hint]    Copy config.yaml.example to config.yaml and fill in your paths.")
        return _DEFAULTS
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return {
        section: {**_DEFAULTS.get(section, {}), **cfg.get(section, {})}
        for section in set(_DEFAULTS) | set(cfg)
    }


def resolve(val: str) -> Path:
    """Resolve a config path value: absolute paths pass through, relative ones anchor to repo root."""
    p = Path(val)
    return p if p.is_absolute() else REPO_ROOT / p
