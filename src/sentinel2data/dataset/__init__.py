"""Shared S2-ROSA-V2 data loading -- any consumer model imports from here.

Native-CRS, no-warp loaders + a model-agnostic sliding-window stitch:

    from sentinel2data.dataset import RoadDataModule, predict_zone, DEFAULT_BANDS

``import sentinel2data.dataset`` pulls torch + lightning; for band metadata only,
import :mod:`sentinel2data.dataset.bands` (numpy-free).
"""
from sentinel2data.dataset.bands import (
    DEFAULT_BANDS,
    ENHANCED_RGB,
    ENHANCED_RGB_BANDS,
    RAW_RGB,
    RAW_RGB_NIR,
    S2_BANDS,
    S2_V2_BANDS,
)
from sentinel2data.dataset.datasets import (
    RoadDataModule,
    RoadTileDataset,
    ZoneDataset,
)
from sentinel2data.dataset.reading import apply_norm, read_window, standardize
from sentinel2data.dataset.sliding import blend_weight, plan_windows, predict_zone

__all__ = [
    # bands
    "S2_BANDS",
    "S2_V2_BANDS",
    "ENHANCED_RGB_BANDS",
    "RAW_RGB",
    "RAW_RGB_NIR",
    "ENHANCED_RGB",
    "DEFAULT_BANDS",
    # reading
    "read_window",
    "standardize",
    "apply_norm",
    # datasets
    "RoadTileDataset",
    "ZoneDataset",
    "RoadDataModule",
    # sliding
    "plan_windows",
    "blend_weight",
    "predict_zone",
]
