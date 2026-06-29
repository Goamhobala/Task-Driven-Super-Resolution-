"""Shared S2-ROSA-V2 data loading -- import directly from the submodules:

    from sentinel2data.dataset.datasets import RoadDataModule              # torch + lightning
    from sentinel2data.dataset.sliding import predict_zone                 # torch
    from sentinel2data.dataset.reading import apply_norm, standardize      # numpy only
    from sentinel2data.dataset.bands import DEFAULT_BANDS, written_band_names  # pure-python
"""
