"""UNet++ ResNet baseline for Sentinel-2 road segmentation.

End-to-end: `train.py` fits the model on the combined multi-band COGs and
`benchmark.py` scores the best checkpoint on the held-out test sites, writing the
per-chip parquet the `benchmarking` package consumes.
"""
