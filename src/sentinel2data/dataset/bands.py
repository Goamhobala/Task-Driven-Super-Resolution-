"""Canonical S2-ROSA band layout -- shared metadata for any consumer model.

1-based on disk. ``S2_BANDS`` is the 20-band source; S2-ROSA-V2 appends 3
CLAHE+gamma enhanced-RGB bands (21-23). ``*_RGB*`` are the common index groups a
model passes to the loaders / ``rasterio.read``.
"""

S2_BANDS = (
    "B4", "B3", "B2", "B8", "B5", "B6", "B7", "B8A", "B11", "B12",
    "VV_ascending", "VH_ascending", "VV_descending", "VH_descending",
    "elevation", "slope", "aspect",
    "esa_urban_10m", "gisa_urban_10m", "wsf_urban_10m",
)
ENHANCED_RGB_BANDS = ("B4_clahe", "B3_clahe", "B2_clahe")
S2_V2_BANDS = S2_BANDS + ENHANCED_RGB_BANDS

# Common 1-based index groups into the 23-band V2 imagery.
RAW_RGB = (1, 2, 3)          # B4, B3, B2
RAW_RGB_NIR = (1, 2, 3, 4)   # + B8 (NIR)
ENHANCED_RGB = (21, 22, 23)  # appended CLAHE+gamma B4, B3, B2
DEFAULT_BANDS = ENHANCED_RGB
