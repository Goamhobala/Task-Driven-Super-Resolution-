"""Canonical S2-ROSA band layout

`S2_BANDS` is the 20-band source
`S2-ROSA-V2` appends 3 CLAHE+gamma enhanced-RGB bands (21-23).
`*_RGB*` are the common index groups a model passes to the loaders / `rasterio.read`.
"""

S2_BANDS = (
    "B4_R", "B3_G", "B2_B", "B8_NIR", "B5", "B6", "B7", "B8A", "B11", "B12",
    "VV_ascending", "VH_ascending", "VV_descending", "VH_descending",
    "elevation", "slope", "aspect",
    "esa_urban_10m", "gisa_urban_10m", "wsf_urban_10m",
)
ENHANCED_RGB_BANDS = ("B4_R_enhanced", "B3_G_enhanced", "B2_B_enhanced")
S2_V2_BANDS = S2_BANDS + ENHANCED_RGB_BANDS

# Common index groups maping into the 23-band V2 imagery.
RGB = (1, 2, 3)                 # B4_R, B3_G, B2_B
S2_10M = (1, 2, 3, 4)           # RBG + B8_NIR
S2_20M = (5, 6, 7, 8, 9, 10)    # B5, B6, B7, B8A, B11, B12
S2_SAR = (11, 12, 13, 14)       # VV/VH ascending/descending
ENHANCED_RGB = (21, 22, 23)     # appended CLAHE+gamma enhanced RGB


DEFAULT_BANDS = RGB


def written_band_names(*, version=2):
    if version == 2:
        return list(S2_V2_BANDS)
    elif version == 1:
        return list(S2_BANDS)
    raise ValueError(f"Unknown version {version}, expected 1 or 2")
