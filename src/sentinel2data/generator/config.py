from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = (".tif", ".tiff")
RASTER_EXTS = {".tif", ".tiff"}
WGS84 = "EPSG:4326"          # the catalogue / "common" CRS every tile is reported in
COMMON_CRS = WGS84           # backwards-friendly alias

# Scaffolded values
SCAFFOLD_DATES = ["2023-01-01"]   # TODO: real composite dates per tile
SCAFFOLD_BIOME = "Unknown"        # TODO: real biome/zone names per tile

# Road Buffer Values
# 5.0 = 1 pixel wide (5*2 = 10m)
ROAD_CLASS_BUFFER_M = {
    "motorway": 15.0,
    "trunk": 12.0,
    "primary": 10.0,
    "secondary": 8.0,
    "tertiary": 6.0,
    "unclassified": 5.0,
    "residential": 5.0,
    "living_street": 4.0,
    "service": 3.0,
    "track": 3.0,
    "pedestrian": 3.0,
    "cycleway": 2.0,
    "footway": 2.0,
    "path": 2.0,
    "steps": 2.0,
    "bridleway": 2.0,
}
ROAD_TIER_BUFFER_M = {"major": 12.0, "medium": 7.0}
DEFAULT_BUFFER_M = 5.0

# CDNGI (Road Vector) Constants
CDNGI_ROADS_LAYER = "TRAN_ROADS_EXP"
CDNGI_CLASS_MAP = {
    "National Freeway": "major",
    "On/OffRamp": "major",
    "National Road": "major",
    "Arterial Road": "major",
    "Main Road": "major",
    "Secondary Road": "medium",
    "Other Road": "medium", # need to double check
}

# Overture (Road Vector) Contants
OVERTURE_MAJOR_MEDIUM = ("motorway", "trunk", "primary", "secondary", "unclassified", "unknown")
OVERTURE_CLASS_MAP = {
    "motorway": "major",
    "trunk": "major",
    "primary": "major",
    "secondary": "medium",
    "tertiary": "medium", # need to double check
    "unclassified": "medium", # need to double check
    "unknown": "medium", # need to double check
}

# Cleaned parquet format from CDNGI
ROAD_VECTOR_COLUMNS = ("source", "source_class", "class", "province", "geometry")


# Classification labels - Urbanisation Clasisfication
CLASS_LABELS = ["Rural", "Peri-Urban", "Urban"]
EMPTY_LABEL = "Empty"

# Biome tagging 
BIOME_COL = "T_BIOME"
NULL_TOKENS = {"<null>", "null", "none", "nan", ""}
UNKNOWN_BIOME = "Unknown"

V2_METADATA_COLUMNS = [
    "image_id",
    "zone_name",
    "split_set",
    "image_path",
    "mask_path",          # raster mask
    "mask_graph_path",
    "tile_size",          # tile edge in px (512)
    "band_count",         # source bands + 3 appended enhanced-RGB bands
    "spatial_resolution",
    "road_pixels",
    "road_density",
    "urbanisation_classification",  # Jenks class from road_density, per split
    "biome",
    "satellite_image_dates",
    "crs",
    "geometry",
]

V2_SPLIT_CSV_COLUMNS = [
    "image_id",
    "zone_name",
    "split_set",
    "image_path",
    "mask_path",
    "mask_graph_path",
    "road_density",
]

# Metadata Column Schema V1
# Include internal patch tiling
V1_METADATA_COLUMNS = [
    # indexing
    "tile_id",
    "patch_row_id",
    "patch_col_id",
    "zone_name",
    # paths relative to the dataset dir
    "tile_path",
    "mask_raster_path",
    "mask_graph_path",
    # classification
    "spatial_resolution",
    "urbanisation_classification",
    "biome",
    "road_density",
    "split_set",
    # additional metadata
    "satellite_image_dates",
    "crs",
    "patch_bounding_geometry",
]
V1_SPLIT_CSV_COLUMNS = [
    "tile_id",
    "patch_row_id",
    "patch_col_id",
    "zone_name",
    "tile_path",
    "mask_raster_path",
    "mask_graph_path",
]

# Dataset Processing Configs
@dataclass(frozen=True)
class TileSpec:
    """Tile / patch geometry shared by tiling strategies."""

    tile_size: int = 512
    patch_size: int = 256

    # TODO: this check is likely not needed
    def __post_init__(self):
        if self.tile_size % self.patch_size != 0:
            raise ValueError(
                f"tile_size ({self.tile_size}) must be a whole multiple of "
                f"patch_size ({self.patch_size}) so tiles divide into clean patches."
            )


@dataclass(frozen=True)
class SplitFractions:
    """Holdout fractions + seed for any SplitStrategy."""

    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class RGBEnhanceConfig:
    """Params for the appended CLAHE + gamma enhanced-RGB bands (scikit-image).

    Pipeline per RGB channel: min-max normalise to [0,1] -> gamma -> CLAHE
    (``skimage.exposure.equalize_adapthist``). The 3 results are appended to the
    imagery as extra float32 bands.
    """

    gamma: float = 0.4
    clahe_kernel_size: tuple = (32, 32)
    clahe_clip_limit: float = 0.01   # tune within 0.01-0.03
    rgb_bands: tuple = (1, 2, 3)      # 1-based source bands for R, G, B (B4, B3, B2)


@dataclass(frozen=True)
class CatalogueSchema:
    """Describes a catalogue variant's row schema so the pipeline can assemble +
    write it without knowing which processor produced the rows.

    ``reindex_tile_id`` -> overwrite ``tile_id`` with a global 0..N-1 index after
    concatenation (cut-tiles), vs keep the processor's ids (patch-windows, where
    ``tile_id`` groups a whole zone for classification + splitting).
    """

    columns: list
    geometry_col: str
    split_csv_columns: list
    reindex_tile_id: bool = False


V2_SCHEMA = CatalogueSchema(
    columns=V2_METADATA_COLUMNS,
    geometry_col="geometry",
    split_csv_columns=V2_SPLIT_CSV_COLUMNS,
    reindex_tile_id=False,  # V2 split-first runner assigns image_id itself
)
V1_SCHEMA = CatalogueSchema(
    columns=V1_METADATA_COLUMNS,
    geometry_col="patch_bounding_geometry",
    split_csv_columns=V1_SPLIT_CSV_COLUMNS,
    reindex_tile_id=False,
)


@dataclass(frozen=True)
class DatasetPaths:
    """Resolves the on-disk layout of a dataset from its root directory.

    Holds every subpath either pipeline variant might write; each processor
    reads only the ones it needs (v2 -> images/masks, v1 -> masks_raster/graph).
    """

    root: Path

    def __post_init__(self):
        # Normalise to Path without breaking frozen-ness.
        object.__setattr__(self, "root", Path(self.root))

    # v2 (cut tiles)
    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    # v1 (patch windows, masks generated in place)
    @property
    def imagery_dir(self) -> Path:
        return self.root / "imagery"

    @property
    def masks_raster_dir(self) -> Path:
        return self.root / "masks_raster"

    @property
    def masks_graph_dir(self) -> Path:
        return self.root / "masks_graph"

    # shared
    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.parquet"


def split_layout(root, split):
    """Per-split V2 output dirs: ``<root>/<split>/{imagery,masks_raster,masks_graph}``."""
    base = Path(root) / split
    return {
        "imagery": base / "imagery",
        "masks_raster": base / "masks_raster",
        "masks_graph": base / "masks_graph",
    }
