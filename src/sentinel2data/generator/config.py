from dataclasses import dataclass
from pathlib import Path

## GENERAL CONSTANTS ##

IMAGE_EXTS = (".tif", ".tiff")
RASTER_EXTS = {".tif", ".tiff"}
WGS84 = "EPSG:4326"          # common lat/lon CRS
COMMON_CRS = WGS84           # google maps view of a map
# Scaffolded values
SCAFFOLD_DATES = ["2023-01-01"]   # composite dates per tile


## ROAD CLASSIFICATION ##

# classification -> road classification -> buffer (half-meter)
# buffer 5.0m = 1 pixel wide in 10m (5*2 = 10m)
OVERTURE_ROAD_CLASSIFICATION = {
    "large": {
        "motorway": 15.0, 
        "trunk": 12.0, 
        "primary": 10.0, 
        "secondary": 8.0
    },
    "medium": {
        "tertiary": 6.0, 
        "unclassified": 5.0, 
        "residential": 5.0
    },
    "links": { # links (not certain about the buffers)
        "motorway_link": 12.0, 
        "trunk_link": 10.0, 
        "primary_link": 8.0, 
        "secondary_link": 6.0,
        "tertiary_link": 5.0,   
    },
    "small": {
        "living_street": 4.0, 
        "track": 3.0
    },
    "ignored": {
        "pedestrian": -1, 
        "path": -1, 
        "footway": -1, 
        "service": -1, 
        "unknown": -1, 
        "steps": -1, 
        "bridleway": -1, 
        "cycleway": -1
    }
}

CDNGI_ROAD_CLASSIFICATION = {
    "large": {
        "National Freeway": 15.0, 
        "National Road": 12.0, 
        "Arterial Road": 10.0, 
        "Main Road": 8.0
    },
    "medium": {
        "On/OffRamp": 6.0, 
        "Secondary Road": 5.0, 
        "Other Road": 5.0
    },
    "small": {
        "Street": 4.0, 
        "Track": 3.0, 
        "Slipway": 3.0,
    },
    "ignored": {
        "Footpath": -1
    }
}

def flatten_classification(road_classification_config: dict, exclude_class: list = ["ignored"]) -> tuple[dict, dict, list]:
    """Flatten config into mappings of road class to scale and buffer. 
    The ignored category is dropped.

    Args:
        road_classification_config (dict): The road classification configuration.
        exclude_class (list): List of scale classes to exclude.
    """
    scale_map, buffer_map, road_classifications = {}, {}, []
    for scale_class, classes in road_classification_config.items():
        if scale_class in exclude_class:
            continue
        for road_class, buffer in classes.items():
            scale_map[road_class] = scale_class
            buffer_map[road_class] = buffer
            road_classifications.append(road_class)
    return scale_map, buffer_map, road_classifications


## METADATA CONSTANTS ##

# Classification labels - Urbanisation Clasisfication
CLASS_LABELS = ["Rural", "Peri-Urban", "Urban"]
EMPTY_LABEL = "Empty"

# Biome tagging 
BIOME_COL = "T_BIOME"
NULL_TOKENS = {"<null>", "null", "none", "nan", ""}
UNKNOWN_BIOME = "Unknown"

V2_METADATA_COLUMNS = [
    # indexing
    "image_id",
    "zone_name",
    "image_path",
    "mask_path",          # raster mask
    "mask_graph_path",

    # tile metadata
    "road_pixels",
    "road_density",
    "urbanisation_classification",  # Jenks class from road_density, independent per split
    "biome",
    "satellite_image_dates",
    "split_set",

    # image metadata
    "crs",
    "tile_bounding_geometry",   # easy visualisation of dataset tiles
    "tile_size",                # tile edge in px. Constant (512px x 512px)
    "band_count",               # source bands + 3 appended enhanced-RGB bands. Constant (23)
    "spatial_resolution",       # spatial resolution. Constant (10)
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

## Dataset Processing Configs ##
@dataclass(frozen=True)
class TileSpec:
    tile_size: int = 512        # tile size from large zone imagery
    patch_size: int = 256       # internal COG patch size

@dataclass(frozen=True)
class SplitFractions:
    """Dataset split proportions and seed"""
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42

@dataclass(frozen=True)
class RGBEnhanceConfig:
    """CLAHE + gamma enhanced-RGB bands (scikit-image) configs."""
    gamma: float = 0.4
    clahe_kernel_size: tuple = (32, 32)
    clahe_clip_limit: float = 0.01   # tune within 0.01-0.03
    rgb_bands: tuple = (1, 2, 3)      # 1-based source bands for R, G, B (B4, B3, B2)

@dataclass(frozen=True)
class CatalogueSchema:
    """Describes a catalogue variant's row schema"""
    columns: list
    geometry_col: str
    split_csv_columns: list
    reindex_tile_id: bool = False # TODO: likely not needed. Overwrite tile_id with global 0..N-1 index after concatenation (cut-tiles) vs keep processor's ids (patch-windows, where tile_id groups a whole zone for classification + splitting)


V2_SCHEMA = CatalogueSchema(
    columns=V2_METADATA_COLUMNS,
    geometry_col="tile_bounding_geometry",
    split_csv_columns=V2_SPLIT_CSV_COLUMNS,
    reindex_tile_id=False,  # V2 split-first runner assigns image_id itself
)

# TODO fix this
@dataclass(frozen=True)
class DatasetPaths:
    """Directory layout of the dataset from its root directory"""
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
