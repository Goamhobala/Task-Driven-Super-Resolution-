"""
Purpose of this module is to extract the desired road centrelines from raw datasets.
Currently support CD:NGI and Overture datasets.
"""
from pathlib import Path
from abc import ABC, abstractmethod
import geopandas as gpd
import pandas as pd
from sentinel2data.generator.config import (
    CDNGI_ROAD_CLASSIFICATION,
    OVERTURE_ROAD_CLASSIFICATION,
    ROAD_VECTOR_COLUMNS,
    WGS84,
    flatten_classification
)

CDNGI_SCALE_MAP, CDNGI_BUFFER_MAP = flatten_classification(CDNGI_ROAD_CLASSIFICATION)
OVERTURE_SCALE_MAP, OVERTURE_BUFFER_MAP = flatten_classification(OVERTURE_ROAD_CLASSIFICATION)

# Overture's ``class`` column holds only real (non-link) class names -- links are
# flagged by ``subclass == 'link'`` and mapped to synthetic ``<class>_link`` keys
# afterwards. So the predicate pushdown filters on the non-link keys only.
OVERTURE_PUSHDOWN_CLASSES = [rc for rc, sc in OVERTURE_SCALE_MAP.items() if sc != "links"]


class RoadSource(ABC):
    """Interface for different road source implementations"""

    #: parquet ``data_source`` literal + GPKG layer name for this source
    data_source: str
    layer_name: str

    @abstractmethod
    def load(self) -> "gpd.GeoDataFrame | None":
        """Returns roads in WGS84"""
        pass


class CdngiSource(RoadSource):
    """Roads from CD:NGI Geopackages"""

    CDNGI_ROADS_LAYER = "TRAN_ROADS_EXP"
    data_source = "cdngi"
    layer_name = "CDNGI_roads"

    def __init__(self, path: str | Path):
        """
        Args:
            path (str | Path): Path to the root directory containing CD:NGI GeoPackages.
        """
        self.path = Path(path)

    def _gpkg_paths(self):
        """Scan for list of .gpkg files"""
        gpkg_files = sorted(self.path.rglob("*.gpkg"))
        print(f"Found {len(gpkg_files)} CD:NGI GeoPackage files in {self.path}")
        return gpkg_files

    def load(self):
        road_type_filter = list(CDNGI_SCALE_MAP)
        where = "FEAT_TYPE IN ({})".format(", ".join(f"'{type}'" for type in road_type_filter))

        provincial_gpd_roads = []
        for gpkg in self._gpkg_paths():
            province = gpkg.stem.split("_")[0]  # assume default naming convention: <province>_NGI_TOPODATA_<year>.gpkg
            print(f"Reading CDNGI {gpkg.name} (province {province})...")
            gdf = gpd.read_file(
                gpkg, layer=CdngiSource.CDNGI_ROADS_LAYER, columns=["FEAT_TYPE"], where=where
            )

            if gdf.empty:
                print(f"Warning: No roads found in {gpkg.name} (province {province}).")
                continue

            gdf = gdf.to_crs(WGS84)
            feat = gdf["FEAT_TYPE"]
            provincial_gpd_roads.append(
                gpd.GeoDataFrame(
                    {
                        "data_source": self.data_source,
                        "road_class": feat.to_numpy(),
                        "scale_class": feat.map(CDNGI_SCALE_MAP).to_numpy(),
                        "buffer": feat.map(CDNGI_BUFFER_MAP).to_numpy(),
                        "geometry": gdf.geometry.to_numpy(),
                    },
                    crs=WGS84,
                )
            )

        if not provincial_gpd_roads:
            return None
        combined = gpd.GeoDataFrame(
            pd.concat(provincial_gpd_roads, ignore_index=True), geometry="geometry", crs=WGS84
        )
        print(f"CDNGI: {len(combined)} road segments.")
        return combined


class OvertureSource(RoadSource):
    """Kept-scale roads from an Overture roads GeoParquet (predicate pushdown)."""

    data_source = "overture"
    layer_name = "OVERTURE_roads"

    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        print(f"Reading Overture {self.path.name} (predicate pushdown)...")
        filters = [
            ("subtype", "==", "road"),
            ("class", "in", OVERTURE_PUSHDOWN_CLASSES),
        ]
        gdf = gpd.read_parquet(
            self.path,
            columns=["subtype", "class", "subclass", "geometry"],
            filters=filters,
        )
        if gdf.empty:
            return None
        gdf = gdf.to_crs(WGS84)

        # Promote link ramps to their synthetic ``<class>_link`` key when the config
        # defines one; everything else keeps its raw Overture class name.
        base = gdf["class"]
        link_key = base + "_link"
        is_link = (gdf["subclass"] == "link") & link_key.isin(OVERTURE_SCALE_MAP)
        road_class = base.where(~is_link, link_key)

        out = gpd.GeoDataFrame(
            {
                "data_source": self.data_source,
                "road_class": road_class.to_numpy(),
                "scale_class": road_class.map(OVERTURE_SCALE_MAP).to_numpy(),
                "buffer": road_class.map(OVERTURE_BUFFER_MAP).to_numpy(),
                "geometry": gdf.geometry.to_numpy(),
            },
            crs=WGS84,
        )
        print(f"Overture: {len(out)} road segments ({int(is_link.sum())} links).")
        return out


class RoadVectorExtractor:
    """Extract kept-scale roads from one :class:`RoadSource`."""

    def __init__(self, out_path, source):
        if source is None:
            raise ValueError("RoadVectorExtractor needs exactly one RoadSource.")
        self.out_path = Path(out_path)
        self.source = source

    @classmethod
    def from_paths(
        cls,
        out_path,
        cdngi_path=None,
        overture_path=None,
    ):
        """Build from one of CDNGI or Overture."""
        if (cdngi_path is None) == (overture_path is None):
            raise ValueError("Provide exactly one of cdngi_path or overture_path.")
        if cdngi_path is not None:
            source = CdngiSource(cdngi_path)
        else:
            source = OvertureSource(overture_path)
        return cls(out_path, source)

    def build(self):
        """Load the source and write the normalized layer. Returns the output path."""
        roads = self.source.load()
        if roads is None or roads.empty:
            raise ValueError("No road features extracted from the source.")

        combined = roads[list(ROAD_VECTOR_COLUMNS)]
        by_scale = combined.groupby("scale_class").size().to_dict()
        print(f"Extracted {len(combined)} segments by scale: {by_scale}")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing roads to {self.out_path}...")
        if self.out_path.suffix.lower() == ".gpkg":
            combined.to_file(self.out_path, driver="GPKG", layer=self.source.layer_name)
        else:
            # Per-row covering bbox so downstream readers can spatially filter
            # (RasterMaskLabeler reads a per-COG bbox window).
            combined.to_parquet(self.out_path, write_covering_bbox=True)
        print("Road vector extraction complete.")
        return self.out_path
