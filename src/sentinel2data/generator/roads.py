"""
Purpose of this module is to extract the desired road centrelines from raw datasets.
Currently support CD:NGI and Overture datasets.
"""
from pathlib import Path
from abc import ABC, abstractmethod
import geopandas as gpd
import pandas as pd
from sentinel2data.generator.config import (
    CDNGI_CLASS_MAP,
    OVERTURE_CLASS_MAP,
    OVERTURE_MAJOR_MEDIUM,
    ROAD_VECTOR_COLUMNS,
    WGS84,
)


class RoadSource(ABC):
    """Interface for different road source implementations"""

    @abstractmethod
    def load(self) -> "gpd.GeoDataFrame | None":
        """Returns roads in WGS84"""
        pass


class CdngiSource(RoadSource):
    """Roads from CD:NGI Geopackages"""

    CDNGI_ROADS_LAYER = "TRAN_ROADS_EXP"

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
        road_type_filter = list(CDNGI_CLASS_MAP)
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
            provincial_gpd_roads.append(
                gpd.GeoDataFrame(
                    {
                        "source": "cdngi",
                        "source_class": gdf["FEAT_TYPE"].to_numpy(),
                        "class": gdf["FEAT_TYPE"].map(CDNGI_CLASS_MAP).to_numpy(),
                        "province": province,
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
        print(f"CDNGI: {len(combined)} major+medium road segments.")
        return combined


class OvertureSource(RoadSource):
    """Major + medium roads from an Overture roads GeoParquet (predicate pushdown)."""

    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        print(f"Reading Overture {self.path.name} (predicate pushdown)...")
        filters = [
            ("subtype", "==", "road"),
            ("class", "in", list(OVERTURE_MAJOR_MEDIUM)),
        ]
        gdf = gpd.read_parquet(
            self.path,
            columns=["subtype", "class", "subclass", "geometry"],
            filters=filters,
        )
        if gdf.empty:
            return None
        gdf = gdf.to_crs(WGS84)
        out = gpd.GeoDataFrame(
            {
                "source": "overture",
                "source_class": gdf["class"].to_numpy(),
                "class": gdf["class"].map(OVERTURE_CLASS_MAP).to_numpy(),
                "province": None,
                "geometry": gdf.geometry.to_numpy(),
            },
            crs=WGS84,
        )
        print(f"Overture: {len(out)} major+medium road segments.")
        return out


class RoadVectorExtractor:
    """Extract major+medium roads from one :class:`RoadSource`.
    """

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
        cdngi_layer=CDNGI_ROADS_LAYER,
    ):
        """Build from one of CDNGI or Overture."""
        if (cdngi_path is None) == (overture_path is None):
            raise ValueError("Provide exactly one of cdngi_path or overture_path.")
        if cdngi_path is not None:
            source = CdngiSource(cdngi_path, layer=cdngi_layer)
        else:
            source = OvertureSource(overture_path)
        return cls(out_path, source)

    def build(self):
        """Load the source and write the normalized layer. Returns the output path."""
        roads = self.source.load()
        if roads is None or roads.empty:
            raise ValueError("No road features extracted from the source.")

        combined = roads[list(ROAD_VECTOR_COLUMNS)]
        by_class = combined.groupby("class").size().to_dict()
        print(f"Extracted {len(combined)} segments by class: {by_class}")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing roads to {self.out_path}...")
        if self.out_path.suffix.lower() == ".gpkg":
            combined.to_file(self.out_path, driver="GPKG", layer="roads_major_medium")
        else:
            # Per-row covering bbox so downstream readers can spatially filter
            # (RasterMaskLabeler reads a per-COG bbox window).
            combined.to_parquet(self.out_path, write_covering_bbox=True)
        print("Road vector extraction complete.")
        return self.out_path
