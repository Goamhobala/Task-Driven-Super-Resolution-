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
    WGS84,
    flatten_classification
)

class RoadSource(ABC):
    """Interface for different road source implementations"""

    data_source: str
    gpkg_layer_name: str

    @abstractmethod
    def load(self) -> "gpd.GeoDataFrame | None":
        """Returns roads in WGS84"""
        pass


class CdngiSource(RoadSource):
    """Roads from CD:NGI Geopackages"""

    CDNGI_ROADS_LAYER = "TRAN_ROADS_EXP"
    data_source = "cdngi"
    gpkg_layer_name = "CDNGI_roads"

    def __init__(self, path: str | Path):
        """
        Args:
            path (str | Path): Path to the root directory containing CD:NGI GeoPackages.
        """
        self.path = Path(path)
        self.scale_map, self.buffer_map, self.road_class = flatten_classification(CDNGI_ROAD_CLASSIFICATION)


    def _gpkg_paths(self):
        """Scan for list of .gpkg files"""
        gpkg_files = sorted(self.path.rglob("*.gpkg"))
        print(f"Found {len(gpkg_files)} CD:NGI GeoPackage files in {self.path}")
        return gpkg_files

    def load(self):
        road_type_filter = self.road_class
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
                        "scale_class": feat.map(self.scale_map.get).to_numpy(),
                        "buffer": feat.map(self.buffer_map.get).to_numpy(),
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

        if combined.isna().any().any():
            na_cols = combined.columns[combined.isna().any()].tolist()
            raise ValueError(f"NA values found in Overture output columns: {na_cols} for file {self.path}.")
        
        return combined


class OvertureSource(RoadSource):
    """Road Centrelines from Overture GeoParquet file from their CLI"""

    data_source = "overture"
    gpkg_layer_name = "OVERTURE_roads"

    def __init__(self, path):
        self.path = Path(path)

        self.road_class_filter = flatten_classification(OVERTURE_ROAD_CLASSIFICATION, exclude_class=["ignored", "links"])[2]
        self.scale_map, self.buffer_map, self.road_class = flatten_classification(OVERTURE_ROAD_CLASSIFICATION)

    def load(self):
        print(f"Reading Overture {self.path.name}")
        
        filters = [
            ("subtype", "==", "road"),
            ("class", "in", self.road_class_filter),
        ]
        gdf = gpd.read_parquet(
            self.path,
            columns=["subtype", "class", "subclass", "geometry"],
            filters=filters,
        )
        
        if gdf.empty:
            raise ValueError(f"No roads found in Overture file {self.path}.")
        
        gdf = gdf.to_crs(WGS84)

        base = gdf["class"]
        
        link_key = base + "_link"
        is_link = (
            (gdf["subclass"].fillna("") == "link")
        ).to_numpy()
        unknown_links = link_key[is_link & ~link_key.isin(self.road_class)]
        if not unknown_links.empty:
            raise ValueError(
                f"Unknown link roads found in Overture file {self.path}: {unknown_links.unique()}." + 
                "\n Maybe check links in Overture Config?"
            )
        
        road_class = base.where(~is_link, link_key)

        out = gpd.GeoDataFrame(
            {
                "data_source": self.data_source,
                "road_class": road_class.to_numpy(),
                "scale_class": road_class.map(self.scale_map.get).to_numpy(),
                "buffer": road_class.map(self.buffer_map.get).to_numpy(),
                "geometry": gdf.geometry.to_numpy(),
            },
            crs=WGS84,
        )
        print(f"Overture: {len(out)} road segments ({int(is_link.sum())} links).")

        if out.isna().any().any():
            na_cols = out.columns[out.isna().any()].tolist()
            raise ValueError(f"NA values found in Overture output columns: {na_cols} for file {self.path}.")
        
        return out



class RoadVectorExtractor:
    """Manager for reading raw data and writing clean road centrelines."""

    # Cleaned output format
    ROAD_VECTOR_COLUMNS = ("data_source", "road_class", "scale_class", "buffer", "geometry")

    def __init__(self, out_path, source: RoadSource):
        self.out_path = Path(out_path)
        self.source = source

    @classmethod
    def from_paths(
        cls,
        out_path,
        cdngi_path=None,
        overture_path=None,
    ):
        """Build from CDNGI or Overture file path."""
        if (cdngi_path is None) == (overture_path is None):
            raise ValueError("Provide exactly one of cdngi_path or overture_path.")
        if cdngi_path is not None:
            source = CdngiSource(cdngi_path)
        else:
            source = OvertureSource(overture_path)
        return cls(out_path, source)

    def build(self):
        """Load the source and write road centerlines. Returns the output path."""
        roads = self.source.load()
        if roads is None or roads.empty:
            raise ValueError("No road features extracted from the source.")

        if roads.columns.difference(self.ROAD_VECTOR_COLUMNS).any():
            raise ValueError(
                f"Unexpected columns in extracted roads: {roads.columns.difference(self.ROAD_VECTOR_COLUMNS).tolist()}"
            )

        by_scale = roads.groupby("scale_class").size().to_dict()
        print(f"Extracted {len(roads)} segments by scale: {by_scale}")

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing roads to {self.out_path}...")
        
        if self.out_path.suffix.lower() == ".gpkg":
            roads.to_file(self.out_path, driver="GPKG", layer=self.source.gpkg_layer_name)
        else:
            roads.to_parquet(self.out_path, write_covering_bbox=True)
        print("Road vector extraction complete.")
        return self.out_path
