"""Road vector sourcing: extract one :class:`RoadSource` (CDNGI *or* Overture)
into a normalized GeoParquet (or GeoPackage).

The source filters to its large/medium classes, reprojects to EPSG:4326 and
remaps its native class onto the simplified major/medium tier carried in the
``class`` column (buffer widths per tier live in
:data:`sentinel2data.generator.config.ROAD_TIER_BUFFER_M`). Exactly one source
is used per output layer, so the result never stacks the duplicate centrelines
two datasets would produce for the same road.
"""
from pathlib import Path
from typing import Protocol, runtime_checkable
import geopandas as gpd
import pandas as pd
from sentinel2data.generator.config import (
    CDNGI_CLASS_MAP,
    CDNGI_ROADS_LAYER,
    OVERTURE_CLASS_MAP,
    OVERTURE_MAJOR_MEDIUM,
    ROAD_VECTOR_COLUMNS,
    WGS84,
)


@runtime_checkable
class RoadSource(Protocol):
    """A provider of normalized major/medium road centrelines (EPSG:4326)."""

    def load(self) -> "gpd.GeoDataFrame | None":
        """Return normalized roads, or ``None`` if the source yields nothing."""
        ...


class CdngiSource:
    """Major + medium roads from one or many CDNGI GeoPackages."""

    def __init__(self, path, layer=CDNGI_ROADS_LAYER):
        self.path = Path(path)
        self.layer = layer

    def _gpkg_paths(self):
        """Resolve ``path`` to a sorted list of .gpkg files (file or directory)."""
        if self.path.is_dir():
            return sorted(self.path.rglob("*.gpkg"))
        return [self.path]

    def load(self):
        keys = list(CDNGI_CLASS_MAP)
        where = "FEAT_TYPE IN ({})".format(", ".join(f"'{k}'" for k in keys))

        parts = []
        for gpkg in self._gpkg_paths():
            province = gpkg.stem.split("_")[0]
            print(f"Reading CDNGI {gpkg.name} (province {province})...")
            gdf = gpd.read_file(
                gpkg, layer=self.layer, columns=["FEAT_TYPE"], where=where
            )
            if gdf.empty:
                continue
            gdf = gdf.to_crs(WGS84)
            parts.append(
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

        if not parts:
            return None
        combined = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), geometry="geometry", crs=WGS84
        )
        print(f"CDNGI: {len(combined)} major+medium road segments.")
        return combined


class OvertureSource:
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
