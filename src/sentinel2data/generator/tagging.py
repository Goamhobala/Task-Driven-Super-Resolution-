"""Catalogue taggers: each adds one derived column to the metadata GeoDataFrame.

A :class:`Tagger` is a swappable strategy run by the pipeline after the
catalogue is assembled (``gdf[tagger.column] = tagger.tag(gdf)``):
  * :class:`BiomeTagger`           -- NVM2024 biome via point-in-polygon join.
  * :class:`UrbanisationClassifier`-- Jenks urbanisation class from road density.
"""
from pathlib import Path
from abc import ABC, abstractmethod
import geopandas as gpd
import jenkspy
import numpy as np
import pandas as pd
from sentinel2data.generator.config import (
    BIOME_COL,
    CLASS_LABELS,
    EMPTY_LABEL,
    NULL_TOKENS,
    UNKNOWN_BIOME,
    WGS84,
)

class Tagger(ABC):
    """Compute one catalogue column aligned to the input rows."""
    column: str
    
    @abstractmethod
    def tag(self, gdf: gpd.GeoDataFrame) -> pd.Series:
        pass


# --------------------------------------------------------------------------- #
# Biome
# --------------------------------------------------------------------------- #

class BiomeTagger(Tagger):
    """Look up the NVM2024 biome for each tile centroid from a GeoParquet.

    Tiles outside the map extent (or in a null biome) become ``"Unknown"``.
    """

    column = "biome"

    def __init__(self, biome_parquet, biome_col=BIOME_COL):
        self.biome_parquet = Path(biome_parquet)
        self.biome_col = biome_col
        self._biomes = None  # lazy

    def _clean_biome(self, s: pd.Series) -> pd.Series:
        """Normalise the ``<Null>`` text + empty values to real NA."""
        s = s.astype("string").str.strip()
        return s.mask(s.str.lower().isin(NULL_TOKENS), pd.NA)
        
    def _load(self):
        if self._biomes is not None:
            return self._biomes
        if not self.biome_parquet.exists():
            raise FileNotFoundError(
                f"Biome GeoParquet not found: {self.biome_parquet}\n"
                "Run:  python scripts/biome.py convert"
            )
        gdf = gpd.read_parquet(self.biome_parquet)
        if self.biome_col not in gdf.columns:
            raise KeyError(
                f"{self.biome_col!r} not in biome parquet columns: {list(gdf.columns)}"
            )
        gdf = gdf.to_crs(WGS84)
        gdf[self.biome_col] = self._clean_biome(gdf[self.biome_col])
        self._biomes = gdf[[self.biome_col, "geometry"]]
        return self._biomes

    def tag(self, gdf, geometry_col="geometry") -> pd.Series:
        biomes = self._load()

        # Centroid in a metric CRS (geographic centroids are distorted/warn),
        # then back to WGS84 for the point-in-polygon join.
        geom = gdf.set_geometry(geometry_col).to_crs(WGS84).geometry
        centroids4326 = geom.to_crs("EPSG:3857").centroid.to_crs(WGS84)
        centroids = gpd.GeoDataFrame(geometry=centroids4326, crs=WGS84)
        centroids["_row"] = range(len(centroids))

        joined = gpd.sjoin(centroids, biomes, how="left", predicate="within")
        # A centroid on a shared polygon edge can match >1 biome; keep the first.
        joined = joined.drop_duplicates(subset="_row", keep="first").sort_values("_row")

        out = joined[self.biome_col].fillna(UNKNOWN_BIOME)
        out.index = gdf.index
        result = out.rename(self.column)
        print(result.value_counts(dropna=False).to_string())
        return result


# --------------------------------------------------------------------------- #
# Urbanisation classification (Jenks natural breaks, computed per tile)
# --------------------------------------------------------------------------- #
class UrbanisationClassifier(Tagger):
    """Classify each tile Rural/Peri-Urban/Urban by road density.

    Breaks are computed *within each split* (``split_set`` group), so the class
    ratio is reported per train/val/test set; tiles with zero road density stay
    ``"Empty"``. Falls back to a single whole-catalogue group when the grouping
    column is absent.
    """

    column = "urbanisation_classification"

    def __init__(self, density_col="road_density", group_col="split_set"):
        self.density_col = density_col
        self.group_col = group_col

    def tag(self, gdf) -> pd.Series:
        out = pd.Series(EMPTY_LABEL, index=gdf.index, dtype=object)
        if self.group_col in gdf.columns:
            groups = (g for _, g in gdf.groupby(self.group_col))
        else:
            groups = [gdf]
        for group in groups:
            labels = self._classify(group[self.density_col].to_numpy())
            out.loc[group.index] = labels
        return out.rename(self.column)

    @staticmethod
    def _classify(densities: np.ndarray) -> np.ndarray:
        labels = np.full(densities.shape, EMPTY_LABEL, dtype=object)
        values = densities[densities > 0]
        if values.size == 0:
            return labels

        # Jenks needs >= 3 distinct values for 3 classes; otherwise span the
        # range directly. Dedup break edges: degenerate inputs (few distinct
        # densities) make Jenks repeat edges, which pd.cut rejects, so the
        # number of usable bins -- and labels -- shrinks accordingly.
        if np.unique(values).size >= 3:
            breaks = sorted(set(jenkspy.jenks_breaks(values, n_classes=3)))
        else:
            breaks = sorted({float(values.min()), float(values.max())})

        n_bins = len(breaks) - 1
        if n_bins < 1:
            per_value = np.full(values.size, CLASS_LABELS[0])
        else:
            bin_labels = CLASS_LABELS[:n_bins]
            per_value = pd.cut(
                values, bins=breaks, labels=bin_labels, include_lowest=True
            ).astype(str)
            print("Density breaks:", [round(b, 5) for b in breaks], "->", bin_labels)

        labels[densities > 0] = per_value
        return labels
