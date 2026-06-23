"""Biome tagging for tile metadata.

Vectorised port of ``scripts/biome.py``'s point lookup: given the NVM2024
``T_BIOME`` GeoParquet (produced by ``python scripts/biome.py convert``) and a
set of tile-centroid points in EPSG:4326, attach the biome each tile falls in
via a single spatial join instead of one ``contains`` test per point.

Both flavours of "null" that exist in ``T_BIOME`` (the literal text ``<Null>``
and genuine empty values) are normalised to a real NA, matching the script.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

WGS84 = "EPSG:4326"
BIOME_COL = "T_BIOME"
# Mirrors scripts/biome.py: the two null flavours both collapse to NA.
NULL_TOKENS = {"<null>", "null", "none", "nan", ""}
UNKNOWN_BIOME = "Unknown"


def _clean_biome(s: pd.Series) -> pd.Series:
    """Normalise the ``<Null>`` text + empty values to real NA (see biome.py)."""
    s = s.astype("string").str.strip()
    return s.mask(s.str.lower().isin(NULL_TOKENS), pd.NA)


class BiomeTagger:
    """Looks up the NVM2024 biome for tile centroids from a GeoParquet."""

    def __init__(self, biome_parquet, biome_col=BIOME_COL):
        self.biome_parquet = Path(biome_parquet)
        self.biome_col = biome_col
        self._biomes = None  # lazy

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
        gdf[self.biome_col] = _clean_biome(gdf[self.biome_col])
        self._biomes = gdf[[self.biome_col, "geometry"]]
        return self._biomes

    def tag(self, tiles_gdf, geometry_col="geometry"):
        """Return a biome ``pd.Series`` aligned to ``tiles_gdf`` rows.

        ``tiles_gdf`` must be a GeoDataFrame in EPSG:4326 (its active geometry is
        used). Tile centroids are joined ``within`` the biome polygons; tiles
        outside the map extent (or in a null biome) become ``"Unknown"``.
        """
        biomes = self._load()

        # Centroid in a metric CRS (geographic centroids are distorted/warn),
        # then back to WGS84 for the point-in-polygon join.
        geom = tiles_gdf.set_geometry(geometry_col).to_crs(WGS84).geometry
        centroids4326 = geom.to_crs("EPSG:3857").centroid.to_crs(WGS84)
        centroids = gpd.GeoDataFrame(geometry=centroids4326, crs=WGS84)
        centroids["_row"] = range(len(centroids))

        joined = gpd.sjoin(centroids, biomes, how="left", predicate="within")
        # A centroid on a shared polygon edge can match >1 biome; keep the first.
        joined = joined.drop_duplicates(subset="_row", keep="first").sort_values("_row")

        out = joined[self.biome_col].fillna(UNKNOWN_BIOME)
        out.index = tiles_gdf.index
        return out.rename("biome")
