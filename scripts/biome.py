#!/usr/bin/env python3
"""NVM2024 biome map tool: shapefile -> GeoParquet, plus point / bbox lookup.

The South African National Vegetation Map 2024 (Beta) ships as an Albers Equal
Area shapefile. This script converts the `T_BIOME` attribute (+ a few useful
companions) to GeoParquet in WGS84 (EPSG:4326) so it is easy to visualise
(QGIS, kepler.gl, lonboard, leafmap, DuckDB) and query by lon/lat.

Two flavours of "null" exist in T_BIOME and both are normalised to a real null:
  - `(null)`  : genuine empty DBF value  (1168 features)
  - `<Null>`  : the literal text "<Null>" (103 features)

Examples
--------
    # one-off convert (shp -> geoparquet, reprojected to 4326)
    python scripts/biome.py convert

    # which biome is at a point?  (lon lat, i.e. x y)
    python scripts/biome.py point 18.42 -33.92        # Cape Town

    # also accepts lat,lon order if you prefer
    python scripts/biome.py point --latlon -33.92 18.42

    # which biomes fall inside a bounding box? (minlon minlat maxlon maxlat)
    python scripts/biome.py bbox 18.3 -34.0 18.7 -33.7

    # quick choropleth PNG for a sanity-check visual
    python scripts/biome.py plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# --- defaults -------------------------------------------------------------
SHP = Path(
    "/Volumes/MacOSFiles/NVM2024Beta_Shapefile/Shapefile/"
    "NVM2024Beta_IEM5_11_01072024.shp"
)
OUT = Path(__file__).resolve().parent.parent / "dataset" / "nvm2024_t_biome.parquet"
# columns worth keeping alongside T_BIOME
KEEP = ["T_BIOME", "T_BIOMEID", "T_Name", "T_MAPCODE", "T_BIOREGIO"]
NULL_TOKENS = {"<null>", "null", "none", "nan", ""}
WGS84 = 4326


# --- helpers --------------------------------------------------------------
def _clean_biome(s: pd.Series) -> pd.Series:
    """Normalise the two null flavours (`<Null>` text + empty) to real NA."""
    s = s.astype("string").str.strip()
    return s.mask(s.str.lower().isin(NULL_TOKENS), pd.NA)


def load(parquet: Path) -> gpd.GeoDataFrame:
    if not parquet.exists():
        sys.exit(f"GeoParquet not found: {parquet}\nRun:  python {sys.argv[0]} convert")
    return gpd.read_parquet(parquet)


# --- commands -------------------------------------------------------------
def cmd_convert(args: argparse.Namespace) -> None:
    shp, out = Path(args.shp), Path(args.out)
    print(f"reading {shp} ...", flush=True)
    gdf = gpd.read_file(shp, columns=KEEP)  # pyogrio: attrs only, geom implicit
    print(f"  {len(gdf):,} features, crs={gdf.crs.to_string() if gdf.crs else '?'}")

    gdf["T_BIOME"] = _clean_biome(gdf["T_BIOME"])
    n_null = gdf["T_BIOME"].isna().sum()
    print(f"  T_BIOME nulls normalised: {n_null:,}")

    print(f"reprojecting -> EPSG:{WGS84} ...", flush=True)
    gdf = gdf.to_crs(WGS84)

    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out, compression="zstd", index=False)
    mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({mb:.1f} MB)")
    print("\nbiome counts:")
    print(gdf["T_BIOME"].value_counts(dropna=False).to_string())


def cmd_point(args: argparse.Namespace) -> None:
    if args.latlon:
        lat, lon = args.coords
    else:
        lon, lat = args.coords
    gdf = load(Path(args.parquet))
    pt = gpd.GeoSeries.from_xy([lon], [lat], crs=WGS84).iloc[0]

    # spatial index -> candidate polygons -> exact contains test
    cand = gdf.iloc[list(gdf.sindex.query(pt, predicate="intersects"))]
    hit = cand[cand.contains(pt)]
    if hit.empty:
        print(f"({lon}, {lat}) -> no polygon (outside map extent)")
        return
    for _, r in hit.iterrows():
        biome = r["T_BIOME"] if pd.notna(r["T_BIOME"]) else "<null>"
        print(f"({lon}, {lat}) -> biome={biome!s} | {r.get('T_Name', '')}")


def cmd_bbox(args: argparse.Namespace) -> None:
    minx, miny, maxx, maxy = args.coords
    gdf = load(Path(args.parquet))
    idx = list(gdf.sindex.query(
        gpd.GeoSeries.from_wkt(
            [f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},"
             f"{minx} {maxy},{minx} {miny}))"], crs=WGS84
        ).iloc[0],
        predicate="intersects",
    ))
    hit = gdf.iloc[idx]
    print(f"bbox {minx,miny,maxx,maxy} -> {len(hit):,} polygons")
    vc = hit["T_BIOME"].value_counts(dropna=False)
    print(vc.to_string())


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gdf = load(Path(args.parquet))
    gdf = gdf.assign(_b=gdf["T_BIOME"].fillna("Unknown (null)"))
    ax = gdf.plot(column="_b", legend=True, figsize=(11, 11), linewidth=0,
                  legend_kwds={"loc": "lower left", "fontsize": 7})
    ax.set_axis_off()
    ax.set_title("NVM2024 T_BIOME")
    out = Path(args.out)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


# --- cli ------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="shapefile -> GeoParquet (WGS84)")
    c.add_argument("--shp", default=str(SHP))
    c.add_argument("--out", default=str(OUT))
    c.set_defaults(func=cmd_convert)

    pt = sub.add_parser("point", help="biome at a point")
    pt.add_argument("coords", nargs=2, type=float, metavar=("LON", "LAT"),
                    help="lon lat (x y); use --latlon to flip")
    pt.add_argument("--latlon", action="store_true", help="interpret as LAT LON")
    pt.add_argument("--parquet", default=str(OUT))
    pt.set_defaults(func=cmd_point)

    bb = sub.add_parser("bbox", help="biomes inside a bounding box")
    bb.add_argument("coords", nargs=4, type=float,
                    metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    bb.add_argument("--parquet", default=str(OUT))
    bb.set_defaults(func=cmd_bbox)

    pl = sub.add_parser("plot", help="quick choropleth PNG")
    pl.add_argument("--parquet", default=str(OUT))
    pl.add_argument("--out", default=str(OUT.with_name("nvm2024_t_biome.png")))
    pl.add_argument("--dpi", type=int, default=150)
    pl.set_defaults(func=cmd_plot)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
