"""Build sites.json / cells.json for the public demo (plan §5 Phase 0).

Site names do NOT parse reliably into biomes: three random-sampling zones carry
no biome at all, two spell Indian Ocean Coastal Belt as `IndianCoastal`, and
Knysna says `Forest` where every other forest zone says `Forests`. The GEE
export script (`ROSASampling60zones2020_images`) is the only record of the true
grouping, so its comment structure is transcribed into OVERRIDES below.
Everything else parses from the filename.

Footprints come from the UNION OF EACH SITE'S GEOTIFF BOUNDS, reprojected to
WGS84 -- not the filename coordinates, which are rounded to ~500 m. Filenames
are `..._{lat}_{lon}_{urbanisation}`; the GEE script's argument order is the
reverse, `createRectangularBBox(lon, lat)`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

ROOT = Path("/Volumes/MAC_KIOXIA/Data/ROSA_New/ROSADataset")
OUT = Path(__file__).parent
SPLITS = ("train", "val", "test")

BIOMES = ["IndianOceanCoastalBelt", "AlbanyThicket", "AzonalVegetation",
          "SucculentKaroo", "NamaKaroo", "Savanna", "Grassland", "Forests",
          "Fynbos", "Desert"]

OVERRIDES = {
    "Durban_IndianCoastal_-29p851_30p94_Urban":      "IndianOceanCoastalBelt",
    "RichardsBay_IndianCoastal_-28p69_32p03_Urban":  "IndianOceanCoastalBelt",
    "Knysna_Forest_-33p98_23p07_Urban":              "Forests",
    "Soebatsfontein_-30p12_17p59_PeriUrban":         "SucculentKaroo",
    "Kenhart_-29p35_21p15_PeriUrban":                "NamaKaroo",
    "Carnarvon_-30p97_22p13_PeriUrban":              "NamaKaroo",
}
GT_DIRNAME = "mask_new_2pt5"


def parse_site(site: str) -> dict:
    parts = site.split("_")
    urbanisation, lon_s, lat_s = parts[-1], parts[-2], parts[-3]
    head = "_".join(parts[:-3])
    if site in OVERRIDES:
        biome, city = OVERRIDES[site], head
        for variant in (biome, "Forest", "IndianCoastal"):
            if city.endswith("_" + variant):
                city = city[: -(len(variant) + 1)]; break
            if city == variant:
                city = ""; break
    else:
        hit = [b for b in BIOMES if head == b or head.endswith("_" + b)]
        if not hit:
            raise ValueError(f"unresolvable site name: {site!r} "
                             "-- add it to OVERRIDES from the GEE script")
        biome = max(hit, key=len)
        city = head[: -len(biome)].rstrip("_")
    return {"biome": biome, "urbanisation": urbanisation, "city": city or None,
            "lat_name": float(lat_s.replace("p", ".")),
            "lon_name": float(lon_s.replace("p", "."))}


def main() -> None:
    cells: list[dict] = []
    per_site_bounds: dict[str, list] = defaultdict(list)
    per_site_split: dict[str, str] = {}
    per_site_crs: dict[str, set] = defaultdict(set)

    for split in SPLITS:
        imagery, gt_dir = ROOT / split / "imagery", ROOT / split / GT_DIRNAME
        for tif in sorted(imagery.glob("*.tif")):
            stem = tif.stem
            site, rc = stem.rsplit("_r", 1)
            r, c = rc.split("_c")
            with rasterio.open(tif) as ds:                 # header only
                b, crs, h, w, n = ds.bounds, ds.crs, ds.height, ds.width, ds.count
            wgs = transform_bounds(crs, "EPSG:4326", *b, densify_pts=21)
            per_site_bounds[site].append(wgs)
            per_site_split[site] = split
            per_site_crs[site].add(str(crs))
            cells.append({"id": stem, "site": site, "row": int(r), "col": int(c),
                          "split": split, "px": [w, h], "bands": n, "crs": str(crs),
                          "bounds_utm": list(b), "bounds_wgs84": list(wgs),
                          "has_gt": (gt_dir / tif.name).exists()})

    sites = []
    for site, boxes in per_site_bounds.items():
        meta = parse_site(site)
        xs0 = min(b[0] for b in boxes); ys0 = min(b[1] for b in boxes)
        xs1 = max(b[2] for b in boxes); ys1 = max(b[3] for b in boxes)
        tiles = [c for c in cells if c["site"] == site]
        sites.append({"id": site, **meta, "split": per_site_split[site],
                      "crs": sorted(per_site_crs[site]), "n_tiles": len(tiles),
                      "rows": sorted({c["row"] for c in tiles}),
                      "cols": sorted({c["col"] for c in tiles}),
                      "footprint_wgs84": [xs0, ys0, xs1, ys1],
                      "centroid_wgs84": [(xs0 + xs1) / 2, (ys0 + ys1) / 2]})
    sites.sort(key=lambda s: (s["split"], s["biome"], s["id"]))
    cells.sort(key=lambda c: (c["site"], c["row"], c["col"]))

    assert all(s["biome"] in BIOMES for s in sites), "unknown biome escaped"
    known = {s["id"] for s in sites}
    assert all(c["site"] in known for c in cells), "orphan cell"
    for s in sites:
        assert len(s["crs"]) == 1, f"{s['id']}: mixed CRS {s['crs']}"
    odd = [c for c in cells if c["px"] != [512, 512] or c["bands"] != 23]
    assert not odd, f"unexpected tile geometry: {odd[:3]}"

    (OUT / "sites.json").write_text(json.dumps(sites, indent=1))
    (OUT / "cells.json").write_text(json.dumps(cells, indent=1))
    print(f"sites.json  {len(sites)} sites")
    print(f"cells.json  {len(cells)} cells   "
          f"({sum(c['has_gt'] for c in cells)} with {GT_DIRNAME} GT)")
    for sp in SPLITS:
        ss = [s for s in sites if s["split"] == sp]
        print(f"  {sp:6} {len(ss):3d} sites  {sum(s['n_tiles'] for s in ss):5d} tiles")


if __name__ == "__main__":
    main()
