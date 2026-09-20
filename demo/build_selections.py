"""Precompute every legal grid selection per site (plan §5 Phase 0/4).

WHY THIS IS COMPUTED ONCE, SERVER-SIDE, AND NOT IN THE UI
---------------------------------------------------------
The browser and the Space must agree EXACTLY on which cells a selection covers
and which pixel window each one maps to. Two implementations of that arithmetic
-- one in JS for the map, one in Python for inference -- is two chances to
drift, and a drift here is silent: the mask simply lands on the wrong ground.
So the legality rules live here, the result ships in the manifest, and the UI
only ever reads a list.

THE LATTICE
-----------
  atom  = 128 px = r2a's forced SEN2SR window (model._required_lr)
  tile  = 512 px = 4x4 atoms          <- the stored unit
  site  = 5x5 tiles = 20x20 atoms = 25.6 km

Selections are aligned to their own size, so 128/256/512 never cross a tile
boundary (a size-aligned block inside a 4x4 lattice nests cleanly). Only 1024
does, and it crosses only WHOLE-tile boundaries: it is exactly 2x2 tiles.
That is why 1024 is the single size needing a real validity check -- sites hold
12-25 of their 25 tiles, so most 2x2 blocks contain a hole.

CONTIGUITY IS ASSERTED, NOT ASSUMED
-----------------------------------
Stitching 2x2 tiles is only sound if their UTM bounds abut exactly. Sites are
single-CRS (checked in build_manifest), but abutting is a separate claim, so
every 1024 candidate is verified against the real bounds from cells.json.
Measured 2026-09-20: 672 quads, max gap 0.0000 m.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent
ATOM_PX, TILE_PX = 128, 512
SIZES = (128, 256, 512, 1024)
TOL_M = 0.01                              # bounds are exact multiples of 10 m


def main() -> None:
    cells = json.loads((OUT / "cells.json").read_text())
    sites = json.loads((OUT / "sites.json").read_text())
    by_site: dict[str, dict[tuple[int, int], dict]] = defaultdict(dict)
    for c in cells:
        by_site[c["site"]][(c["row"], c["col"])] = c

    worst_gap, n_checked, out = 0.0, 0, {}
    for site in sites:
        grid = by_site[site["id"]]
        present = sorted(grid)
        counts = {str(s): len(present) * (TILE_PX // s) ** 2 for s in (128, 256, 512)}
        anchors = []
        for (r, c) in present:
            quad = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            if not all(q in grid for q in quad):
                continue
            b = {q: grid[q]["bounds_utm"] for q in quad}       # [l, b, r, t]
            gaps = [abs(b[(r, c)][2] - b[(r, c + 1)][0]),
                    abs(b[(r + 1, c)][2] - b[(r + 1, c + 1)][0]),
                    abs(b[(r, c)][1] - b[(r + 1, c)][3]),
                    abs(b[(r, c + 1)][1] - b[(r + 1, c + 1)][3])]
            n_checked += 1
            worst_gap = max(worst_gap, max(gaps))
            if max(gaps) > TOL_M:
                raise AssertionError(
                    f"{site['id']} anchor r{r}_c{c}: tiles do not abut, "
                    f"max gap {max(gaps):.3f} m -- 1024 stitching is unsound here")
            anchors.append([r, c])
        counts["1024"] = len(anchors)
        out[site["id"]] = {"tiles_present": [[r, c] for r, c in present],
                           "n_tiles": len(present), "counts": counts,
                           "anchors_1024": anchors}

    (OUT / "selections.json").write_text(json.dumps(
        {"atom_px": ATOM_PX, "tile_px": TILE_PX, "sizes": list(SIZES),
         "sites": out}, indent=1))

    tot = {str(s): sum(v["counts"][str(s)] for v in out.values()) for s in SIZES}
    print(f"selections.json  {len(out)} sites")
    print(f"1024 contiguity: {n_checked} quads checked, max UTM gap {worst_gap:.4f} m")
    for s in SIZES:
        print(f"  {s:>4}  {tot[str(s)]:>7,d} selections   {s*10/1000:>5.2f} km")
    print(f"sites with a complete 5x5 sheet: "
          f"{sum(1 for v in out.values() if v['n_tiles'] == 25)}/{len(out)}")


if __name__ == "__main__":
    main()
