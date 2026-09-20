"""Export ROSA_New tiles into the two artifacts the demo serves (plan §5 Phase 2).

Two consumers, two artifacts, and they must not be confused:

  tiles/<id>.npy   bands 1-4 = [B4,B3,B2,B8] at 0-1 reflectance, fp16, (4,512,512).
                   The Space reads these; the browser never does. One file per
                   tile ON PURPOSE -- that is what lets the Space lazily fetch a
                   single 2.1 MB tile instead of snapshotting a 2.6 GB repo on
                   every cold boot.
  chips/<id>.jpg   display-only RGB for the map. JPEG, not PNG: these are the
                   bandwidth line item (Render includes 100 GB/mo), and lossy is
                   fine for something no model ever reads. ~85 KB vs ~340 KB.

THE STRETCH IS PER SITE, NOT PER TILE. `viz_tile` fixes its 2-98 percentile
stretch across models within one figure, for the same reason a figure must not
invite the reader to compare histograms. Here the analogous artefact is the
5x5 sheet: stretching each tile independently makes adjacent cells of ONE site
jump in brightness, and the seams read as data. So the percentiles are pooled
over every tile in a site, recorded in stretch.json, and applied uniformly.

Band 1 is B4 (red), so RGB is just the first three bands in order -- do not
"fix" this into 3,2,1. Measured fp16 cost: max 4.8e-4 absolute, 0.03% of mask
pixels flipped at theta* (demo/space/parity.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path("/Volumes/MAC_KIOXIA/Data/ROSA_New/ROSADataset")
DEFAULT_OUT = Path("/Volumes/KIOXIA/instaroad_demo_export")
BANDS = (1, 2, 3, 4)          # rasterio is 1-indexed: B4, B3, B2, B8
PCT = (2.0, 98.0)
STRIDE = 2                    # percentile subsample; 4.9M samples is ample
JPEG_QUALITY = 85


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).parent / "cells.json")
    args = ap.parse_args()

    cells = [c for c in json.loads(args.manifest.read_text())
             if c["split"] in args.splits]
    by_site: dict[str, list[dict]] = {}
    for c in cells:
        by_site.setdefault(c["site"], []).append(c)

    tiles_dir, chips_dir = args.out / "tiles", args.out / "chips"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    chips_dir.mkdir(parents=True, exist_ok=True)
    stretch_path = args.out / "stretch.json"
    stretch = json.loads(stretch_path.read_text()) if stretch_path.exists() else {}

    n_tiles = n_bytes = chip_bytes = 0
    worst_err = 0.0
    for si, (site, group) in enumerate(sorted(by_site.items()), 1):
        group.sort(key=lambda c: (c["row"], c["col"]))
        arrays = {}
        for c in group:
            with rasterio.open(ROOT / c["split"] / "imagery" / f"{c['id']}.tif") as ds:
                arrays[c["id"]] = ds.read(BANDS).astype(np.float32)

        pool = np.concatenate(
            [a[:3, ::STRIDE, ::STRIDE].reshape(3, -1) for a in arrays.values()], axis=1)
        lo = np.percentile(pool, PCT[0], axis=1).astype(float)
        hi = np.percentile(pool, PCT[1], axis=1).astype(float)
        stretch[site] = {"lo": lo.tolist(), "hi": hi.tolist(),
                         "pct": list(PCT), "n_tiles": len(group)}
        del pool

        for cid, a in arrays.items():
            half = a.astype(np.float16)
            worst_err = max(worst_err, float(np.abs(half.astype(np.float32) - a).max()))
            npy = tiles_dir / f"{cid}.npy"
            np.save(npy, half)
            n_bytes += npy.stat().st_size
            n_tiles += 1

            rgb = (a[:3] - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None]
            img = Image.fromarray(
                (np.clip(rgb, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))
            jpg = chips_dir / f"{cid}.jpg"
            img.save(jpg, quality=JPEG_QUALITY, optimize=True)
            chip_bytes += jpg.stat().st_size
        del arrays
        print(f"[{si:2d}/{len(by_site)}] {site:48s} {len(group):3d} tiles")

    stretch_path.write_text(json.dumps(stretch, indent=1))
    n = max(n_tiles, 1)
    print(f"\n{n_tiles} tiles")
    print(f"  npy  {n_bytes/1e6:8.1f} MB   ({n_bytes/n/1e6:.2f} MB/tile)")
    print(f"  jpg  {chip_bytes/1e6:8.1f} MB   ({chip_bytes/n/1e3:.0f} KB/tile)")
    print(f"max |fp16 - fp32| = {worst_err:.2e}   (tolerance 1e-3)")
    assert worst_err < 1e-3, "fp16 cast exceeded tolerance"


if __name__ == "__main__":
    main()
