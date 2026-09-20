"""Parity gate for the serving path (plan §5 Phase 3).

Two separate questions, deliberately not conflated:

  A. Does demo/space/infer.py replay the checkpoint the SAME way the
     benchmarked path does? Feed BOTH the identical array and demand exact
     equality. Any difference is my windowing, not the data.

  B. What does fp16 tile STORAGE cost the prediction? Feed the fp32 GeoTIFF to
     the reference and the shipped fp16 .npy to the serving path. This is the
     honest number for "is the demo showing the same mask as the thesis".

Both matter because the known failure modes here (reflectance-scale coupling,
eval-collapsed SEN2SR convs) produce a plausible-looking WRONG mask, never an
exception.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import rasterio                                            # noqa: E402
from infer import Arm, load_arm, pick_device               # noqa: E402
from sr.viz_tile import run_tile                           # noqa: E402

TILE = "CapeTown_Fynbos_-33p96_18p61_Urban_r2_c2"
SPLIT = "test"
ROSA = Path("/Volumes/MAC_KIOXIA/Data/ROSA_New/ROSADataset")
NPY = Path("/Volumes/KIOXIA/instaroad_demo_export/tiles")
CKPT = REPO / "demo" / "weights" / "r2a" / "model.ckpt"
SEN2SR = REPO / "models" / "SEN2SRLite_RGBN"


def main() -> int:
    dev = pick_device()
    print(f"device: {dev}\ntile:   {TILE}\n")

    with rasterio.open(ROSA / SPLIT / "imagery" / f"{TILE}.tif") as ds:
        fp32 = ds.read((1, 2, 3, 4)).astype(np.float32)
    fp16 = np.load(NPY / f"{TILE}.npy").astype(np.float32)
    print(f"input   max|fp16 - fp32| = {np.abs(fp16 - fp32).max():.3e}\n")

    arm = load_arm("r2a")
    print(f"arm     window={arm.step}px  up={arm.up}  pad={arm.pad}  "
          f"rs={arm.rs}  theta*={arm.theta}\n")

    # ---- A. windowing parity: identical input to both paths -------------
    sr_ref, p_ref = run_tile(str(CKPT), str(SEN2SR), fp32, dev)
    sr_mine, p_mine = arm.predict(fp32)
    d_sr = float(np.abs(sr_ref - sr_mine).max())
    d_p = float(np.abs(p_ref - p_mine).max())
    print("A. serving vs benchmarked windowing, SAME array")
    print(f"     max|d SR|   = {d_sr:.3e}")
    print(f"     max|d prob| = {d_p:.3e}")
    ok_a = d_sr == 0.0 and d_p == 0.0
    print(f"     {'PASS - bit identical' if ok_a else 'FAIL - windowing differs'}\n")

    # ---- B. what fp16 storage costs -------------------------------------
    _, p_fp16 = arm.predict(fp16)
    d_prob = float(np.abs(p_ref - p_fp16).max())
    m_ref = p_ref > arm.theta
    m_16 = p_fp16 > arm.theta
    flipped = int((m_ref ^ m_16).sum())
    tot = m_ref.size
    print("B. shipped fp16 tile vs source fp32 GeoTIFF")
    print(f"     max|d prob|      = {d_prob:.3e}")
    print(f"     pixels flipped   = {flipped:,} / {tot:,} ({100*flipped/tot:.4f}%)")
    print(f"     road px (fp32)   = {int(m_ref.sum()):,}")
    print(f"     road px (fp16)   = {int(m_16.sum()):,}")
    print(f"     IoU(masks)       = "
          f"{(m_ref & m_16).sum() / max((m_ref | m_16).sum(), 1):.6f}")
    return 0 if ok_a else 1


if __name__ == "__main__":
    raise SystemExit(main())
