#!/usr/bin/env python
"""Re-bench every swept arm LOCALLY, adding AP and the buffered-rho sweep.

The existing stores predate `--ap-bins` and `--buffer-px`, and those columns
cannot be back-filled: buffered scores need the per-chip prediction MASK and AP
needs the probability MAP, neither of which the store keeps (it keeps
tp/fp/fn/tn). So the only way to get them is to score the checkpoints again.

Measured at ~10.4 min/arm on an M-series Mac over MPS — about 2.5x faster than
the same arm on a Modal L4 container, because ~86% of a bench is serial CPU
(APLS/clDice have no thread pool) and one fast core beats two slow ones. 18
arms is therefore ~3 h and costs nothing.

Each arm is benched at ITS OWN theta*, read from its `sweep.json`, so the rows
match the operating point the existing tables already report. One arm failing
does not stop the rest; the summary lists what fell over.

    python scripts/local/rebench_all.py --runs-dir <runslightning> --store-dir <new store>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FINAL_CKPT = Path("checkpoints") / "unet_s2rosa_jointsr_final.ckpt"

# model_name -> run dir. Explicit rather than globbed: `sr_r0_new_tl_ce_holdout`
# prefix-matches three platform variants, and picking silently would bench a
# different arm from the one the existing tables describe.
ARMS = [
    ("sr_r0_new_dice_holdout",              "sr_r0_new_dice_holdout_seed0_L4_lightning"),
    ("sr_r0_new_sdice_holdout",             "sr_r0_new_sdice_holdout_seed0_L4_lightning"),
    ("sr_r0_new_lcdice_holdout",            "sr_r0_new_lcdice_holdout_seed0_L4_lightning"),
    ("sr_r0_new_tl_ce_holdout",             "sr_r0_new_tl_ce_holdout_seed0_L4_lightning"),
    ("sr_r0_new_t2_ce_holdout",             "sr_r0_new_t2_ce_holdout_seed0_L4_modal"),
    ("sr_r0_new_t4_ce_holdout",             "sr_r0_new_t4_ce_holdout_seed0_L4_modal"),
    ("sr_r0_new_gap_t2_ce_holdout",         "sr_r0_new_gap_t2_ce_holdout_seed0_L4_modal"),
    ("sr_r0_new_gap_t4_ce_holdout",         "sr_r0_new_gap_t4_ce_holdout_seed0_L4_modal"),
    ("sr_r0_new_gap_t2t4_ce_holdout",       "sr_r0_new_gap_t2t4_ce_holdout_seed0_L4_modal"),
    ("sr_r0_new_gap_tl_ce_holdout",         "sr_r0_new_gap_tl_ce_holdout_seed0_L4_lightning"),
    ("sr_r0_new_pstar_dice_holdout",        "sr_r0_new_pstar_dice_holdout_seed0"),
    ("sr_r0_new_pstar_sdice_holdout",       "sr_r0_new_pstar_sdice_holdout_seed0"),
    ("sr_r0_new_pstar_lcdice_holdout",      "sr_r0_new_pstar_lcdice_holdout_seed0"),
    ("sr_r0_new_gap_t2_ce_dice_holdout",    "sr_r0_new_gap_t2_ce_dice_holdout_seed0"),
    ("sr_r0_new_gap_t2_ce_sdice_holdout",   "sr_r0_new_gap_t2_ce_sdice_holdout_seed0"),
    ("sr_r0_new_gap_t2_ce_lcdice_holdout",  "sr_r0_new_gap_t2_ce_lcdice_holdout_seed0"),
    ("sr_r0_new_gapt4_pstar_dice_holdout",  "sr_r0_new_gapt4_pstar_dice_holdout_seed0"),
    ("sr_r0_new_gapt4_pstar_sdice_holdout", "sr_r0_new_gapt4_pstar_sdice_holdout_seed0"),
    ("sr_r0_new_gapt4_pstar_lcdice_holdout", "sr_r0_new_gapt4_pstar_lcdice_holdout_seed0"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--dataset-dir",
                    default="/Volumes/MAC_KIOXIA/Data/ROSA_New/ROSADataset")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--buffer-px", default="1,2,3,4,5")
    ap.add_argument("--ap-bins", type=int, default=101)
    ap.add_argument("--select-on", default="iou_mean",
                    help="which criterion's θ* to bench at, re-argmaxed from sweep.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.runs_dir)
    plan, missing = [], []
    for model_name, dirname in ARMS:
        d = root / dirname
        ck, sw = d / FINAL_CKPT, d / "sweep.json"
        if not (ck.is_file() and sw.is_file()):
            missing.append(f"{dirname} (ckpt={ck.is_file()} sweep={sw.is_file()})")
            continue
        s = json.loads(sw.read_text())["sweep"]
        have = {t: v[args.select_on] for t, v in s.items()
                if args.select_on in v and v[args.select_on] == v[args.select_on]}
        if not have:
            missing.append(f"{dirname} (no {args.select_on} in sweep.json)")
            continue
        theta = float(max(have, key=lambda t: have[t]))
        plan.append((model_name, d, theta))

    print(f"store    : {args.store_dir}")
    print(f"device   : {args.device}   buffer_px={args.buffer_px}  ap_bins={args.ap_bins}")
    print(f"θ* from  : {args.select_on}\n")
    for m, d, th in plan:
        print(f"  {m:<40} θ*={th:<6} {d.name}")
    for x in missing:
        print(f"  SKIP {x}")
    print(f"\n{len(plan)} arm(s) to bench, {len(missing)} skipped")
    if args.dry_run:
        return 0

    done, failed = [], []
    t_all = time.time()
    for i, (model_name, d, theta) in enumerate(plan, 1):
        cmd = [
            sys.executable, "-m", "benchmarking.cli", "eval",
            "--dataset-dir", args.dataset_dir,
            "--checkpoint", str(d / FINAL_CKPT),
            "--model", "sr", "--model-name", model_name, "--seed", "0",
            "--store-dir", args.store_dir, "--split", args.split,
            "--mask-source", "raster", "--mask-dirname", "mask_new_2pt5",
            "--exp-tag", "r0_new", "--label-source", "new",
            "--tile-metric", "apls", "--tile-metric", "cldice",
            "--threshold", str(theta), "--device", args.device,
            "--buffer-px", args.buffer_px, "--ap-bins", str(args.ap_bins),
        ]
        cfg = d / "best_params.yaml"
        if cfg.is_file():
            cmd += ["--config-yaml", str(cfg)]
        t0 = time.time()
        print(f"\n[{i}/{len(plan)}] {model_name}  θ*={theta}", flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        dt = time.time() - t0
        if rc == 0:
            done.append(model_name)
            print(f"    ok in {dt/60:.1f} min", flush=True)
        else:
            failed.append(f"{model_name} (rc={rc})")
            print(f"    FAILED rc={rc} after {dt/60:.1f} min", flush=True)

    print(f"\n=== {len(done)} benched, {len(failed)} failed, "
          f"{(time.time()-t_all)/60:.1f} min total ===")
    for f in failed:
        print(f"  FAILED {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
