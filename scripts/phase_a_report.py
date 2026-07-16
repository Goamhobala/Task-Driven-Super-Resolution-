#!/usr/bin/env python3
"""Decision-A table: collect loss-ablation runs and print them against the
bce_dice seed-noise band (protocol: margins inside the band = tie -> simpler
loss). Reads the loss-ablation run dirs the staged engine writes
(``<runs>/loss_*/train_meta.json``; sweep.json is folded into train_meta).

Run:  python scripts/phase_a_report.py --runs /scratch/$USER/InstaRoad/runs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()

    # The staged engine writes run dirs as runs/loss_<tag>_seed<n>/; match those
    # (and tolerate being pointed straight at a runs/phase_a-style subdir).
    rows = []
    for pattern in ("loss_*/train_meta.json", "*/train_meta.json"):
        for meta_path in sorted(Path(args.runs).glob(pattern)):
            rows.append(json.loads(meta_path.read_text()))
        if rows:
            break
    if not rows:
        raise SystemExit(f"no loss_*/train_meta.json under {args.runs}")

    # seed-noise band from the bce_dice anchor replicates (val F1 @ tuned θ)
    anchor = [r["f1_at_best_threshold"] for r in rows if r["arm"] == "bce_dice"]
    band = (max(anchor) - min(anchor)) if len(anchor) >= 2 else None

    print(f"{'run':30s} {'arm':14s} {'θ*':5s} {'F1@θ*':8s} {'F1@0.5':8s} {'best_ep':7s}")
    print("-" * 78)
    for r in sorted(rows, key=lambda r: -r["f1_at_best_threshold"]):
        print(f"{r['run_name']:30s} {r['arm']:14s} {r['best_threshold']:<5} "
              f"{r['f1_at_best_threshold']:<8.4f} {r['best_val_f1']:<8.4f} "
              f"{r['best_epoch']:<7d}")

    if band is not None:
        mean_anchor = sum(anchor) / len(anchor)
        print(f"\nbce_dice anchor: n={len(anchor)}  mean F1@θ* {mean_anchor:.4f}  "
              f"seed band (max-min) {band:.4f}")
        print("Decision rule: a Phase A arm must clear the best anchor-comparable "
              "number by MORE than the band to count as a real win; otherwise tie "
              "-> prefer the simpler loss.")
    else:
        print("\n(no bce_dice replicates found yet — run array tasks 7-9 for the band)")
    print("\nNB pixel F1/IoU only. Connectivity comes from the benchmark step "
          "(STAGE=bench scores --tile-metric apls by default); fold it in with:\n"
          "  python -m benchmarking.cli report --store-dir <benchmarks_loss> "
          "--metric f1 --metric iou --metric apls\n"
          "Decision A weighs pixel AND apls jointly (protocol composite).")


if __name__ == "__main__":
    main()
