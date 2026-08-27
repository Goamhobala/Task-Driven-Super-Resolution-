#!/usr/bin/env python
"""Sweep theta* and bench the nine gap_ce Phase B fits, N arms at a time.

WHY NOT rebench_all.py
----------------------
That driver is pinned to ``--seed 0`` and SKIPS any run without ``sweep.json``.
Both assumptions fail here: these runs are seeds 1-3, and the Phase B fit stage
never wrote a sweep, so all nine would be silently skipped. This one sweeps
first when the file is absent, then benches at the resulting theta*.

WHY THE RENAME
--------------
The run dirs are ``sr_r0_new_gapce_pstar_<compound>_holdout_seedN``, but every
other compound in the store is named ``<parent>_<compound>`` --
``sr_r0_new_gap_t2_ce_dice_holdout``. ``report`` groups on ``model_name``, so
without an explicit rename these would sort as a separate family instead of
completing the 4x3 parent-by-compound factorial. model_name carries no seed, so
the three seeds merge on their own.

WHY PARALLEL
------------
~86% of a bench is serial Python (APLS/clDice have no thread pool, rasterio
reads on the main thread), so one arm cannot use more than about one core.
Running arms as separate PROCESSES scales nearly linearly until the cores run
out -- 9 arms drop from ~3.75 h to ~1 h at 4 jobs. Keep --jobs at or below the
performance-core count: on a fanless M1 Air, oversubscribing just throttles.

    python scripts/local/bench_gapce.py --store-dir <store> --split val --jobs 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FINAL_CKPT = Path("checkpoints") / "unet_s2rosa_jointsr_final.ckpt"
DEF_RUNS = "/Volumes/MAC_KIOXIA/Data/InstaRoad/runslightning"
DEF_DATASET = "/Volumes/MAC_KIOXIA/Data/ROSA_New/ROSADataset"

# run-dir stem -> the model_name the store should see. The dirs are named by
# the engine's EXP_TAG/LOSS_ARM; every other compound in the store is
# <parent>_<region>, so the rename is what lets these join the parent-by-region
# grid instead of sorting as their own family.
ARMS = [
    ("sr_r0_new_gapce_pstar_dice",   "sr_r0_new_gap_ce_dice_holdout"),
    ("sr_r0_new_gapce_pstar_sdice",  "sr_r0_new_gap_ce_sdice_holdout"),
    ("sr_r0_new_gapce_pstar_lcdice", "sr_r0_new_gap_ce_lcdice_holdout"),
    ("sr_r0_new_tl_pstar_dice",      "sr_r0_new_tl_ce_dice_holdout"),
]
SEEDS = (1, 2, 3)


def arms(runs_dir: Path, only: str = ""):
    """--only is a duplicate GUARD, not a convenience. The store is append-only
    with uuid run_ids and no dedupe, so re-benching an arm that is already in it
    adds a SECOND shard and every mean silently averages those chips twice."""
    want = [w for w in only.replace(",", " ").split() if w]
    for seed in SEEDS:
        for stem, model_name in ARMS:
            if want and not any(w in stem or w in model_name for w in want):
                continue
            yield model_name, runs_dir / f"{stem}_holdout_seed{seed}", seed


# --- the missing THIRD test seed -------------------------------------------
# (run_dir, model_name, seed). Explicit triples, not a cross product: the third
# seed is 0 for the pilot fits and 3 for the six pool_d refits, and the seed-0
# dirs carry a platform suffix that no pattern can guess -- picking wrong would
# silently bench a different model from the one the table names.
#
# wbce:3 and wbce_dice:3 are ABSENT: those two refits live only on the cluster.
# Do them with scripts/hpc/loss/bench3/pool_test3.sh (ARMS="wbce:3 wbce_dice:3")
# or pull their run dirs down first.
THIRD_SEED = [
    ("sr_r0_new_dice_holdout_seed0_L4_lightning",      "sr_r0_new_dice_holdout", 0),
    ("sr_r0_new_sdice_holdout_seed0_L4_lightning",     "sr_r0_new_sdice_holdout", 0),
    ("sr_r0_new_lcdice_holdout_seed0_L4_lightning",    "sr_r0_new_lcdice_holdout", 0),
    ("sr_r0_new_tl_ce_holdout_seed0_L4_lightning",     "sr_r0_new_tl_ce_holdout", 0),
    ("sr_r0_new_gap_tl_ce_holdout_seed0_L4_lightning", "sr_r0_new_gap_tl_ce_holdout", 0),
    ("sr_r0_new_t2_ce_holdout_seed0_L4_modal",         "sr_r0_new_t2_ce_holdout", 0),
    ("sr_r0_new_t4_ce_holdout_seed0_L4_modal",         "sr_r0_new_t4_ce_holdout", 0),
    ("sr_r0_new_gap_t2_ce_holdout_seed0_L4_modal",     "sr_r0_new_gap_t2_ce_holdout", 0),
    ("sr_r0_new_gap_t4_ce_holdout_seed0_L4_modal",     "sr_r0_new_gap_t4_ce_holdout", 0),
    ("sr_r0_new_gap_t2t4_ce_holdout_seed0_L4_modal",   "sr_r0_new_gap_t2t4_ce_holdout", 0),
    ("sr_r0_new_gap_t2_ce_dice_holdout_seed0",         "sr_r0_new_gap_t2_ce_dice_holdout", 0),
    ("sr_r0_new_gap_t2_ce_sdice_holdout_seed0",        "sr_r0_new_gap_t2_ce_sdice_holdout", 0),
    ("sr_r0_new_gap_t2_ce_lcdice_holdout_seed0",       "sr_r0_new_gap_t2_ce_lcdice_holdout", 0),
    ("sr_r0_new_gapt4_pstar_dice_holdout_seed0",       "sr_r0_new_gapt4_pstar_dice_holdout", 0),
    ("sr_r0_new_gapt4_pstar_sdice_holdout_seed0",      "sr_r0_new_gapt4_pstar_sdice_holdout", 0),
    ("sr_r0_new_gapt4_pstar_lcdice_holdout_seed0",     "sr_r0_new_gapt4_pstar_lcdice_holdout", 0),
    ("sr_r0_new_gap_ce_holdout_seed3",                 "sr_r0_new_gap_ce_holdout", 3),
    ("sr_r0_new_gaptl_pstar_dice_holdout_seed3",       "sr_r0_new_gap_tl_dice_holdout", 3),
    ("sr_r0_new_gaptl_pstar_lcdice_holdout_seed3",     "sr_r0_new_gap_tl_ce_lcdice_holdout", 3),
    ("sr_r0_new_gaptl_pstar_sdice_holdout_seed3",      "sr_r0_new_gap_tl_ce_sdice_holdout", 3),
]


def already_in_store(store_dir, model_name, seed, split) -> bool:
    """The store is append-only with uuid run_ids and no dedupe, so a second row
    for the same (model, seed, split) makes every mean average those chips
    twice. This is what makes a restart safe after the drive drops mid-run."""
    try:
        sys.path.insert(0, str(REPO / "src"))
        from benchmarking.store import load_runs
        runs = load_runs(Path(store_dir))
    except Exception:
        return False
    if runs is None or getattr(runs, "empty", True) or "model_name" not in runs.columns:
        return False
    hit = runs[(runs["model_name"] == model_name) & (runs["seed"] == seed)]
    if "dataset_split" in runs.columns:
        hit = hit[hit["dataset_split"] == split]
    return len(hit) > 0


def theta_from_sweep(sweep: Path, select_on: str) -> float | None:
    """Re-argmax rather than trusting `best_threshold`: the stored value was
    chosen under whatever criterion the sweep ran with, which need not be the
    one being reported.

    TWO SCHEMAS EXIST and both are legitimate:
      theta_sweep_bench (local) writes iou_mean / f1_mean / *_micro per theta;
      _stages_tv.sh (HPC) writes the bare iou / f1.
    Both select on macro IoU over val -- verified by `criterion: iou` vs
    `selected_on: iou_mean`, and in both files best_threshold == argmax(iou).
    So the bare name is an ALIAS, not a different quantity, and falling back to
    it keeps a seed-3 row on the same operating-point rule as its seed-0
    siblings instead of failing outright.
    """
    doc = json.loads(sweep.read_text())
    grid = doc.get("sweep", {})
    for key in (select_on, select_on.removesuffix("_mean")):
        have = {t: v[key] for t, v in grid.items()
                if key in v and v[key] == v[key]}
        if have:
            return float(max(have, key=lambda t: have[t]))
    # Last resort: the sweep's own recorded choice. Better than refusing, and
    # it is what _stages_tv.sh's bench stage uses for the seed-1/2 rows.
    bt = doc.get("best_threshold")
    return float(bt) if bt is not None else None


def one_arm(model_name: str, d: Path, seed: int, args) -> tuple[str, int, str]:
    """Sweep (if needed) then bench. Returns (label, rc, note)."""
    label = f"{model_name.replace('sr_r0_new_', '').replace('_holdout', '')} seed{seed}"
    log = Path(args.log_dir) / f"{d.name}_{args.split}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)

    with log.open("w") as fh:
        sweep = d / "sweep.json"
        if not sweep.is_file():
            # theta* is always selected on val, whatever split we then bench.
            cmd = [sys.executable, "-m", "scripts.local.theta_sweep_bench",
                   "--run-dir", str(d), "--model-name", model_name,
                   "--exp-tag", "r0_new", "--seed", str(seed),
                   "--dataset-dir", args.dataset_dir,
                   "--store-dir", str(Path(args.store_dir).parent / "_sweep_scratch"),
                   "--device", args.device, "--select-on", args.select_on,
                   # theta_sweep_bench runs sweep -> bench -> report by default.
                   # We only want sweep.json here; the bench below is the one
                   # whose rows go to the real store. Without this every arm is
                   # scored TWICE and the second pass is thrown away -- which is
                   # why the nine-arm val pass took 6.3 h rather than ~3.
                   "--skip-bench"]
            fh.write(f"### SWEEP {' '.join(cmd)}\n"); fh.flush()
            rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                return label, rc, "sweep failed"
        if not sweep.is_file():
            return label, 1, "sweep.json still absent"

        theta = theta_from_sweep(sweep, args.select_on)
        if theta is None:
            return label, 1, f"no {args.select_on} in sweep.json"

        cmd = [sys.executable, "-m", "benchmarking.cli", "eval",
               "--dataset-dir", args.dataset_dir,
               "--checkpoint", str(d / FINAL_CKPT),
               "--model", "sr", "--model-name", model_name, "--seed", str(seed),
               "--store-dir", args.store_dir, "--split", args.split,
               "--mask-source", "raster", "--mask-dirname", "mask_new_2pt5",
               "--exp-tag", "r0_new", "--label-source", "new",
               "--tile-metric", "apls", "--tile-metric", "cldice",
               "--threshold", str(theta), "--device", args.device,
               "--buffer-px", args.buffer_px, "--ap-bins", str(args.ap_bins)]
        cfg = d / "best_params.yaml"
        if cfg.is_file():
            cmd += ["--config-yaml", str(cfg)]
        fh.write(f"\n### BENCH theta*={theta}\n{' '.join(cmd)}\n"); fh.flush()
        rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    return label, rc, f"theta*={theta}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default=DEF_RUNS)
    p.add_argument("--store-dir", required=True)
    p.add_argument("--dataset-dir", default=DEF_DATASET)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="mps")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--buffer-px", default="1,2,3,4,5")
    p.add_argument("--ap-bins", type=int, default=101)
    p.add_argument("--select-on", default="iou_mean")
    p.add_argument("--log-dir", default="/tmp/bench_gapce_logs")
    p.add_argument("--only", default="",
                   help="substring filter on run-dir stem or model_name. REQUIRED in "
                        "practice: the store has no dedupe, so re-benching an arm "
                        "already in it double-counts its chips.")
    p.add_argument("--third-seed", action="store_true",
                   help="bench the missing THIRD seed for the 22-arm table "
                        "(seed 0 for pilot fits, seed 3 for pool_d refits)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    plan, missing, dup = [], [], []
    src = ([(m, Path(args.runs_dir) / dn, sd) for dn, m, sd in THIRD_SEED]
           if args.third_seed else list(arms(Path(args.runs_dir), args.only)))
    for model_name, d, seed in src:
        if already_in_store(args.store_dir, model_name, seed, args.split):
            dup.append(f"{model_name} seed{seed} (already in {args.split} store)")
            continue
        ck = d / FINAL_CKPT
        if not ck.is_file():
            missing.append(f"{d.name}: no final ckpt")
            continue
        # A fit that stopped short is NOT comparable to the 50-epoch arms it
        # would sit beside, and `final.ckpt` exists from epoch 0 onward (the
        # callback is monitor:null/save_top_k:1), so its presence proves
        # nothing. Check the epoch or do not bench it.
        try:
            import torch
            ep = torch.load(ck, map_location="cpu", weights_only=False).get("epoch")
        except Exception as exc:
            missing.append(f"{d.name}: unreadable ckpt ({exc})")
            continue
        # The epoch guard exists because a half-trained fit is not comparable
        # to the 50-epoch arms beside it. It does NOT apply to --third-seed:
        # those checkpoints are the ALREADY-PUBLISHED seed-0/3 models whose val
        # rows are in the table, so whatever epoch they stopped at is the model
        # being reported, not a truncation introduced here.
        if ep != 49 and not args.third_seed:
            missing.append(f"{d.name}: epoch {ep}/49 — INCOMPLETE, refusing to bench")
            continue
        plan.append((model_name, d, seed))

    print(f"store  : {args.store_dir}\nsplit  : {args.split}   device={args.device} "
          f"jobs={args.jobs}\ntheta* : {args.select_on}\n")
    for m, d, s in plan:
        print(f"  {m:<38} seed{s}  {d.name}")
    for x in dup:
        print(f"  SKIP {x}")
    for x in missing:
        print(f"  SKIP {x}")
    print(f"\n{len(plan)} to bench, {len(missing)} skipped")
    if args.dry_run or not plan:
        return 0 if not missing else 1

    t0 = time.time()
    done, failed = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(one_arm, m, d, s, args): m for m, d, s in plan}
        for i, f in enumerate(as_completed(futs), 1):
            label, rc, note = f.result()
            if rc == 0:
                done.append(label)
                print(f"[{i}/{len(plan)}] ok      {label}  ({note})", flush=True)
            else:
                failed.append(f"{label} rc={rc} {note}")
                print(f"[{i}/{len(plan)}] FAILED  {label}  ({note})", flush=True)

    print(f"\n=== {len(done)} benched, {len(failed)} failed, "
          f"{(time.time()-t0)/60:.1f} min total (jobs={args.jobs}) ===")
    for f in failed:
        print(f"  FAILED {f}")
    print(f"logs: {args.log_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
