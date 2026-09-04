#!/usr/bin/env python
"""Re-bench a run series with the FULL metric set (buffered rho=1..5, AP, clDice).

WHY RE-BENCH RATHER THAN BACK-FILL
----------------------------------
Buffered scores need the per-chip prediction MASK and AP needs the probability
MAP. The store keeps neither -- only tp/fp/fn/tn -- so these columns cannot be
derived from existing rows. Scoring the checkpoint again is the only route.

WHY A NEW STORE
---------------
--store-dir MUST NOT be the store that already holds these models. It is
append-only with uuid run_ids and no dedupe, so a second row for the same
(model, seed, split) makes every mean average those chips twice. Write to a
fresh directory and swap it in once you have checked it.

    python scripts/local/bench_series.py \
        --runs-dir .../SRruns/refits --store-dir .../benchmarks_buffered \
        --pattern 'sr_r*_new_*' --jobs 3
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEF_DATASET = "/Volumes/MAC_KIOXIA/Data/ROSA_New/ROSADataset"


def discover(runs_dir: Path, pattern: str, suffix: str = ""):
    """(run_dir, model_name, seed, ckpt) for each matching run.

    model_name drops the trailing _seedN and appends --model-suffix. The R
    series is named ..._ap in the store (AP-theta* protocol) so it needs
    "_ap"; the rlruns family is ..._rails_holdout and takes no suffix.
    Getting this wrong silently creates a NEW model rather than matching the
    existing rows.
    """
    out = []
    for d in sorted(runs_dir.glob(pattern)):
        if not d.is_dir():
            continue
        m = re.search(r"_seed(\d+)$", d.name)
        if not m:
            continue
        seed = int(m.group(1))
        # The final checkpoint is NOT always unet_*: r2a seed66 writes
        # r2a_s2rosa_jointsr_final.ckpt. Glob the suffix, never the prefix.
        ck = sorted(d.glob("checkpoints/*jointsr_final.ckpt"))
        if not ck:
            out.append((d, None, seed, None))
            continue
        model = re.sub(r"_seed\d+$", suffix, d.name)
        out.append((d, model, seed, ck[0]))
    return out


def exp_tag(model: str) -> str:
    """sr_r2b_new_nohc_gap_ce_... -> r2b_new"""
    m = re.match(r"sr_(r[0-9a-z]+_new)", model)
    return m.group(1) if m else "r0_new"


def theta_of(d: Path, select_on: str) -> float | None:
    f = d / "sweep.json"
    if not f.is_file():
        return None
    doc = json.loads(f.read_text())
    grid = doc.get("sweep", {})
    for key in (select_on, select_on.removesuffix("_mean")):
        have = {t: v[key] for t, v in grid.items()
                if key in v and v[key] == v[key]}
        if have:
            return float(max(have, key=lambda t: have[t]))
    bt = doc.get("best_threshold")
    return float(bt) if bt is not None else None


def in_store(store: str, model: str, seed: int, split: str) -> bool:
    try:
        sys.path.insert(0, str(REPO / "src"))
        from benchmarking.store import load_runs
        runs = load_runs(Path(store))
    except Exception:
        return False
    if runs is None or getattr(runs, "empty", True) or "model_name" not in runs.columns:
        return False
    hit = runs[(runs["model_name"] == model) & (runs["seed"] == seed)]
    if "dataset_split" in runs.columns:
        hit = hit[hit["dataset_split"] == split]
    return len(hit) > 0


def one(d: Path, model: str, seed: int, ck: Path, args):
    label = f"{model.replace('sr_', '').replace('_gap_ce_anorm_recalpost_ap', '')} s{seed}"
    log = Path(args.log_dir) / f"{d.name}_{args.split}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        theta = theta_of(d, args.select_on)
        if theta is None:
            # No sweep.json (r2a seed1). Produce one the same way its siblings
            # got theirs -- selection on val -- rather than borrowing another
            # seed's theta, which would not be that run's operating point.
            cmd = [sys.executable, "-m", "scripts.local.theta_sweep_bench",
                   "--run-dir", str(d), "--model-name", model,
                   "--exp-tag", exp_tag(model), "--seed", str(seed),
                   "--dataset-dir", args.dataset_dir,
                   "--store-dir", str(Path(args.store_dir).parent / "_sweep_scratch"),
                   "--device", args.device, "--select-on", args.select_on,
                   "--sen2sr-dir", args.sen2sr_dir, "--skip-bench"]
            fh.write(f"### SWEEP {' '.join(cmd)}\n"); fh.flush()
            rc = subprocess.run(cmd, cwd=REPO, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
            if rc != 0:
                return label, rc, "sweep failed"
            theta = theta_of(d, args.select_on)
            if theta is None:
                return label, 1, "sweep produced no theta"

        cmd = [sys.executable, "-m", "benchmarking.cli", "eval",
               "--dataset-dir", args.dataset_dir,
               "--checkpoint", str(ck),
               "--model", "sr", "--model-name", model, "--seed", str(seed),
               "--store-dir", args.store_dir, "--split", args.split,
               "--mask-source", "raster", "--mask-dirname", "mask_new_2pt5",
               "--exp-tag", exp_tag(model), "--label-source", "new",
               "--tile-metric", "apls", "--tile-metric", "cldice",
               # The checkpoint bakes in the SR weights path it was TRAINED
               # with -- /scratch/... on the cluster, which does not exist here.
               # Only r0 survives without this because r0 is bicubic and loads
               # no SR network at all; every r1/r2/rl1/rl2 run dies on
               # FileNotFoundError for model.safetensor.
               "--sen2sr-dir", args.sen2sr_dir,
               "--threshold", str(theta), "--device", args.device,
               "--buffer-px", args.buffer_px, "--ap-bins", str(args.ap_bins),
               # Chips per forward pass. The ONLY safe OOM lever: --cell-m is
               # the protocol footprint (2560 m) and changing it would make the
               # rows incomparable with every other arm in the store.
               "--batch-size", str(args.batch_size)]
        cfg = d / "best_params.yaml"
        if cfg.is_file():
            cmd += ["--config-yaml", str(cfg)]
        fh.write(f"\n### BENCH theta*={theta}\n{' '.join(cmd)}\n"); fh.flush()
        rc = subprocess.run(cmd, cwd=REPO, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    return label, rc, f"theta*={theta}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--store-dir", required=True)
    p.add_argument("--pattern", default="sr_*")
    p.add_argument("--model-suffix", default="",
                   help="appended after stripping _seedN. '_ap' for the R series.")
    p.add_argument("--dataset-dir", default=DEF_DATASET)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="mps")
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--buffer-px", default="1,2,3,4,5")
    p.add_argument("--ap-bins", type=int, default=101)
    p.add_argument("--batch-size", type=int, default=8,
                   help="chips per forward pass; lower it (4, 2, 1) on OOM. Does "
                        "NOT affect the scores, only peak memory.")
    p.add_argument("--select-on", default="iou_mean")
    p.add_argument("--sen2sr-dir",
                   default="/Volumes/MAC_KIOXIA/InstaRoadPrototype/models/SEN2SRLite_RGBN",
                   help="local SR weights dir; overrides the cluster path baked "
                        "into every checkpoint. Must be the MODEL directory "
                        "(the loader opens <dir>/model.safetensor), not its "
                        "parent. upsampler=sen2sr -> SEN2SRLite_RGBN, which is "
                        "every SR run here except rl4 (sr4rs, no ckpt anyway).")
    p.add_argument("--log-dir", default="/tmp/bench_series_logs")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    plan, skip = [], []
    for d, model, seed, ck in discover(Path(args.runs_dir), args.pattern,
                                       args.model_suffix):
        if model is None or ck is None:
            skip.append(f"{d.name}: no *jointsr_final.ckpt")
            continue
        if in_store(args.store_dir, model, seed, args.split):
            skip.append(f"{model} s{seed}: already in target store")
            continue
        plan.append((d, model, seed, ck))

    print(f"store  : {args.store_dir}\nsplit  : {args.split}   device={args.device} "
          f"jobs={args.jobs}\nmetrics: buffered rho={args.buffer_px}, ap_bins={args.ap_bins}, "
          f"apls+cldice\n")
    for d, m, s, _ in plan:
        t = theta_of(d, args.select_on)
        print(f"  {m:<46} s{s:<5} theta={t if t is not None else 'SWEEP NEEDED'}")
    for x in skip:
        print(f"  SKIP {x}")
    print(f"\n{len(plan)} to bench, {len(skip)} skipped")
    if args.dry_run or not plan:
        return 0

    t0, done, failed = time.time(), [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(one, d, m, s, c, args) for d, m, s, c in plan]
        for i, f in enumerate(as_completed(futs), 1):
            label, rc, note = f.result()
            (done if rc == 0 else failed).append(label)
            print(f"[{i}/{len(plan)}] {'ok    ' if rc == 0 else 'FAILED'} {label}  ({note})",
                  flush=True)
    print(f"\n=== {len(done)} benched, {len(failed)} failed, "
          f"{(time.time() - t0) / 60:.1f} min ===")
    for f in failed:
        print(f"  FAILED {f}")
    print(f"logs: {args.log_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    raise SystemExit(main())
