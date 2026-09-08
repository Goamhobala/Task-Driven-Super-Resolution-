#!/usr/bin/env python
"""Re-bench the staged R-series checkpoints on the relabelled test split.

NO TYPER. This calls ``benchmarking.runner.evaluate`` directly rather than
going through ``benchmarking.cli``, which imports typer -- absent from the
cluster interpreter and not worth installing for a scoring job. Everything the
CLI would have done (θ* lookup, weights-dir resolution, the store guard) is
here in plain argparse + stdlib.

THE FILENAMES ARE THE INTERFACE. The staged dir is flat:

    <RUN_TAG>_seed<N>.ckpt
    <RUN_TAG>_seed<N>.sweep.json
    <RUN_TAG>_seed<N>.best_params.yaml

and model_name (``<RUN_TAG>_ap``), seed and exp_tag are parsed back out of the
name. Renaming a file silently re-labels the row it produces.

    python rebench_rseries.py --runs-dir ... --store-dir ... --dataset-dir ...
    python rebench_rseries.py ... --arms r4b r3a --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

# SR weights live in per-generator dirs and the loader opens a DIFFERENT file in
# each (sen2sr -> model.safetensor, sr4rs -> gen_weights.safetensors). Handing
# over the wrong one dies at the pre-flight check.
SR_DIRS = {"sen2sr": "SEN2SRLite_RGBN", "sr4rs": "SR4RS_RGBN", "bicubic": None}


def theta_of(path: Path) -> float | None:
    """θ* from a sweep.json, tolerating both schemas in the wild.

    theta_sweep_bench writes ``iou_mean``; _stages_tv.sh writes a bare ``iou``.
    Same quantity, verified equal on the runs that have both.
    """
    doc = json.loads(path.read_text())
    grid = doc.get("sweep", {})
    for key in ("iou_mean", "iou"):
        have = {t: v[key] for t, v in grid.items() if key in v and v[key] == v[key]}
        if have:
            return float(max(have, key=lambda t: have[t]))
    bt = doc.get("best_threshold")
    return float(bt) if bt is not None else None


def upsampler_of(cfg: Path) -> str:
    """Read the arm's upsampler off best_params.yaml.

    Raises rather than defaulting: a wrong guess here does not crash, it
    silently scores the arm through the wrong generator. An earlier shell
    version defaulted to sen2sr on any probe failure and would have benched all
    four SR4RS arms against SEN2SR weights.
    """
    import yaml

    doc = yaml.safe_load(cfg.read_text()) or {}
    ups = (doc.get("model") or {}).get("upsampler")
    if ups not in SR_DIRS:
        raise SystemExit(f"ERROR: {cfg.name}: unknown upsampler {ups!r} "
                         f"(known: {sorted(SR_DIRS)})")
    return ups


def guard_store(store_dir: Path) -> None:
    """Refuse to append new-label rows to a store holding old-label ones.

    Nothing in the runs table distinguishes them -- dataset_dir, mask_dirname,
    mask_source, gt_res_m and cell_m are all identical across the relabelling,
    because the dataset path was reused. The only tell is the tile count: the
    old test split had 181, the relabelled one has 174. The store is
    append-only with no dedupe, so one stray append blends both label sets into
    every cross-model mean, unrecoverably.
    """
    runs_dir = store_dir / "runs"
    if not runs_dir.is_dir() or not any(runs_dir.glob("*.parquet")):
        return
    # ONE dataset scan of a single column, not one read per shard: a mature
    # store is hundreds of files, and opening them individually is minutes on a
    # slow filesystem -- too slow for a check that must never be the reason
    # someone reaches for a way to skip it.
    import pyarrow.dataset as ds

    counts = set(ds.dataset(str(runs_dir), format="parquet")
                 .to_table(columns=["n_tiles"]).column("n_tiles").to_pylist())
    if 181 in counts:
        raise SystemExit(
            f"ERROR: {store_dir} already holds 181-tile (old-label) rows "
            f"(tile counts present: {sorted(counts)}).\n"
            "  Mixing label sets corrupts every mean. Use a fresh --store-dir.")


def already_done(store_dir: Path, model: str, seed: int, split: str) -> bool:
    """Skip work already in the target store, so a re-submit is cheap."""
    hits = glob.glob(str(store_dir / "runs" / f"{model}_seed{seed}_{split}_*.parquet"))
    return bool(hits)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", required=True, type=Path)
    p.add_argument("--store-dir", required=True, type=Path)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--models-root", required=True, type=Path,
                   help="parent of SEN2SRLite_RGBN / SR4RS_RGBN")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--mask-dirname", default="mask_new_2pt5")
    p.add_argument("--buffer-px", default="1,2,3,4,5")
    p.add_argument("--ap-bins", type=int, default=101)
    p.add_argument("--tile-metrics", default="apls,cldice")
    p.add_argument("--arms", nargs="*", default=None,
                   help="substring filter on the tag, e.g. --arms r4b r3a")
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    from benchmarking.runner import evaluate

    ckpts = sorted(a.runs_dir.glob("*.ckpt"))
    if not ckpts:
        raise SystemExit(f"ERROR: no *.ckpt under {a.runs_dir}")
    if not a.dry_run:
        guard_store(a.store_dir)

    tile_metrics = tuple(x for x in a.tile_metrics.split(",") if x)
    buffer_px = [float(x) for x in a.buffer_px.split(",") if x] or None
    if buffer_px and len(buffer_px) == 1:
        buffer_px = buffer_px[0]

    plan = []
    for ck in ckpts:
        tag = ck.stem
        m = re.match(r"(.+)_seed(\d+)$", tag)
        if not m:
            print(f"SKIP {tag}: name does not end in _seed<N>")
            continue
        run_tag, seed = m.group(1), int(m.group(2))
        if a.arms and not any(x in tag for x in a.arms):
            continue
        sweep, cfg = a.runs_dir / f"{tag}.sweep.json", a.runs_dir / f"{tag}.best_params.yaml"
        if not sweep.is_file():
            print(f"SKIP {tag}: no sweep.json")
            continue
        if not cfg.is_file():
            raise SystemExit(f"ERROR: {tag}: no best_params.yaml -- cannot resolve "
                             "the upsampler, and guessing would score it through "
                             "the wrong generator")
        theta = theta_of(sweep)
        if theta is None:
            print(f"SKIP {tag}: no theta in sweep.json")
            continue
        ups = upsampler_of(cfg)
        sub = SR_DIRS[ups]
        plan.append(dict(tag=tag, ckpt=ck, model=f"{run_tag}_ap", seed=seed,
                         exp_tag=re.sub(r"^sr_(r[0-9]+[ab]?_new).*", r"\1", run_tag),
                         theta=theta, ups=ups, cfg=cfg,
                         sen2sr_dir=(a.models_root / sub) if sub else None))

    print("=== R-SERIES RE-BENCH (no typer) ===")
    print(f"  runs   : {a.runs_dir}  ({len(plan)} to bench)")
    print(f"  store  : {a.store_dir}")
    print(f"  dataset: {a.dataset_dir}   split={a.split}")
    print(f"  metrics: tile={tile_metrics} ap_bins={a.ap_bins} buffer={a.buffer_px}\n")
    for j in plan:
        print(f"  {j['model']:<48} seed={j['seed']:<5} ups={j['ups']:<8} theta={j['theta']}")
    if a.dry_run:
        return 0

    # A missing weights dir is a 20-job failure discovered one job at a time;
    # check every one up front instead.
    for d in {j["sen2sr_dir"] for j in plan if j["sen2sr_dir"]}:
        if not Path(d).is_dir():
            raise SystemExit(f"ERROR: SR weights dir not found: {d}")

    t0, done, failed, skipped = time.time(), [], [], []
    for i, j in enumerate(plan, 1):
        if already_done(a.store_dir, j["model"], j["seed"], a.split):
            print(f"[{i}/{len(plan)}] skip  {j['tag']} (already in store)", flush=True)
            skipped.append(j["tag"])
            continue
        print(f"\n[{i}/{len(plan)}] {j['tag']}  theta={j['theta']}", flush=True)
        try:
            evaluate(dataset_dir=a.dataset_dir, checkpoint=j["ckpt"],
                     model_name=j["model"], seed=j["seed"], store_dir=a.store_dir,
                     split=a.split, model="sr", batch_size=a.batch_size,
                     mask_source="raster", mask_dirname=a.mask_dirname,
                     sen2sr_dir=j["sen2sr_dir"], config_yaml_path=j["cfg"],
                     exp_tag=j["exp_tag"], label_source="new",
                     tile_metrics=tile_metrics, device=a.device,
                     threshold=j["theta"], buffer_px=buffer_px, ap_bins=a.ap_bins)
            done.append(j["tag"])
        except Exception as e:                      # one bad arm must not sink 19 good ones
            import traceback
            traceback.print_exc()
            print(f"FAILED {j['tag']}: {e}", flush=True)
            failed.append(j["tag"])

    print(f"\n=== {len(done)} benched, {len(skipped)} skipped, {len(failed)} failed, "
          f"{(time.time() - t0) / 60:.1f} min ===")
    for f in failed:
        print(f"  FAILED {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
