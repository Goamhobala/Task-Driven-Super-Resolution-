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

and model_name (``<RUN_TAG>_ap``) and seed are parsed back out of the name.
Renaming a file silently re-labels the row it produces.

``manifest.json`` beside them, when present, is AUTHORITATIVE for model_name /
seed / exp_tag -- it was built by reading the arms' existing store rows, so the
new rows group with the old ones by construction. exp_tag in particular does
not follow one pattern (``r1b_new`` for the R series, ``r2grid_off_ls1e-4`` for
the grid), and a regex that guesses it would quietly file an arm under a tag
nothing else uses.

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


def guard_store(store_dir: Path, mask_dirname: str) -> None:
    """Refuse to mix LABEL GENERATIONS in one store.

    The test labels have now been through three generations at the SAME
    dataset path -- the original, the friends' relabelling, and the corrected
    set -- and dataset_dir / mask_source / gt_res_m / cell_m are identical
    across all three. The ONLY thing in the runs table that separates them is
    mask_dirname, so that is what this checks. Rows scored against different
    mask dirs are different quantities; `report` would average them into one
    mean without complaint, and the store is append-only with no dedupe, so a
    single stray append is unrecoverable.
    """
    runs_dir = store_dir / "runs"
    if not runs_dir.is_dir() or not any(runs_dir.glob("*.parquet")):
        return
    import pyarrow.dataset as ds

    have = set(ds.dataset(str(runs_dir), format="parquet")
               .to_table(columns=["mask_dirname"]).column("mask_dirname").to_pylist())
    other = {m for m in have if m and m != mask_dirname}
    if other:
        raise SystemExit(
            f"ERROR: {store_dir} already holds rows scored against {sorted(other)}, "
            f"but this run uses {mask_dirname!r}.\n"
            "  Different label generations are different quantities. Use a fresh --store-dir.")


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

    man_path = a.runs_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.is_file() else {}
    if manifest:
        print(f"manifest: {len(manifest)} entries ({man_path.name})")

    # Two layouts, both accepted:
    #   flat    <dir>/<TAG>.ckpt + <TAG>.sweep.json + <TAG>.best_params.yaml
    #   rundirs <dir>/<TAG>/checkpoints/*jointsr_final.ckpt + sweep.json + best_params.yaml
    # The rundirs form is what /scratch/.../runs already looks like, so seeds
    # that were fitted on the cluster need no staging or re-upload at all.
    sources = []          # (tag, ckpt, sweep, cfg)
    for ck in sorted(a.runs_dir.glob("*.ckpt")):
        t = ck.stem
        sources.append((t, ck, a.runs_dir / f"{t}.sweep.json",
                        a.runs_dir / f"{t}.best_params.yaml"))
    for d in sorted(a.runs_dir.glob("*/")):
        cks = sorted(d.glob("checkpoints/*jointsr_final.ckpt"))
        if cks:
            sources.append((d.name, cks[0], d / "sweep.json", d / "best_params.yaml"))
    if not sources:
        raise SystemExit(f"ERROR: no checkpoints found under {a.runs_dir} "
                         "(looked for *.ckpt and */checkpoints/*jointsr_final.ckpt)")
    if not a.dry_run:
        guard_store(a.store_dir, a.mask_dirname)

    tile_metrics = tuple(x for x in a.tile_metrics.split(",") if x)
    buffer_px = [float(x) for x in a.buffer_px.split(",") if x] or None
    if buffer_px and len(buffer_px) == 1:
        buffer_px = buffer_px[0]

    plan = []
    for tag, ck, sweep, cfg in sources:
        m = re.match(r"(.+)_seed(\d+)$", tag)
        if not m:
            print(f"SKIP {tag}: name does not end in _seed<N>")
            continue
        run_tag, seed = m.group(1), int(m.group(2))
        if a.arms and not any(x in tag for x in a.arms):
            continue
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
        entry = manifest.get(tag)
        if entry:
            model, seed_m, exp_tag = entry["model_name"], int(entry["seed"]), entry["exp_tag"]
            if seed_m != seed:
                raise SystemExit(f"ERROR: {tag}: manifest seed {seed_m} != filename seed {seed}")
        else:
            model = f"{run_tag}_ap"
            exp_tag = re.sub(r"^sr_(r[0-9]+[ab]?_new).*", r"\1", run_tag)
            if exp_tag == run_tag:      # the regex did not bite -- do not invent one
                raise SystemExit(
                    f"ERROR: {tag} is absent from manifest.json and its exp_tag "
                    "cannot be derived from the name. Re-stage so the manifest "
                    "covers it rather than letting it land under a wrong tag.")
        plan.append(dict(tag=tag, ckpt=ck, model=model, seed=seed, exp_tag=exp_tag,
                         theta=theta, ups=ups, cfg=cfg,
                         sen2sr_dir=(a.models_root / sub) if sub else None))

    print("=== R-SERIES RE-BENCH (no typer) ===")
    print(f"  runs   : {a.runs_dir}  ({len(plan)} to bench)")
    print(f"  store  : {a.store_dir}")
    print(f"  dataset: {a.dataset_dir}   split={a.split}   masks={a.mask_dirname}")
    print(f"  metrics: tile={tile_metrics} ap_bins={a.ap_bins} buffer={a.buffer_px}\n")
    for j in plan:
        print(f"  {j['model']:<58} seed={j['seed']:<4} ups={j['ups']:<8} "
              f"exp={j['exp_tag']:<20} theta={j['theta']}")
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
