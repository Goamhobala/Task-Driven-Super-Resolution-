#!/usr/bin/env python
"""Per-arm θ sweep + final bench for the 2.5 m loss pilot, run LOCALLY.

WHY THIS EXISTS
---------------
You are right that there is no ``sweep.json`` and that every arm is being
scored at θ = 0.5. The cause is structural, not a missing flag:

  * ``unet/train_ablation.py`` (the 10 m Phase A harness) ends with a val
    threshold sweep over θ = 0.05…0.95 and writes ``sweep.json`` +
    ``train_meta.json['best_threshold']``.
  * The 2.5 m pilot runs a different path entirely — ``sr/_stages_tv.sh`` →
    ``sr.cli fit`` → ``benchmarking.cli eval``. That path has NO sweep stage.
    ``benchmarking.runner`` therefore falls back to
    ``self.threshold = float(hp.get("threshold", 0.5))``, and the SR training
    configs never set a ``threshold`` hparam. Hence 0.5, for every arm.

The sweep was never ported when the pilot moved to 2.5 m. This script is that
missing stage, run after the fact against finished checkpoints.

WHY IT MATTERS FOR THE RESULT (not just tidiness)
-------------------------------------------------
θ* is loss-dependent by construction. An arm trained at λ ≈ 15 emits
systematically higher road probabilities than a plain Dice arm; its optimal
operating point sits well away from 0.5. Scoring every arm at a COMMON 0.5
therefore confounds probability calibration with segmentation quality, and it
penalises exactly the high-λ arms the pilot is trying to evaluate. Protocol
v2.1 already says "Bench on val at the arm's θ* (sweep 0.05–0.95)" — the
current tables do not implement that sentence.

Two honest caveats to record in the amendment log:
  1. θ* is selected on val and the pilot also REPORTS on val. Those numbers are
     optimistically biased. Every arm is biased the same way, so the RANKING
     stands; the absolute values are not generalisation estimates. (Test is
     untouched, as intended.)
  2. ``train_ablation`` picked θ* by argmax of MICRO-averaged F1 (counts pooled
     over all pixels). The pilot's statistics are chip-paired, so this script
     selects on the MACRO mean of per-chip IoU by default — the same quantity
     the report ranks on. Both micro and macro figures are written into
     sweep.json so the choice is auditable, and ``--select-on`` switches it.

WHAT IT DOES
------------
For every ``sr_<exp>_<tag>_holdout_seed<N>/`` dir under --runs-dir that has a
final checkpoint:

  1. sweep   coarse θ grid, then a fine grid around the winner, scoring each θ
             through ``benchmarking.runner.evaluate`` with tile metrics OFF,
             into a scratch store. Writes ``sweep.json`` in the run dir.
  2. bench   ONE final evaluate() at θ*, with apls + cldice, into the real
             store — matching exactly what ``_stages_tv.sh`` STAGE=bench emits
             (same model_name, exp_tag, label_source, mask source, split).
  3. report  ``benchmarking.cli report`` over the store.

Each θ costs a full inference pass. This is deliberate: it reuses the scoring
path verbatim rather than reimplementing the chip/window/mask logic, which is
where a hand-rolled sweep would silently diverge. See HANDOVER.md for the
one-pass optimisation if it turns out too slow.

USAGE
-----
    python scripts/local/theta_sweep_bench.py --dry-run          # plan only
    python scripts/local/theta_sweep_bench.py --max-tiles 4      # smoke test
    python scripts/local/theta_sweep_bench.py                    # real run

Device defaults to mps on Apple silicon, else cuda, else cpu.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Defaults — these mirror scripts/LightningStudio/sr/_stages_tv.sh in pilot
# (holdout) mode. Change them here and the emitted rows stop matching the
# existing shards, so don't, unless you mean to.
#
# PATHS FOLLOW THE ORCHESTRATOR'S ENVIRONMENT. All three pilot orchestrators
# (LightningStudio/hpc `_stages_tv.sh`, `pilot_kaggle.sh`, `pilot_modal.sh`)
# export the same three variables, so exporting them once — or running inside
# the Modal container, where modal_app._base_env sets them — is enough to point
# this script at the right place. Local laptop paths are the last resort.
#
#   DATASET_DIR  the dir CONTAINING splits/ (local: …/ROSA_New/ROSADataset,
#                Modal: /data/ROSA_New — same meaning, different layout depth)
#   RUNS_ROOT    where the sr_<exp>_<tag>_holdout_seed<N>/ dirs live
#   SEN2SR_DIR   SR weights, for --sen2sr-dir (r0 is bicubic and needs none)
#
# STORE_DIR is deliberately NOT taken as-is: on Modal it points at
# benchmarks_loss_pilot, which already holds the θ=0.5 shards, and θ* rows must
# not land in the same store (the store is append-only, and mixing protocols in
# one table is exactly what the amendment log warns against). So a bare
# STORE_DIR gets a `_theta` suffix; set THETA_STORE_DIR to override outright.
# --------------------------------------------------------------------------- #
def _default_store() -> str:
    if os.environ.get("THETA_STORE_DIR"):
        return os.environ["THETA_STORE_DIR"]
    if os.environ.get("STORE_DIR"):
        return os.environ["STORE_DIR"].rstrip("/") + "_theta"
    return "/Volumes/MAC_KIOXIA/Data/benchmarks_loss_pilot_theta"


DEF_RUNS_DIR = os.environ.get("RUNS_ROOT") or "/Volumes/MAC_KIOXIA/Data/runslightning"
DEF_DATASET = os.environ.get("DATASET_DIR") or "/Volumes/MAC_KIOXIA/Data/ROSA_New/ROSADataset"
DEF_STORE = _default_store()
DEF_SEN2SR = os.environ.get("SEN2SR_DIR") or None

SPLIT = "val"                     # pilot decision split; test stays unseen
MODEL_FAMILY = "sr"
MASK_SOURCE = "raster"            # LABELS=new
MASK_DIRNAME = "mask_new_2pt5"
LABEL_SOURCE = "new"
TILE_METRICS = ("apls", "cldice")
FINAL_CKPT = Path("checkpoints") / "unet_s2rosa_jointsr_final.ckpt"

# Known loss tags, longest-first so `gap_t2t4_ce` wins over `t2_ce` etc. when
# splitting `sr_r0_new_<tag>_holdout`. Mirrors the TAG tables in the three
# pilot orchestrators.
TAGS = sorted(
    ["bce", "gap_ce", "tl_ce", "gap_tl_ce", "wbce", "sdice", "lcdice",
     "balance_ce", "dice", "t2_ce", "t4_ce", "gap_t2_ce", "gap_t4_ce",
     "gap_t2t4_ce", "pstar_dice", "pstar_sdice", "pstar_lcdice"],
    key=len, reverse=True,
)


def parse_run_dir(name: str) -> dict | None:
    """`sr_r0_new_gap_tl_ce_holdout_seed0[_L4_modal]` -> exp/tag/seed/platform.

    ``_stages_tv.sh`` emits the canonical name with no trailing platform tag;
    the optional suffix is added by hand on download to keep runs of the same
    arm from different machines apart (``_L4_lightning``, ``_T4_kaggle``,
    ``_L4_modal``). It is NOT part of ``model_name`` — the emitted rows must
    stay shape-compatible with the shards the pilot already published.
    """
    m = re.match(r"^sr_(?P<mid>.+)_holdout_seed(?P<seed>\d+)(?:_(?P<plat>.+))?$", name)
    if not m:
        return None
    mid, seed, plat = m["mid"], int(m["seed"]), m["plat"] or ""
    for tag in TAGS:                       # longest match first
        if mid.endswith("_" + tag):
            return {"exp": mid[: -len(tag) - 1], "tag": tag, "seed": seed,
                    "platform": plat}
    return None


def discover(runs_dir: Path) -> list[dict]:
    out = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        parsed = parse_run_dir(d.name)
        if parsed is None:
            continue
        ckpt = d / FINAL_CKPT
        out.append({
            "run_dir": d,
            "checkpoint": ckpt if ckpt.is_file() else None,
            "model_name": f"sr_{parsed['exp']}_{parsed['tag']}_holdout",
            "exp_tag": parsed["exp"],
            "tag": parsed["tag"],
            "seed": parsed["seed"],
            "platform": parsed["platform"],
            "config_yaml": (d / "best_params.yaml") if (d / "best_params.yaml").is_file() else None,
        })
    return out


def resolve_duplicates(specs: list[dict], prefer: str) -> tuple[list[dict], list[str]]:
    """One arm trained on two machines -> two dirs, ONE model_name.

    The store keys on (model_name, seed), so benching both would write two
    shards for the same arm and every per-model mean would average duplicated
    chips (the append-only uuid trap). Keep the preferred platform, report the
    rest — never pick silently.
    """
    groups: dict[tuple[str, int], list[dict]] = {}
    for s in specs:
        groups.setdefault((s["model_name"], s["seed"]), []).append(s)
    kept, dropped = [], []
    for (name, seed), g in groups.items():
        if len(g) == 1:
            kept.append(g[0])
            continue
        pick = next((s for s in g if s["platform"] == prefer), None) or g[0]
        kept.append(pick)
        for s in g:
            if s is not pick:
                dropped.append(f"{name} seed={seed} [{s['platform'] or 'no-platform'}]"
                               f" (kept [{pick['platform'] or 'no-platform'}])")
    return kept, dropped


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def theta_grid(step: float, lo: float, hi: float):
    """The full sweep grid. evaluate() requires 0 < θ < 1 strictly, so it never
    touches 0 or 1. Every θ is scored off ONE forward pass, so there is no
    reason to go coarse-to-fine — sweep the whole grid at full resolution."""
    n = int(round((hi - lo) / step))
    return [t for t in (round(lo + i * step, 4) for i in range(n + 1)) if 0.0 < t < 1.0]


def summarise(c) -> dict:
    """Per-chip rows at one θ -> the summary recorded in sweep.json."""
    tp, fp, fn = float(c["tp"].sum()), float(c["fp"].sum()), float(c["fn"].sum())
    eps = 1e-9
    return {
        # macro = mean over chips; this is what report/Wilcoxon rank on
        "iou_mean": float(c["iou"].mean()),
        "f1_mean": float(c["f1"].mean()),
        "precision_mean": float(c["precision"].mean()),
        "recall_mean": float(c["recall"].mean()),
        # micro = counts pooled over all pixels; what train_ablation used
        "iou_micro": tp / (tp + fp + fn + eps),
        "f1_micro": 2 * tp / (2 * tp + fp + fn + eps),
        "n_chips": int(len(c)),
    }


def sweep_arm(spec: dict, args) -> dict:
    """Score the whole θ grid off a single inference pass -> sweep.json."""
    from benchmarking.runner import evaluate

    key = args.select_on                       # e.g. "iou_mean"
    grid = theta_grid(args.step, args.lo, args.hi)
    print(f"  sweeping {len(grid)} θ ({grid[0]}..{grid[-1]} step {args.step}) "
          f"in one pass")

    t0 = time.time()
    per_theta = evaluate(
        dataset_dir=Path(args.dataset_dir),
        checkpoint=spec["checkpoint"],
        model_name=spec["model_name"],
        seed=spec["seed"],
        store_dir=None,                   # sweep mode writes nothing
        split=SPLIT,
        model=MODEL_FAMILY,
        mask_source=MASK_SOURCE,
        mask_dirname=MASK_DIRNAME,
        config_yaml_path=spec["config_yaml"],
        sen2sr_dir=Path(args.sen2sr_dir) if args.sen2sr_dir else None,
        exp_tag=spec["exp_tag"],
        label_source=LABEL_SOURCE,
        tile_metrics=(),                  # θ-dependent; rejected in sweep mode
        check="first",
        device=args.device,
        sweep_thresholds=grid,
        max_tiles=args.sweep_max_tiles or args.max_tiles,
    )
    results = {t: summarise(c) for t, c in per_theta.items()}
    print(f"  [{time.time() - t0:.0f}s for the whole grid]")
    for t in sorted(results):
        print(f"    θ={t:<6} {key}={results[t][key]:.4f}")

    best = max(results, key=lambda t: results[t][key])
    edge = best in (grid[0], grid[-1])
    payload = {
        "run": spec["model_name"],
        "seed": spec["seed"],
        "platform": spec["platform"],
        "source": "scripts/local/theta_sweep_bench.py",
        "split": SPLIT,
        "selected_on": key,
        "aggregation": "macro (mean over chips)" if key.endswith("_mean") else "micro (pooled counts)",
        "method": "single inference pass, all θ scored off the same probs",
        "grid": {"lo": args.lo, "hi": args.hi, "step": args.step, "n": len(grid)},
        "sweep_max_tiles": args.sweep_max_tiles or args.max_tiles,
        "best_threshold": best,
        "best_on_grid_edge": edge,
        "sweep": {f"{t:.4f}": v for t, v in sorted(results.items())},
    }
    (spec["run_dir"] / "sweep.json").write_text(json.dumps(payload, indent=1))
    flag = "  ** ON GRID EDGE — true optimum is outside the swept range **" if edge else ""
    print(f"  θ* = {best}  ({key}={results[best][key]:.4f})"
          f"  -> {spec['run_dir'] / 'sweep.json'}{flag}")
    return payload


def already_in_store(store_dir: Path, model_name: str, seed: int) -> bool:
    """Guard against the append-only double-count trap.

    run_id is uuid4 and _write_shard refuses to overwrite, so re-benching an
    arm into a store that already has it produces a SECOND shard and every mean
    silently averages duplicated chips.
    """
    try:
        from benchmarking.store import load_runs
        runs = load_runs(store_dir)
    except Exception:          # empty/absent store — nothing to collide with
        return False
    if runs is None or getattr(runs, "empty", True):
        return False
    if "model_name" not in runs.columns:
        return False
    hit = runs[(runs["model_name"] == model_name) & (runs["seed"] == seed)]
    if "dataset_split" in runs.columns:
        hit = hit[hit["dataset_split"] == SPLIT]
    return len(hit) > 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default=DEF_RUNS_DIR,
                   help="Dir of run dirs to discover [default: $RUNS_ROOT]")
    p.add_argument("--run-dir", default=None,
                   help="Explicit SINGLE run dir — bypasses --runs-dir discovery "
                        "and the TAGS name-parsing entirely (this is how the "
                        "_stages_tv.sh bench stage invokes the sweep, so it works "
                        "for any current or future arm name).")
    p.add_argument("--model-name", default=None, help="with --run-dir: the bench model_name")
    p.add_argument("--exp-tag", default=None, help="with --run-dir: the bench exp_tag")
    p.add_argument("--seed", type=int, default=0, help="with --run-dir: the run's seed")
    p.add_argument("--dataset-dir", default=DEF_DATASET,
                   help="Dataset root containing splits/ [default: $DATASET_DIR]")
    p.add_argument("--store-dir", default=DEF_STORE,
                   help="Where the θ* bench rows go [default: $THETA_STORE_DIR, else "
                        "${STORE_DIR}_theta — never STORE_DIR itself, which holds the "
                        "θ=0.5 shards]")
    p.add_argument("--arms", default="", help="space/comma list of loss tags (default: all found)")
    p.add_argument("--device", default=None, help="mps | cuda | cpu (default: auto)")
    p.add_argument("--sen2sr-dir", default=DEF_SEN2SR,
                   help="Override the checkpoint's baked-in SR weights dir. r0 arms are "
                        "bicubic and need none, but the hparams still record the TRAINING "
                        "node's path — set this if load_from_checkpoint goes looking for it. "
                        "[default: $SEN2SR_DIR]")
    p.add_argument("--select-on", default="iou_mean",
                   choices=["iou_mean", "f1_mean", "iou_micro", "f1_micro"])
    p.add_argument("--lo", type=float, default=0.05)
    p.add_argument("--hi", type=float, default=0.95)
    p.add_argument("--step", type=float, default=0.025,
                   help="θ grid resolution (default 0.025 -> 37 points). Every θ "
                        "is scored off the SAME forward pass, so a finer grid is "
                        "essentially free.")
    p.add_argument("--prefer-platform", default="L4_modal",
                   help="when one arm exists from several machines (same "
                        "model_name), bench this platform suffix and report the "
                        "rest as skipped duplicates")
    p.add_argument("--max-tiles", type=int, default=None, help="cap tiles everywhere (smoke test)")
    p.add_argument("--sweep-max-tiles", type=int, default=None,
                   help="cap tiles for the SWEEP only; the final bench still uses all")
    p.add_argument("--refresh-sweep", action="store_true", help="redo the sweep even if sweep.json exists")
    p.add_argument("--skip-bench", action="store_true", help="sweep only, no final eval")
    p.add_argument("--allow-duplicate", action="store_true",
                   help="bench an arm already present in the store (creates a SECOND shard)")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    runs_dir, store_dir = Path(args.runs_dir), Path(args.store_dir)
    if not Path(args.dataset_dir).is_dir():
        print(f"ERROR: --dataset-dir not found: {args.dataset_dir}", file=sys.stderr)
        return 2

    if args.run_dir:
        d = Path(args.run_dir)
        if not d.is_dir():
            print(f"ERROR: --run-dir not found: {d}", file=sys.stderr)
            return 2
        parsed = parse_run_dir(d.name) or {}
        ckpt = d / FINAL_CKPT
        specs = [{
            "run_dir": d,
            "checkpoint": ckpt if ckpt.is_file() else None,
            "model_name": args.model_name
                or (f"sr_{parsed['exp']}_{parsed['tag']}_holdout" if parsed else d.name),
            "exp_tag": args.exp_tag or parsed.get("exp", ""),
            "tag": parsed.get("tag", d.name),
            "seed": args.seed if args.model_name else parsed.get("seed", args.seed),
            "platform": parsed.get("platform", ""),
            "config_yaml": (d / "best_params.yaml")
                if (d / "best_params.yaml").is_file() else None,
        }]
    else:
        if not runs_dir.is_dir():
            print(f"ERROR: --runs-dir not found: {runs_dir}", file=sys.stderr)
            return 2
        specs = discover(runs_dir)
    wanted = {t.strip() for t in args.arms.replace(",", " ").split() if t.strip()}
    if wanted:
        missing = wanted - {s["tag"] for s in specs}
        if missing:
            print(f"WARNING: no run dir for requested arm(s): {', '.join(sorted(missing))}",
                  file=sys.stderr)
        specs = [s for s in specs if s["tag"] in wanted]

    # Only arms with a usable checkpoint can collide in the store.
    usable = [s for s in specs if s["checkpoint"] is not None]
    unusable = [s for s in specs if s["checkpoint"] is None]
    usable, dupes = resolve_duplicates(usable, args.prefer_platform)

    def _src(var: str) -> str:
        return f"  [${var}]" if os.environ.get(var) else ""

    print(f"runs-dir   : {runs_dir}{_src('RUNS_ROOT')}")
    print(f"dataset    : {args.dataset_dir}{_src('DATASET_DIR')}")
    if args.skip_bench:
        print("store      : (unused — --skip-bench writes only sweep.json)")
    else:
        print(f"store      : {store_dir}"
              f"{_src('THETA_STORE_DIR') or (_src('STORE_DIR') and '  [$STORE_DIR + _theta]')}")
    if args.sen2sr_dir:
        print(f"sen2sr     : {args.sen2sr_dir}{_src('SEN2SR_DIR')}")
    print(f"select θ*  : {args.select_on}")
    print(f"θ grid     : {args.lo}..{args.hi} step {args.step} "
          f"({len(theta_grid(args.step, args.lo, args.hi))} points, one pass)")
    print(f"discovered : {len(specs)} arm dir(s)\n")
    todo = []
    for s in sorted(usable + unusable, key=lambda x: (x["tag"], x["platform"])):
        sweep = s["run_dir"] / "sweep.json"
        note = []
        if s["checkpoint"] is None:
            note.append("NO FINAL CKPT — skip")
        if sweep.is_file() and not args.refresh_sweep:
            note.append(f"sweep.json exists (θ*={json.loads(sweep.read_text()).get('best_threshold')})")
        if s["config_yaml"] is None:
            note.append("no best_params.yaml (config_hash will be empty)")
        print(f"  {s['tag']:<12} seed={s['seed']} [{s['platform'] or '-':<12}] "
              f"{s['model_name']:<32} {'; '.join(note)}")
        if s in usable:
            todo.append(s)
    if dupes:
        print("\n  duplicate arm dirs (same model_name+seed, one shard allowed) — "
              f"kept --prefer-platform={args.prefer_platform}:")
        for d in dupes:
            print(f"    dropped {d}")

    if args.dry_run:
        print(f"\n--dry-run: {len(todo)} arm(s) would be processed. Nothing executed.")
        return 0
    if not todo:
        print("\nnothing to do.")
        return 0

    args.device = pick_device(args.device)
    print(f"\ndevice     : {args.device}")
    if args.device == "mps":
        print("  NB benchmarking.runner._sync() only synchronises CUDA, so the "
              "`inference_ms` column is meaningless on mps. Metrics are unaffected.")
    # Only touch the store if something is actually going to be written to it.
    # --skip-bench (how _stages_tv.sh drives the sweep) writes sweep.json into
    # the RUN dir and nothing else, so creating the store there would fail on a
    # default path that does not exist on this machine — for no reason at all.
    if not args.skip_bench:
        store_dir.mkdir(parents=True, exist_ok=True)

    benched, skipped, failed = [], [], []
    for i, spec in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {spec['model_name']} seed={spec['seed']}")
        try:
            _process(spec, args, store_dir, benched, skipped)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            # Unattended run: one bad arm must not cost you the other twelve.
            failed.append((spec["model_name"], f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"\nbenched {len(benched)} arm(s); skipped {len(skipped)}; failed {len(failed)}.")
    if skipped:
        print("  skipped: " + ", ".join(skipped))
    for name, err in failed:
        print(f"  FAILED  {name}: {err}")

    if not args.no_report and benched:
        cmd = [sys.executable, "-m", "benchmarking.cli", "report",
               "--store-dir", str(store_dir),
               "--metric", "f1", "--metric", "iou",
               "--metric", "apls", "--metric", "cldice"]
        print("\n" + " ".join(cmd) + "\n" + "=" * 70)
        env = dict(os.environ)
        repo_src = str(Path(__file__).resolve().parents[2] / "src")
        env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(cmd, env=env, check=False)
    return 1 if failed else 0


def _process(spec: dict, args, store_dir: Path, benched: list, skipped: list) -> None:
    """Sweep (or reuse) θ* for one arm, then bench it once at θ*."""
    sweep_path = spec["run_dir"] / "sweep.json"

    if sweep_path.is_file() and not args.refresh_sweep:
        theta = float(json.loads(sweep_path.read_text())["best_threshold"])
        print(f"  reusing sweep.json: θ* = {theta}")
    else:
        theta = float(sweep_arm(spec, args)["best_threshold"])

    if args.skip_bench:
        return
    if already_in_store(store_dir, spec["model_name"], spec["seed"]) and not args.allow_duplicate:
        print("  SKIP final bench — already in this store. The store is append-only "
              "with uuid run_ids, so a second shard would double-count. "
              "Use --allow-duplicate only if you know why.")
        skipped.append(spec["model_name"])
        return

    from benchmarking.runner import evaluate
    print(f"  final bench at θ*={theta} with {', '.join(TILE_METRICS)}")
    evaluate(
        dataset_dir=Path(args.dataset_dir),
        checkpoint=spec["checkpoint"],
        model_name=spec["model_name"],
        seed=spec["seed"],
        store_dir=store_dir,
        split=SPLIT,
        model=MODEL_FAMILY,
        mask_source=MASK_SOURCE,
        mask_dirname=MASK_DIRNAME,
        config_yaml_path=spec["config_yaml"],
        sen2sr_dir=Path(args.sen2sr_dir) if args.sen2sr_dir else None,
        exp_tag=spec["exp_tag"],
        label_source=LABEL_SOURCE,
        tile_metrics=TILE_METRICS,
        check="first",
        device=args.device,
        threshold=theta,
        max_tiles=args.max_tiles,
    )
    benched.append(spec["model_name"])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    raise SystemExit(main())
