#!/usr/bin/env python
"""Phase B matrix: 3 attention parents x 3 region terms, with θ*/λ* inherited.

The tedious part you hit is that a compound's pixel slot must reuse the PARENT
arm's tuned `pos_weight`, `tl_theta` and `gap_theta` — otherwise the compound
re-searches them and you are no longer measuring "what does adding a region
term do to THIS attention arm", you are measuring a fresh joint search. This
reads each parent's `best_params.yaml` and emits the pinning env for you.

What stays searched in the child: **lr and mix_w only** (encoder and batch are
protocol constants). That is exactly the "retune the learning rate and the
combination weights" you asked for.

    python scripts/local/phase_b_matrix.py                 # print the 9 commands
    python scripts/local/phase_b_matrix.py --stage fit     # the fit commands
    python scripts/local/phase_b_matrix.py --run           # actually run them

HOW THE PINNING WORKS (no engine changes needed)

  PSTAR=<parent>            picks the pixel slot inside the pstar_* compound
  SEARCH_THETAS=false       stops sr.tune searching tl_theta / gap_theta ...
  TL_THETA= GAP_THETA=      ... and supplies the parent's values instead, which
                            flow through loss_hp into both the trials AND the
                            child's own best_params.yaml
  POS_WEIGHT_MIN=MAX=λ*     a degenerate search range pins λ (sr.tune always
                            searches pos_weight for λ-consuming pixel slots,
                            and gap_*/tl_* are all λ-consuming)

WHY EXP_TAG IS OVERRIDDEN — this one is a real trap

  `_stages_tv.sh` builds LOSS_TAG from LOSS_ARM alone. LOSS_ARM is
  `pstar_dice` whatever PSTAR happens to be, so all three parents would share
  ONE run dir, ONE Optuna study and ONE bench `model_name`
  (`sr_r0_new_pstar_dice_holdout_seed0`). The second parent would silently
  resume the first's study and add a duplicate shard to the store. EXP_TAG is
  therefore given a per-parent suffix; it feeds only RUN_DIR / STUDY_NAME /
  MODEL_NAME / the store's `exp_tag` column, nothing functional.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

# Attention parents -> short tag used to disambiguate the run dir.
PARENTS = {
    "gap_tl_ce": "gaptl",
    "gap_t2_ce": "gapt2",
    "gap_t4_ce": "gapt4",
}

# Region slot -> the arm script that builds it.
REGIONS = {
    "pstar_dice": "loss/l5_new.sh",
    "pstar_sdice": "loss/l13_new.sh",
    "pstar_lcdice": "loss/l14_new.sh",
}

# Hyperparameters carried over from the parent's tune, verbatim.
INHERIT = ("pos_weight", "tl_theta", "gap_theta")


REPO = Path(__file__).resolve().parents[2]
FINAL_CKPT = Path("checkpoints") / "unet_s2rosa_jointsr_final.ckpt"


def default_runs_root() -> Path:
    """Same rule as scripts/LightningStudio/env.sh.

    env.sh is a *shell* file sourced by run.sh, so INSTAROAD_ROOT is not in this
    process's environment unless you exported it yourself. Falling back to the
    repo's parent matches what env.sh would have computed, instead of silently
    resolving ./runs and reporting every parent as missing.
    """
    env = os.environ.get("INSTAROAD_ROOT")
    return (Path(env) if env else REPO.parent) / "runs"


def parent_overlay(runs_root: Path, parent: str, exp: str, seed: int) -> Path:
    return runs_root / f"sr_{exp}_{parent}_holdout_seed{seed}" / "best_params.yaml"


def trials_done(study_db: Path, study_name: str) -> int:
    """COMPLETE+PRUNED count, so an interrupted tune tops up to the target
    instead of adding a second full budget on top of it (the same rule
    pilot_seq.sh uses)."""
    try:
        import optuna
        s = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_db}")
        return sum(t.state.name in ("COMPLETE", "PRUNED") for t in s.trials)
    except Exception:
        return 0


def read_parent(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    model = cfg.get("model", {})
    missing = [k for k in INHERIT if k not in model]
    if missing:
        raise SystemExit(
            f"{path}: missing {missing}.\n"
            "  pos_weight is written only when the arm consumed λ, and the θs only\n"
            "  when the tune searched them. If a θ is absent the parent was tuned\n"
            "  with --search-thetas false, and its fixed value is the engine default\n"
            "  (tl_theta 0.375 / gap_theta 0.5) — pass --allow-defaults to use those."
        )
    return {k: model[k] for k in INHERIT}


def read_parent_lenient(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    model = cfg.get("model", {})
    return {
        "pos_weight": model.get("pos_weight"),
        "tl_theta": model.get("tl_theta", 0.375),
        "gap_theta": model.get("gap_theta", 0.5),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=None,
                    help="default: $INSTAROAD_ROOT/runs")
    ap.add_argument("--exp", default="r0_new", help="parent EXP_TAG")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="all",
                    choices=["all", "tune", "fit", "bench"],
                    help="'all' chains tune -> fit -> bench per arm and is "
                         "idempotent: finished stages are skipped and a partial "
                         "Optuna study tops up to --trials. Re-run after any "
                         "interruption.")
    ap.add_argument("--parents", default=" ".join(PARENTS))
    ap.add_argument("--regions", default=" ".join(REGIONS))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--mix-w-min", type=float, default=0.25)
    ap.add_argument("--mix-w-max", type=float, default=0.75)
    ap.add_argument("--allow-defaults", action="store_true",
                    help="use engine θ defaults when a parent overlay lacks them")
    ap.add_argument("--no-bench", action="store_true",
                    help="chain tune -> fit only, leaving the arm unscored. The θ "
                         "sweep lives inside STAGE=bench, so this skips sweeping "
                         "too — use it to train a batch of arms now and sweep + "
                         "bench them all together later, which keeps every arm's "
                         "θ* selected under one identical protocol run.")
    ap.add_argument("--run", action="store_true", help="execute instead of printing")
    args = ap.parse_args(argv)

    root = Path(args.runs_root) if args.runs_root else default_runs_root()

    parents = [p for p in args.parents.replace(",", " ").split() if p]
    regions = [r for r in args.regions.replace(",", " ").split() if r]
    for p in parents:
        if p not in PARENTS:
            raise SystemExit(f"unknown parent {p!r} (known: {list(PARENTS)})")
    for r in regions:
        if r not in REGIONS:
            raise SystemExit(f"unknown region {r!r} (known: {list(REGIONS)})")

    print(f"# runs root : {root}")
    print(f"# stage     : {args.stage}"
          + ("  (tune -> fit only; NOT swept, NOT benched)" if args.no_bench else ""))
    print(f"# inherited : {', '.join(INHERIT)}  (searched in the child: lr, mix_w)")
    print(f"# mix_w     : [{args.mix_w_min}, {args.mix_w_max}]\n")

    cmds: list[dict] = []
    skipped: list[str] = []
    for parent in parents:
        ov = parent_overlay(root, parent, args.exp, args.seed)
        if not ov.is_file():
            raise SystemExit(f"parent overlay not found: {ov}\n"
                             f"  (has {parent} finished STAGE=tune?)")
        hp = read_parent_lenient(ov) if args.allow_defaults else read_parent(ov)
        if hp["pos_weight"] is None:
            raise SystemExit(f"{ov}: no pos_weight — {parent} is not a λ-consuming arm?")
        print(f"# {parent}: λ*={hp['pos_weight']:.6g}  "
              f"tl_theta={hp['tl_theta']:.6g}  gap_theta={hp['gap_theta']:.6g}")

        for region in regions:
            exp_tag = f"{args.exp}_{PARENTS[parent]}"
            run_name = f"sr_{exp_tag}_{region}_holdout_seed{args.seed}"
            run_dir = root / run_name
            pinned = [
                f"EXP_TAG={exp_tag}",
                f"PSTAR={parent}",
                "SEARCH_THETAS=false",
                f"TL_THETA={hp['tl_theta']}",
                f"GAP_THETA={hp['gap_theta']}",
                f"POS_WEIGHT_MIN={hp['pos_weight']}",
                f"POS_WEIGHT_MAX={hp['pos_weight']}",
                f"SEED={args.seed}",
            ]

            chain = ("tune", "fit") if args.no_bench else ("tune", "fit", "bench")
            for stage in (chain if args.stage == "all" else (args.stage,)):
                # Idempotency, same markers pilot_seq.sh skips on.
                if stage == "tune":
                    if (run_dir / "best_params.yaml").is_file():
                        skipped.append(f"{run_name} tune (best_params.yaml exists)")
                        continue
                    done = trials_done(run_dir / "study.db", run_name)
                    rem = max(args.trials - done, 0)
                    if rem == 0 and done:
                        skipped.append(f"{run_name} tune ({done} trials already)")
                        continue
                    extra = [f"N_TRIALS={rem}",
                             f"MIX_W_MIN={args.mix_w_min}",
                             f"MIX_W_MAX={args.mix_w_max}"]
                    note = f"{done}/{args.trials} done, running {rem} more"
                elif stage == "fit":
                    if (run_dir / FINAL_CKPT).is_file():
                        skipped.append(f"{run_name} fit (final ckpt exists)")
                        continue
                    extra, note = ["RESUME_FIT=1"], "resume if possible"
                else:
                    if (run_dir / ".bench_done").is_file():
                        skipped.append(f"{run_name} bench (done)")
                        continue
                    extra, note = [], "val, apls+cldice"

                cmds.append({
                    "label": f"{parent} x {region} [{stage}]",
                    "run_dir": run_dir,
                    "stage": stage,
                    "note": note,
                    "cmd": ["bash", "scripts/LightningStudio/run.sh", REGIONS[region],
                            f"STAGE={stage}", *pinned, *extra],
                })

    print()
    for c in cmds:
        print(f"# {c['label']}  ({c['note']})  ->  {c['run_dir'].name}")
        print("  " + " ".join(c["cmd"]) + "\n")
    for s in skipped:
        print(f"# SKIP {s}")
    if skipped:
        print()

    if not args.run:
        print(f"# {len(cmds)} command(s). Re-run with --run to execute them in order.")
        return 0

    for i, c in enumerate(cmds, 1):
        print(f"\n===== [{i}/{len(cmds)}] {c['label']} =====", flush=True)
        rc = subprocess.run(c["cmd"], cwd=REPO).returncode
        if rc != 0:
            print(f"FAILED ({rc}) on {c['label']} — stopping. Re-run to resume; "
                  "finished stages are skipped.", file=sys.stderr)
            return rc
        if c["stage"] == "bench":
            # Same marker the pilot orchestrators use to skip a finished bench.
            (c["run_dir"] / ".bench_done").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
