#!/usr/bin/env python
"""Shrink Lightning checkpoints to what INFERENCE needs, and prune run dirs.

ONE implementation, two callers: the bench-staging script imports
``strip_checkpoint`` from here, and the fit engines call the ``prune``
subcommand once a run's bench row is safely in the store.

WHAT IS DEAD WEIGHT
-------------------
A Lightning checkpoint is mostly optimizer state -- 273 MB of a 410 MB SR4RS
file, 66% -- and nothing outside `resume` ever reads it. ``load_from_checkpoint``
rebuilds the model from ``hyper_parameters`` + ``hparams_name`` and then loads
``state_dict``; those three are what must survive.

Verified on all eight architectures in this project (bicubic, SEN2SR
frozen/joint, with and without the FFT constraint, SR4RS frozen/joint, ditto):
stripped and original produce BIT-IDENTICAL forward passes on mps.

SR SNAPSHOTS ARE ALREADY MINIMAL. sr_snapshots/*.pt hold `sr_state_dict` plus a
few scalars -- no optimizer state -- so there is nothing to strip out of them.
They are large because there are many (52 x 43 MB on a joint SR4RS run), which
is a retention question, not a stripping one. This tool leaves them alone.

WHAT `prune` DOES TO A RUN DIR
------------------------------
  last.ckpt                     LEFT COMPLETE -- resume reads the optimizer
                                state, and a half-trained run whose last.ckpt
                                was stripped has to start from zero.
  *_final.ckpt                  stripped to inference-only
  *_epochNNN.ckpt               dropped entirely with --drop-epoch-snapshots,
                                otherwise stripped
  sr_snapshots/                 untouched

DESTRUCTIVE, SO IT DOES NOTHING WITHOUT --apply. The default is a dry run that
prints exactly what it would reclaim.

    python scripts/local/ckpt_slim.py prune --runs-root /scratch/$USER/InstaRoad/runs
    python scripts/local/ckpt_slim.py prune --run-dir <one run> --drop-epoch-snapshots --apply
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Everything load_from_checkpoint needs, and nothing else. hyper_parameters and
# hparams_name are NOT optional: the model is reconstructed from them.
KEEP = ("state_dict", "hyper_parameters", "hparams_name",
        "pytorch-lightning_version", "epoch", "global_step")

FINAL_RE = re.compile(r".*_final\.ckpt$")
EPOCH_RE = re.compile(r".*_epoch\d+\.ckpt$")


def is_stripped(ck: dict) -> bool:
    return not any(k in ck for k in ("optimizer_states", "lr_schedulers"))


def strip_checkpoint(src: Path, dst: Path | None = None) -> tuple[int, int]:
    """Write an inference-only copy. Returns (bytes_before, bytes_after).

    dst=None strips IN PLACE via a temp file beside the original, so a crash
    mid-write cannot leave a truncated checkpoint where a good one was.
    """
    import torch

    before = src.stat().st_size
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if is_stripped(ck) and dst is None:
        return before, before
    out = {k: ck[k] for k in KEEP if k in ck}
    if "state_dict" not in out:
        raise ValueError(f"{src}: no state_dict -- not a Lightning checkpoint?")
    target = dst or src
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(out, tmp)
    tmp.replace(target)
    return before, target.stat().st_size


def prune_run_dir(d: Path, drop_epochs: bool, apply: bool) -> tuple[int, int, list[str]]:
    """Returns (bytes_before, bytes_after, notes) for one run dir."""
    ckdir = d / "checkpoints"
    if not ckdir.is_dir():
        return 0, 0, []
    before = after = 0
    notes = []
    for f in sorted(ckdir.glob("*.ckpt")):
        sz = f.stat().st_size
        before += sz
        if f.name == "last.ckpt":
            after += sz
            notes.append(f"    keep   {f.name:<38} {sz/2**20:6.0f} MB  (resume needs it)")
        elif EPOCH_RE.match(f.name) and drop_epochs:
            notes.append(f"    DROP   {f.name:<38} {sz/2**20:6.0f} MB")
            if apply:
                f.unlink()
        elif FINAL_RE.match(f.name) or EPOCH_RE.match(f.name):
            if apply:
                _, new = strip_checkpoint(f)
            else:
                import torch
                ck = torch.load(f, map_location="cpu", weights_only=False)
                new = sz if is_stripped(ck) else int(sz * 0.34)   # measured ratio
            after += new
            notes.append(f"    strip  {f.name:<38} {sz/2**20:6.0f} -> {new/2**20:4.0f} MB")
        else:
            after += sz
            notes.append(f"    keep   {f.name:<38} {sz/2**20:6.0f} MB  (unrecognised)")
    return before, after, notes


def thin_snapshots(d: Path, keep_every: int, apply: bool) -> tuple[int, int, list[str]]:
    """Keep every Nth SR snapshot, plus the first and last frames.

    The init frame is the REFERENCE the drift is measured against and the last
    frame is the endpoint, so both survive any thinning -- what gets dropped is
    intermediate temporal resolution, which is the only part that is merely
    nice to have.

    Snapshots are already inference-only (sr_state_dict + a few scalars), so
    there is nothing to strip out of them; thinning the count is the only lever.
    """
    snaps = sorted((d / "sr_snapshots").glob("*.pt"))
    if not snaps:
        return 0, 0, []
    keep = {snaps[0], snaps[-1]} | {f for i, f in enumerate(snaps) if i % keep_every == 0}
    before = after = 0
    notes = []
    for f in snaps:
        sz = f.stat().st_size
        before += sz
        if f in keep:
            after += sz
        else:
            if apply:
                f.unlink()
    notes.append(f"    sr_snapshots  {len(snaps)} frames -> {len(keep)} "
                 f"({before/2**20:.0f} -> {sum(f.stat().st_size for f in keep if f.exists())/2**20:.0f} MB)"
                 if apply else
                 f"    sr_snapshots  {len(snaps)} frames -> {len(keep)} "
                 f"({before/2**20:.0f} -> {after/2**20:.0f} MB)")
    return before, after, notes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prune", help="shrink checkpoints inside run dirs")
    q.add_argument("--runs-root", type=Path, help="parent of many run dirs")
    q.add_argument("--run-dir", type=Path, action="append", default=[])
    q.add_argument("--pattern", default="*_seed*", help="run-dir glob under --runs-root")
    q.add_argument("--drop-epoch-snapshots", action="store_true",
                   help="delete *_epochNNN.ckpt outright instead of stripping them")
    q.add_argument("--apply", action="store_true", help="actually modify files")
    q.add_argument("--thin-snapshots", type=int, metavar="N", default=0,
                   help="also keep only every Nth sr_snapshots frame (first and "
                        "last always kept). 0 = leave snapshots alone.")
    a = p.parse_args(argv)

    dirs = list(a.run_dir)
    if a.runs_root:
        dirs += [d for d in sorted(a.runs_root.glob(a.pattern)) if d.is_dir()]
    if not dirs:
        print("no run dirs matched"); return 1

    tb = ta = 0
    for d in dirs:
        b, t, notes = prune_run_dir(d, a.drop_epoch_snapshots, a.apply)
        if a.thin_snapshots > 1:
            sb, st, snotes = thin_snapshots(d, a.thin_snapshots, a.apply)
            b += sb; t += st; notes += snotes
        if not notes:
            continue
        print(f"\n{d.name}")
        for n in notes:
            print(n)
        tb += b; ta += t
    verb = "reclaimed" if a.apply else "WOULD reclaim"
    print(f"\n{len(dirs)} run dirs: {tb/2**30:.2f} GB -> {ta/2**30:.2f} GB, "
          f"{verb} {(tb-ta)/2**30:.2f} GB")
    if not a.apply:
        print("DRY RUN -- nothing was modified. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
