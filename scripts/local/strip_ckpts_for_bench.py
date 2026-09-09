#!/usr/bin/env python
"""Strip a staged bench folder down to what INFERENCE actually needs.

A Lightning checkpoint is mostly optimizer state -- 273 MB of a 410 MB r4a
file, 66% of it -- and benching never touches it. Dropping it takes the R
series from ~9.2 GB to ~3.2 GB, which is the difference between fitting in a
20 GB cluster quota and not.

WHAT IS KEPT: state_dict (the weights), hyper_parameters + hparams_name (the
model is RECONSTRUCTED from these by load_from_checkpoint, so they are not
optional), pytorch-lightning_version, and epoch/global_step for provenance.

WHAT IS DROPPED: optimizer_states, lr_schedulers, callbacks, loops. All of it
is resume-only. A stripped checkpoint CANNOT be trained onward -- keep the
originals if any of these runs might still be resumed.

    python scripts/local/strip_ckpts_for_bench.py --src <staged> --dst <upload>
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import torch

# The keep-list lives in ckpt_slim so the staging path and the in-place prune
# the engines run can never disagree about what inference needs.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ckpt_slim import KEEP, strip_checkpoint  # noqa: E402


def digest(sd) -> str:
    """Order-independent hash of a state_dict's tensors, to prove nothing moved."""
    h = hashlib.md5()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes() if hasattr(v, "detach") else repr(v).encode())
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--verify", action="store_true",
                   help="re-read each stripped file and check the weights hash matches")
    a = p.parse_args(argv)
    a.dst.mkdir(parents=True, exist_ok=True)

    tot_in = tot_out = 0
    for src in sorted(a.src.glob("*.ckpt")):
        before = digest(torch.load(src, map_location="cpu",
                                   weights_only=False)["state_dict"])
        dst = a.dst / src.name
        si, so = strip_checkpoint(src, dst)
        tot_in += si; tot_out += so
        note = ""
        if a.verify:
            re = torch.load(dst, map_location="cpu", weights_only=False)
            assert digest(re["state_dict"]) == before, f"{src.name}: weights changed!"
            assert "hyper_parameters" in re, f"{src.name}: lost hyper_parameters"
            note = "  verified"
        print(f"  {src.name:<62} {si/2**20:6.0f} -> {so/2**20:6.0f} MB{note}")

    for pat in ("*.sweep.json", "*.best_params.yaml", "manifest.json"):
        for f in a.src.glob(pat):
            shutil.copy2(f, a.dst / f.name)
    print(f"\n{tot_in/2**30:.2f} GB -> {tot_out/2**30:.2f} GB "
          f"({100*(1-tot_out/tot_in):.0f}% smaller)  -> {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
