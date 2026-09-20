"""Strip a training checkpoint to a serving artifact (plan §5 Phase 1).

A Lightning checkpoint carries the optimiser, the LR schedulers, the fit loop
state and the callback dicts -- none of which a forward pass needs, and which
here are two thirds of the file (298.8 MB -> 101.6 MB for r2a). The callbacks
dict is also the last place the training node's absolute `/scratch/...` paths
survive, so dropping it is a portability fix as much as a size one.

WHAT IS DELIBERATELY KEPT
------------------------
`hyper_parameters`, so `JointSRUNetLightning.load_from_checkpoint()` keeps
working UNCHANGED on the output -- the same call `sr/viz_tile.py:106` makes.
Rebuilding the architecture by hand from a bare state_dict would be a second
implementation of the model's construction, free to drift from the real one.

NOTHING IS CAST. SEN2SR runs in an fp32 island (its FFT hard constraint has no
half-precision kernels), and 46 of r2a's 486 tensors are int64 BatchNorm
`num_batches_tracked`. So the invariant asserted is "no fp16 present", not
"everything is fp32", which would be false.

theta* IS CROSS-CHECKED, NOT TRUSTED
------------------------------------
The operating point is read back from the run's own sweep.json and compared to
the one passed in. A stale theta is exactly the kind of error that ships a
plausible-looking wrong mask, so a disagreement is fatal here rather than
discovered in the demo.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

KEEP = ("state_dict", "hyper_parameters", "hparams_name",
        "pytorch-lightning_version", "epoch", "global_step")

DEFAULT_SRC = ("/Volumes/MAC_KIOXIA/Data/InstaRoad/SRruns/"
               "sr_r2a_new_gap_ce_anorm_recalpost_seed42/checkpoints/"
               "unet_s2rosa_jointsr_final.ckpt")
REL_SEN2SR = {"sen2sr": "models/SEN2SRLite_RGBN", "sr4rs": "models/SR4RS_RGBN"}


def mb(n: int) -> float:
    return n / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out-dir", default="demo/weights")
    ap.add_argument("--arm", default="r2a")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="theta*, cross-checked against the run's sweep.json")
    ap.add_argument("--window-px", type=int, default=128,
                    help="128 for a pinned SEN2SR arm; 256 for a convolutional one")
    ap.add_argument("--allow-theta-mismatch", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    run_dir = src.parent.parent
    out = Path(args.out_dir) / args.arm
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(src, map_location="cpu", weights_only=False)
    dropped = sorted(set(ck) - set(KEEP))
    sd = ck["state_dict"]
    hp = dict(ck["hyper_parameters"])

    bad = [k for k, v in sd.items() if getattr(v, "dtype", None) == torch.float16]
    assert not bad, f"fp16 tensors present, refusing to ship: {bad[:3]}"

    orig_dir = hp.get("sen2sr_dir")
    rel = REL_SEN2SR.get(str(hp.get("upsampler")))
    if rel:
        hp["sen2sr_dir"] = rel

    # theta* cross-check against the run's own selection sweep.
    sweep_p = run_dir / "sweep.json"
    sweep_meta, sweep_theta = {}, None
    if sweep_p.exists():
        s = json.loads(sweep_p.read_text())
        sweep_theta = float(s["best_threshold"])
        sweep_meta = {k: s.get(k) for k in ("split", "criterion", "purpose")}
        if abs(sweep_theta - args.threshold) > 1e-9 and not args.allow_theta_mismatch:
            raise SystemExit(
                f"theta mismatch: --threshold {args.threshold} but {sweep_p} says "
                f"{sweep_theta}. Pass --allow-theta-mismatch only if you mean it.")

    torch.save({**{k: ck[k] for k in KEEP if k in ck}, "hyper_parameters": hp},
               out / "model.ckpt")

    n_bytes = sum(v.numel() * v.element_size() for v in sd.values()
                  if hasattr(v, "numel"))
    cfg = {
        "arm": args.arm,
        "source_run": run_dir.name,
        "source_ckpt": src.name,
        "epoch": ck.get("epoch"), "global_step": ck.get("global_step"),
        "upsampler": hp.get("upsampler"), "sr_hc": hp.get("sr_hc"),
        "reflectance_scale": hp.get("reflectance_scale"),
        "adaptive_norm": hp.get("adaptive_norm"),
        "encoder_name": hp.get("encoder_name"),
        "threshold": args.threshold,
        "threshold_source": (f"sweep.json (split={sweep_meta.get('split')}, "
                             f"criterion={sweep_meta.get('criterion')}, "
                             f"purpose={sweep_meta.get('purpose')})"
                             if sweep_theta is not None else "supplied, no sweep.json"),
        "window_px": args.window_px,
        "window_px_reason": ("SEN2SR HardConstraint pins the LR input; "
                             "model._required_lr" if args.window_px == 128
                             else "bench footprint cell; generator is convolutional"),
        "bands": [1, 2, 3, 4], "band_names": ["B4", "B3", "B2", "B8"],
        "sen2sr_dir": hp.get("sen2sr_dir"), "sen2sr_dir_original": orig_dir,
        "state_dict_mb": round(mb(n_bytes), 3), "n_tensors": len(sd),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    del ck, sd
    gc.collect()

    a, b = src.stat().st_size, (out / "model.ckpt").stat().st_size
    print(f"{args.arm}: {mb(a):.1f} MB -> {mb(b):.1f} MB  ({100*b/a:.1f}%)")
    print(f"  dropped   : {', '.join(dropped)}")
    print(f"  tensors   : {cfg['n_tensors']} ({cfg['state_dict_mb']} MB payload)")
    print(f"  epoch     : {cfg['epoch']}   theta* {args.threshold} "
          f"({'sweep.json agrees' if sweep_theta is not None else 'no sweep.json'})")
    print(f"  sen2sr_dir: {orig_dir}\n           -> {cfg['sen2sr_dir']}")
    print(f"\nVerify with:  uv run --no-sync python demo/space/parity.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
