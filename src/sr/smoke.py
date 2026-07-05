"""Smoke + single-batch overfit tests for the joint SR+seg pipeline (R2).

Runs on synthetic data (no dataset needed — only the SEN2SR weights dir), so it
works on a laptop before touching the HPC. Checks, in order:

  1. forward: output logits are exactly 4x the input spatial size;
  2. backward: BOTH the SEN2SR and U-Net param groups receive non-zero
     gradients (and, under freeze_sr, SEN2SR receives none) — printing the
     per-group grad norms so the differential-LR/alpha behaviour is visible;
  3. overfit: a fixed batch, optimised with the module's own two-group
     optimiser, drives RoadSegLoss down — sanity that end-to-end joint
     learning works through SEN2SR -> adapter -> U-Net.

NOTE the loss used for the backward probe must be spatially varying: SEN2SR's
hard constraint takes the DC Fourier bin from the LR input, so a plain
`output.mean()` probe would report exactly-zero SR gradients on a perfectly
healthy network (see sr.sen2sr_loader docstring). RoadSegLoss qualifies.

Run:
    python -m sr.smoke --sen2sr-dir /path/to/SEN2SRLite_RGBN [--download]
"""
from __future__ import annotations

import argparse
import math
import time

import torch

from sr.module import JointSRSegModule

# Frozen per-band stats of the curated dataset (Data.npz, M0 slice) — synthetic
# reflectance below is drawn to be plausible under these. Overridable if the
# dataset stats are regenerated.
DEFAULT_BAND_MEAN = (1034.26, 866.66, 622.78, 2244.93)
DEFAULT_BAND_STD = (531.00, 375.97, 321.51, 639.03)


def make_synthetic_batch(batch: int = 2, patch: int = 128, scale: int = 4,
                         seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A reflectance-like LR image and a correlated HR road mask.

    Roads are diagonal bands drawn at HR; the LR image is the mask's
    area-downsampled brightness bump over smooth background noise, so there is
    real (sub-pixel) image evidence for the mask — the situation R2 targets.
    """
    g = torch.Generator().manual_seed(seed)
    hr = patch * scale
    yy, xx = torch.meshgrid(torch.arange(hr), torch.arange(hr), indexing="ij")
    mask = torch.zeros(batch, 1, hr, hr)
    for b in range(batch):
        for k in range(3):
            offset = int(torch.randint(-hr // 2, hr // 2, (1,), generator=g))
            width = int(torch.randint(3, 8, (1,), generator=g))  # ~road width @2.5m
            band = ((xx - yy + offset).abs() < width) if k % 2 == 0 else \
                   ((xx + yy - hr - offset).abs() < width)
            mask[b, 0][band] = 1.0

    base = 0.08 + 0.15 * torch.rand(batch, 4, 1, 1, generator=g)
    noise = 0.02 * torch.randn(batch, 4, patch, patch, generator=g)
    lr_mask = torch.nn.functional.avg_pool2d(mask, scale)  # fraction of road per 10m pixel
    x = (base + noise + 0.10 * lr_mask).clamp(0.0, 1.0)
    return x, mask


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # MPS coverage for SEN2SR's FFT ops varies by torch version — probe it.
        try:
            torch.fft.fftn(torch.randn(1, 4, 8, 8, device="mps"), dim=(-2, -1))
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


def build_module(sen2sr_dir: str, freeze_sr: bool = False,
                 encoder_weights: str | None = "imagenet",
                 lr_sr: float = 1e-4, lr_seg: float = 1e-3,
                 pos_weight: float = 10.0) -> JointSRSegModule:
    return JointSRSegModule(
        upsampler="sen2sr", sen2sr_dir=sen2sr_dir,
        band_mean=DEFAULT_BAND_MEAN, band_std=DEFAULT_BAND_STD,
        encoder="resnet34", encoder_weights=encoder_weights,
        pos_weight=pos_weight, lr_sr=lr_sr, lr_seg=lr_seg, freeze_sr=freeze_sr,
    )


def run_smoke(module: JointSRSegModule, x: torch.Tensor, y: torch.Tensor) -> dict:
    """One forward+backward; assert shapes and per-group gradient flow."""
    module.train()
    module.zero_grad()
    logits = module(x)

    scale = module.scale
    expected = (x.shape[0], 1, x.shape[-2] * scale, x.shape[-1] * scale)
    assert tuple(logits.shape) == expected, \
        f"output {tuple(logits.shape)} != {expected} (input x{scale})"

    loss = module.criterion(logits, y)
    loss.backward()
    norms = module.grad_norms()
    alpha = module.hparams.lr_sr / module.hparams.lr_seg
    print(f"  loss={loss.item():.4f}  grad norms: "
          + "  ".join(f"{k}={v:.4e}" for k, v in norms.items())
          + f"  (lr_sr={module.hparams.lr_sr:g}, lr_seg={module.hparams.lr_seg:g}, "
            f"alpha={alpha:g})")

    assert norms["seg"] > 0, "U-Net received no gradient"
    if module.hparams.freeze_sr:
        assert norms["sr"] == 0, "frozen SEN2SR unexpectedly received gradient"
    else:
        assert norms["sr"] > 0, (
            "SEN2SR received no gradient — trainable-module wiring is broken "
            "(compiled/eval-mode SEN2SR, or a DC-only loss probe?)"
        )
    module.zero_grad()
    return norms


def run_overfit(module: JointSRSegModule, x: torch.Tensor, y: torch.Tensor,
                steps: int = 60, print_every: int = 5) -> tuple[float, float]:
    """Optimise the module's own two-group optimiser on one fixed batch."""
    module.train()
    opt = module.configure_optimizers()
    sr_before = torch.cat([p.detach().flatten().cpu() for p in
                           module.upsampler.parameters() if p.requires_grad])

    first = last = None
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = module.criterion(module(x), y)
        loss.backward()
        opt.step()
        v = loss.item()
        first = first if first is not None else v
        last = v
        if step == 1 or step % print_every == 0 or step == steps:
            print(f"  step {step:>4}/{steps}  loss {v:.4f}")

    sr_after = torch.cat([p.detach().flatten().cpu() for p in
                          module.upsampler.parameters() if p.requires_grad])
    print(f"  SEN2SR weight movement |Δw| = {(sr_after - sr_before).norm():.4e} "
          f"(> 0 shows the low-LR group is updating)")
    assert math.isfinite(last), "loss diverged to non-finite"
    assert last < 0.5 * first, \
        f"single-batch overfit failed to halve the loss ({first:.4f} -> {last:.4f})"
    assert (sr_after - sr_before).norm() > 0, "SEN2SR weights did not move"
    return first, last


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sen2sr-dir", required=True)
    ap.add_argument("--download", action="store_true", help="fetch weights if missing")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, default=128, help="pinned to 128 by the shipped FFT mask")
    ap.add_argument("--overfit-steps", type=int, default=60)
    ap.add_argument("--skip-overfit", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--encoder-weights", default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.download:
        from sr.sen2sr_loader import download_sen2sr
        download_sen2sr(args.sen2sr_dir)

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    enc_w = None if args.encoder_weights == "none" else args.encoder_weights
    x, y = make_synthetic_batch(args.batch, args.patch, seed=args.seed)
    print(f"device={device}  batch={tuple(x.shape)} -> mask={tuple(y.shape)}  "
          f"road frac={y.mean():.4f}")
    pos_weight = float(((1 - y).sum() / y.sum().clamp(min=1)).clamp(max=50))

    print("\n[1/3] smoke: R2 (joint, trainable SEN2SR)")
    t0 = time.time()
    module = build_module(args.sen2sr_dir, freeze_sr=False, encoder_weights=enc_w,
                          pos_weight=pos_weight).to(device)
    run_smoke(module, x.to(device), y.to(device))
    print(f"  ok ({time.time() - t0:.1f}s)")

    print("\n[2/3] smoke: R1 (frozen SEN2SR) — sr grads must be zero")
    r1 = build_module(args.sen2sr_dir, freeze_sr=True, encoder_weights=enc_w,
                      pos_weight=pos_weight).to(device)
    run_smoke(r1, x.to(device), y.to(device))
    print("  ok")

    if args.skip_overfit:
        print("\n[3/3] overfit: skipped")
        return
    print(f"\n[3/3] overfit: single batch, {args.overfit_steps} steps "
          f"(lr_seg=1e-3, lr_sr=1e-4)")
    first, last = run_overfit(module, x.to(device), y.to(device),
                              steps=args.overfit_steps)
    print(f"  ok: loss {first:.4f} -> {last:.4f}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
