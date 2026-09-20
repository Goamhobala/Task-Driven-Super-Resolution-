"""Serving-side inference for the demo Space (plan §5 Phase 3).

THE WINDOWING IS NOT A DETAIL -- IT IS THE RESULT
-------------------------------------------------
`sr.viz_tile.run_tile` and `benchmarking.runner._score_tile_sr` both replay a
checkpoint in fixed windows, never in one whole-tile pass, for two reasons that
apply equally here:

  * r2a's SEN2SR FFT HardConstraint PINS the LR input to `model._required_lr`
    (128 px). Any other size is a shape error, not a slower path.
  * Even for an unpinned generator, convolution borders differ between a 256 px
    cell and a 512 px tile, so a whole-tile pass produces a prediction that
    merely RESEMBLES the benchmarked one.

So this module mirrors `run_tile` step for step, and `parity.py` asserts the
two agree exactly rather than trusting that they do.

Values stay in RAW reflectance all the way to `model(t)`: the checkpoint's
normalisation is applied INSIDE the forward, after super-resolution. Dividing
by `reflectance_scale` here would double-apply it -- the SR panel divides
because `model.sr` is entered directly, the prediction does not.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO / "src",):                      # vendored flat in the Space
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

TILE_PX = 512
DATASET_REPO = "Goamhobala/instaroad-rosa-tiles"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # The U-Net decoder's x2 nearest upsample wedges MPS
        # (upsample_nearest2d -> waitAndReadIntTensorData). runner.py's shim
        # reroutes it through repeat_interleave, which is bit-identical.
        from benchmarking.runner import _patch_mps_nearest_interpolate
        _patch_mps_nearest_interpolate()
        return torch.device("mps")
    return torch.device("cpu")


class Arm:
    """One checkpoint, loaded once, replayed in its own window size."""

    def __init__(self, ckpt: Path, config: dict, device: torch.device):
        from sr.model import JointSRUNetLightning
        self.cfg = config
        self.device = device
        model = JointSRUNetLightning.load_from_checkpoint(str(ckpt), map_location="cpu")
        # ZeroGPU wants the model on cuda at MODULE level (CUDA is emulated
        # outside @spaces.GPU); .to() here is that placement, not a lazy load.
        self.model = model.eval().float().to(device)
        if getattr(self.model.hparams, "reflectance_scale", None) is None:
            self.model.hparams.reflectance_scale = 1.0
        self.rs = float(self.model.hparams.reflectance_scale)
        self.pad = int(self.model.hparams.sr_pad)
        self.up = int(self.model.hparams.upscale)
        req = self.model._required_lr
        self.step = int(req) if req else int(config["window_px"])
        self.theta = float(config["threshold"])

    @torch.no_grad()
    def predict(self, img_chw: np.ndarray):
        """(4,H,W) raw reflectance -> (sr (4,H*up,W*up), prob (H*up,W*up))."""
        C, H, W = img_chw.shape
        if H % self.step or W % self.step:
            raise ValueError(f"{H}x{W} is not a multiple of the window "
                             f"({self.step} px) for {self.cfg['arm']}")
        up, pad = self.up, self.pad
        sr_out = np.empty((C, H * up, W * up), dtype=np.float32)
        prob = np.empty((H * up, W * up), dtype=np.float32)

        for r in range(0, H, self.step):
            for c in range(0, W, self.step):
                chunk = img_chw[:, r:r + self.step, c:c + self.step]
                t = torch.from_numpy(np.ascontiguousarray(chunk))[None].float().to(self.device)

                t_ref = t / self.rs
                t_sr = (torch.nn.functional.pad(t_ref, (pad,) * 4, mode="reflect")
                        if pad else t_ref)
                hr = self.model.sr(t_sr)
                if pad:
                    q = pad * up
                    hr = hr[..., q:-q, q:-q]

                logits = self.model(t)          # RAW values: normalisation is internal

                R, Cc = r * up, c * up
                s = self.step * up
                sr_out[:, R:R + s, Cc:Cc + s] = hr[0].cpu().numpy()
                prob[R:R + s, Cc:Cc + s] = torch.sigmoid(logits)[0, 0].cpu().numpy()
        return sr_out, prob


@lru_cache(maxsize=4)
def load_arm(arm: str, weights_dir: str | None = None) -> Arm:
    root = Path(weights_dir) if weights_dir else REPO / "demo" / "weights"
    d = root / arm
    cfg = json.loads((d / "config.json").read_text())
    return Arm(d / "model.ckpt", cfg, pick_device())


@lru_cache(maxsize=256)
def _tile(cell_id: str, tiles_dir: str | None = None) -> np.ndarray:
    """One 512 px tile as (4,512,512) float32, fetched lazily and cached.

    One file per tile on the Hub is what makes this cheap: a cold Space pulls
    2.1 MB for the cell it was asked about, not the 2.6 GB dataset.
    """
    if tiles_dir:
        return np.load(Path(tiles_dir) / f"{cell_id}.npy").astype(np.float32)
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(DATASET_REPO, f"tiles/{cell_id}.npy", repo_type="dataset")
    return np.load(p).astype(np.float32)


def assemble(cells: dict, site: str, row: int, col: int, size: int,
             sub_row: int = 0, sub_col: int = 0, tiles_dir: str | None = None):
    """Resolve a selection to (4,size,size) raw reflectance.

    Sizes 128/256/512 are size-aligned inside ONE stored tile. Only 1024 spans
    tiles, and only whole ones -- exactly the 2x2 block whose UTM contiguity
    build_selections.py verified to 0.0000 m.
    """
    grid = cells[site]
    if size <= TILE_PX:
        a = _tile(grid[f"{row}_{col}"]["id"], tiles_dir)
        r0, c0 = sub_row * size, sub_col * size
        return a[:, r0:r0 + size, c0:c0 + size]
    out = np.empty((4, TILE_PX * 2, TILE_PX * 2), dtype=np.float32)
    for dr in (0, 1):
        for dc in (0, 1):
            out[:, dr * TILE_PX:(dr + 1) * TILE_PX,
                   dc * TILE_PX:(dc + 1) * TILE_PX] = _tile(
                       grid[f"{row + dr}_{col + dc}"]["id"], tiles_dir)
    return out
