"""InstaRoad demo — ZeroGPU inference Space (plan §5 Phase 3).

CONTRACT
--------
`/predict` takes a CELL ID and returns images. The browser never uploads
pixels: the Space holds its own copy of the imagery and a request is a few
dozen bytes. That is also why the UI can call this directly from the browser --
ZeroGPU bills its daily quota to the CALLER, so each visitor spends their own
allowance instead of draining one server token for everyone.

ZEROGPU RULES THIS FILE OBEYS
-----------------------------
* Models are placed on cuda at MODULE level, not inside @spaces.GPU. CUDA is
  emulated outside the decorator precisely so this works, and the docs are
  explicit that lazy placement inside is slower.
* No torch.compile (unsupported).
* Gradio SDK only.
* `duration` is kept tight: quota bills EFFECTIVE runtime, and a shorter
  declared duration wins better queue priority.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

try:
    import spaces                                    # ZeroGPU runtime
except ImportError:                                  # local dev
    class _Shim:
        @staticmethod
        def GPU(*a, **k):
            return (lambda f: f) if not a or not callable(a[0]) else a[0]
    spaces = _Shim()

from infer import assemble, load_arm                 # noqa: E402

HERE = Path(__file__).resolve().parent
CELLS = json.loads((HERE / "data" / "cells.json").read_text())
STRETCH = json.loads((HERE / "data" / "stretch.json").read_text())
TILES_DIR = os.environ.get("TILES_DIR")              # set for a local copy
# In the deployed bundle the weights sit next to app.py; locally they live in
# demo/weights, so the path is overridable rather than duplicated on disk.
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", str(HERE / "weights"))
ARMS = ["r2a"]                                       # rl4 joins once theta* lands

# Module-level placement: on ZeroGPU this runs under CUDA emulation and is the
# CORRECT time to do it. Loading lazily inside @spaces.GPU is the slow path.
MODELS = {a: load_arm(a, WEIGHTS_DIR) for a in ARMS}


def _rgb(chw: np.ndarray, site: str) -> Image.Image:
    """SR bands -> 8-bit RGB using the SITE's pooled 2-98 stretch.

    The same stretch the chips were built with, so the panel a visitor sees
    next to the map is on the map's own scale rather than autoscaled per
    request (which would make brightness look like a model difference).
    """
    s = STRETCH[site]
    lo = np.asarray(s["lo"])[:, None, None]
    hi = np.asarray(s["hi"])[:, None, None]
    x = np.clip((chw[:3] - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return Image.fromarray((x.transpose(1, 2, 0) * 255).astype(np.uint8))


def _overlay(rgb: Image.Image, mask: np.ndarray) -> Image.Image:
    a = np.asarray(rgb).astype(np.float32)
    m = mask[..., None].astype(np.float32)
    tint = np.array([255.0, 64.0, 96.0])            # road = warm, against cool imagery
    return Image.fromarray((a * (1 - 0.55 * m) + tint * 0.55 * m)
                           .clip(0, 255).astype(np.uint8))


@spaces.GPU(duration=60)
def predict(site: str, row: int, col: int, size: int, arm: str,
            sub_row: int = 0, sub_col: int = 0):
    t0 = time.time()
    if arm not in MODELS:
        raise gr.Error(f"{arm} is not servable yet (no tuned θ*)")
    if site not in CELLS:
        raise gr.Error(f"unknown site {site!r}")
    model = MODELS[arm]

    img = assemble(CELLS, site, int(row), int(col), int(size),
                   int(sub_row), int(sub_col), TILES_DIR)
    sr, prob = model.predict(img)
    mask = prob > model.theta

    rgb = _rgb(sr, site)
    meta = {
        "arm": arm, "site": site, "cell": f"r{row}_c{col}",
        "size_px": int(size), "ground_km": round(size * 10 / 1000, 2),
        "window_px": model.step, "theta": model.theta,
        "head": model.cfg.get("encoder_name") or "1x1 conv (linear probe)",
        "road_fraction": round(float(mask.mean()), 5),
        "seconds": round(time.time() - t0, 2),
    }
    return rgb, Image.fromarray((mask * 255).astype(np.uint8)), _overlay(rgb, mask), meta


with gr.Blocks(title="InstaRoad — road extraction from Sentinel-2") as demo:
    gr.Markdown(
        "## InstaRoad\n"
        "Road extraction from 10 m Sentinel-2 via learned super-resolution. "
        "This Space is the inference back end for the map UI; it takes a cell "
        "id, not an image."
    )
    with gr.Row():
        with gr.Column(scale=1):
            i_site = gr.Textbox(label="site id")
            with gr.Row():
                i_row = gr.Number(label="row", value=0, precision=0)
                i_col = gr.Number(label="col", value=0, precision=0)
            with gr.Row():
                i_sr = gr.Number(label="sub row", value=0, precision=0)
                i_sc = gr.Number(label="sub col", value=0, precision=0)
            i_size = gr.Dropdown([128, 256, 512, 1024], value=512, label="size (px @ 10 m)")
            i_arm = gr.Dropdown(ARMS, value="r2a", label="model")
            go = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            o_sr = gr.Image(label="super-resolved 2.5 m", type="pil")
            o_ov = gr.Image(label="prediction overlay", type="pil")
            # PNG, not gradio 6's default WebP: that codec is LOSSY, and a
            # binary mask came back with 36 distinct grey values (0.12% of
            # pixels intermediate). Cosmetically invisible under the overlay,
            # but it corrupts any measurement taken from the returned mask --
            # including the parity check against local inference.
            o_mask = gr.Image(label="binary mask", type="pil", format="png",
                              visible=False)
            o_meta = gr.JSON(label="meta")

    go.click(predict,
             inputs=[i_site, i_row, i_col, i_size, i_arm, i_sr, i_sc],
             outputs=[o_sr, o_mask, o_ov, o_meta],
             api_name="predict")

if __name__ == "__main__":
    demo.launch()
