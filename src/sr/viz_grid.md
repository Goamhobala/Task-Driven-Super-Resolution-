# viz_grid — R-series 2xN comparison figure

Top row: Original 10 m crop, then each experiment's **UNet input** (its own
SR/bicubic output in reflectance, after that checkpoint's `sr_pad` pad+crop —
read from the ckpt's saved hparams). Bottom row: GT mask, then each
**prediction** (sigmoid > threshold), with optional test IoU/F1 under the label.
All image panels share one percentile stretch computed from the original crop.

## Setup: the `src/sr/examples/` folder (gitignored)

Everything defaults from here. Expected contents:

```
src/sr/examples/
  Durban_r4_c3.tif                        # example test tile (first .tif is used)
  Durban_r4_c3_mask.tif                   # GT mask (10 m or 2.5 m; auto-detected)
  model.safetensor                        # SEN2SR-Lite weights
  hard_constraint.safetensor
  unet_s2rosa_bicubic_best.ckpt           # R0
  unet_s2rosa_pretrained_nopad_best.ckpt  # R1b
  unet_s2rosa_pretrained_pad_best.ckpt    # R1a
  unet_s2rosa_jointsr_nopad_best.ckpt     # R2b
  unet_s2rosa_jointsr_pad_best.ckpt       # R2a
```

Missing ckpts keep their column but render as blank "(pending)" placeholders,
so partial grids stay column-aligned while runs finish. GT is simply
`{tile}_mask.tif` beside the image: the 10 m pipeline mask works as-is
(nearest-upsampled x4 for display) and a 2.5 m HR mask also works — the
resolution is auto-detected from the mask's dimensions.

## Commands (copy-paste)

```bash
# zero-config: convention ckpts, first tile in examples/, crop at (0,0)
python -m sr.viz_grid

# pick the crop + output name
python -m sr.viz_grid --row 256 --col 128 --out r_series_grid.png

# with test metrics under the prediction labels (all optional, repeatable)
python -m sr.viz_grid --row 256 --col 128 \
    --metrics R0:iou=0.29,f1=0.41 \
    --metrics R1b:iou=0.27,f1=0.39 \
    --metrics R1a:iou=0.30,f1=0.43 \
    --metrics R2b:iou=0.28,f1=0.40 \
    --metrics R2a:iou=0.31,f1=0.44

# GPU (faster; five 512px UNet forwards), different threshold
python -m sr.viz_grid --device cuda --threshold 0.5
```

Explicit overrides when not using the examples convention:

```bash
python -m sr.viz_grid --image <tile.tif> --sen2sr-dir <weights_dir> \
    --exp R0=<ckpt> --exp R1b=<ckpt> --exp R1a=<ckpt> \
    --exp R2b=<ckpt> --exp R2a=<ckpt> --mask <hr_gt.tif>
```

Notes: run where torch + the repo venv exist (`PYTHONPATH=src` or installed);
crop is pinned to 128 px (SEN2SR's FFT mask); metric numbers come from each
experiment's fit-stage `=== BENCHMARK ===` output (test-set scores, constant
across crops); use **test** tiles for thesis figures. If the R0 ckpt name in
your folder differs from `unet_s2rosa_bicubic_best.ckpt`, either rename it or
pass it via `--exp R0=...` (the convention table lives at the top of
`viz_grid.py`).
