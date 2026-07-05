# sr — resolution-enhancement experiments (RQ A2: R0 / R1 / R2)

One LightningModule ([module.py](module.py) `JointSRSegModule`) covers all three
configs so everything except the upsampler treatment is held constant:

| Exp | Upsampler                    | SEN2SR weights                                                 | How to select       |
| --- | ---------------------------- | -------------------------------------------------------------- | ------------------- |
| R0  | bicubic ×4 (parameter-free) | –                                                             | `--experiment R0` |
| R1  | SEN2SR-Lite RGBN ×4         | frozen (`freeze_sr=True`, runs under `no_grad`)            | `--experiment R1` |
| R2  | SEN2SR-Lite RGBN ×4         | fine-tuned by the**segmentation loss alone** at a low LR | `--experiment R2` |

R2 is the task-driven SR variant (Haris et al.): there is **no reconstruction /
L1 / perceptual term** — the only loss is `RoadSegLoss(logits, 2.5 m mask)`,
and the Figure-1 `∇L_seg × α` scaling is implemented as differential learning
rates (`--lr-sr` ≪ `--lr-seg`, α = lr_sr/lr_seg), plus an optional warm-up
(`--freeze-sr-steps` holds the SR LR at 0, `--sr-lr-ramp-steps` ramps it up).

```
python -m sr.train --data /scratch/$USER/InstaRoad --experiment R2 --out runs/sr_r2
sbatch scripts/train_sr.sh          # HPC; edit the CONFIG block
python -m sr.smoke --sen2sr-dir <dir> [--download]   # laptop-friendly sanity checks
```

## How the trainable SEN2SR was obtained (the non-obvious part)

`mlstac.load(dir).compiled_model()` is inference-only, and the advertised
`trainable_model()` is *also* broken for training: both construct
`CNNSR(..., train_mode=False)`, and in that mode every `Conv3XC` block
re-collapses its conv/sk branch into a frozen `eval_conv` with **detached**
weights on every forward — `loss.backward()` reaches the input but no
parameter ever receives a gradient. [sen2sr_loader.py](sen2sr_loader.py)
instead instantiates the same architecture with `train_mode=True` and strictly
loads the shipped `model.safetensor` (which contains the train-branch
weights). Verified: output matches the compiled model to max|Δ| ≈ 2e-6, and
gradients reach the SR parameters (see `sr.smoke`).

Three further SEN2SR facts the code depends on:

1. **The FFT hard constraint steals the DC bin.** The frozen `HardConstraint`
   takes the zero-frequency component of the output entirely from the
   bicubic-upsampled LR input. Consequence: a loss that only probes the
   spatial mean has *exactly zero* gradient w.r.t. SEN2SR — a naive
   `output.mean().backward()` smoke probe reports a dead network that is in
   fact healthy. Any real (spatially varying) segmentation loss flows fine.
2. **Blocks 1–4 are structurally dead.** Upstream `CNNSR.forward` feeds the
   stem output to every SPAB block instead of chaining them, and only consumes
   block 0's and the last block's outputs. The shipped weights were *trained*
   with this topology, so we keep it (do not "fix" the chaining) and mark the
   middle blocks `requires_grad=False` — identical training on 1 GPU, and DDP
   stops erroring on never-gradded parameters. ~240k of 472k params train.
3. **`low_pass_mask` is a plain attribute** (not a registered buffer), so
   Lightning's device moves would leave it on CPU. The loader re-registers it
   as a buffer. Its fixed 512×512 size also pins LR patches to **128×128**.

## Normalisation adapter

SEN2SR consumes/produces surface reflectance (DN/10000); the baseline U-Net
consumes per-band z-scored DN using the frozen `Data.npz` stats
(`RoadSegDataset`). The adapter in `JointSRSegModule.forward` is exactly that
transform — `(reflectance × 10000 − mean) / std` with the M0 slice of the same
stats — as a differentiable buffer-based affine, so the U-Net's input
distribution matches R0/R1 and the M-series baselines, and gradients flow
through it into SEN2SR. (The repo baseline does **not** use ImageNet input
normalisation; "match what R0/R1 apply" means the Data.npz z-score.)

## Assumptions about the on-disk layout (flag if wrong)

* `mask_2pt5m/{site}{suffix}.tif` are rasterised on the imagery grid upsampled
  by exactly 4 (so dims are exactly 4× the site's 10 m COG). Asserted per site
  at dataset init; a `scale` arg exists on `SRRoadSegDataset` if that changes.
  The mask filename suffix is auto-discovered (`resolve_mask_suffix`), same as
  the baseline.
* The combined COGs store bands `[B4, B3, B2, B8, ...]` (per
  `sentinel2data.processor.dataset.BAND_NAMES`), which is **already** SEN2SR's
  required `[B04, B03, B02, B08]` = R, G, B, NIR order — the M0 slice is used
  as-is, no permutation (`sr.sen2sr_loader.SEN2SR_BAND_ORDER` is the single
  point of truth if the layout ever changes).
* `Data.npz` mean/std are in DN units on the same 14-band layout (M0 slice
  taken for the adapter).
* NODATA (−32768) pixels are zeroed *before* the /10000 scaling, mirroring the
  baseline's zero-after-z-score convention in reflectance space.

## Segmentation net & loss

The segmentation model is the baseline's own construction
(`baseline.model.build_model` — smp UNet++), default encoder `resnet34`
(proposal Table 5); keep `--encoder` identical across R0/R1/R2 — encoder
constancy is the point of the ablation. The criterion is injectable: pass a
built instance to `JointSRSegModule(criterion=...)` or select by name via
`--loss` (registry in `sr.module.LOSS_REGISTRY`, default `roadseg` =
`baseline.model.RoadSegLoss` with `pos_weight` computed from the 2.5 m train
masks). No auxiliary losses are added anywhere.

## Augmentation

Only the baseline's lossless default (D4 flips/rotations) is carried over,
applied to the LR image and HR mask with the same group element via torch ops
(Albumentations can't jointly transform an image/mask pair of different
sizes). The photometric extras in `baseline.augment` are tuned for z-scored
input and are deliberately not applied to reflectance.

## Tests

`python -m sr.smoke --sen2sr-dir <dir>` — synthetic data, no dataset needed:
asserts the 4× output size, non-zero grads in *both* param groups for R2
(printing per-group grad norms), zero SR grads for R1, and that a single fixed
batch overfits (loss halves) through the joint pipeline. Pytest wrappers live
in [tests/test_smoke.py](tests/test_smoke.py) (set `SEN2SR_DIR`; overfit is
opt-in via `RUN_OVERFIT=1`). On a laptop use `--batch 1 --overfit-steps 30 --device cpu`; the 512×512 U-Net stage is heavy.
