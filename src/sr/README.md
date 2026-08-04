# sr — resolution-enhancement experiments (RQ A2: R0 / R1 / R2)

One LightningModule ([model.py](model.py) `JointSRUNetLightning`, a subclass of
the baseline `unet.model.UNetLightning`) covers all three configs, so the
segmentation network, its loss, the per-crop IoU/F1 metrics and the `val_iou`
checkpoint convention are **identical to the U-Net baseline** — only the
upsampler treatment varies:

| Exp | Upsampler                    | SEN2SR weights                                              | How to select                          |
| --- | ---------------------------- | ---------------------------------------------------------- | -------------------------------------- |
| R0  | bicubic ×4 (parameter-free)  | –                                                          | `model.upsampler: bicubic`             |
| R1  | SEN2SR-Lite RGBN ×4          | frozen (`freeze_sr: true`, runs under `no_grad`)           | `model.upsampler: sen2sr` + `freeze_sr` |
| R2  | SEN2SR-Lite RGBN ×4          | fine-tuned by the **segmentation loss alone** at a low LR  | `model.upsampler: sen2sr` (default)     |

R2 is the task-driven SR variant (Haris et al.): there is **no reconstruction /
L1 / perceptual term** — the only loss is the baseline's segmentation loss on
the 2.5 m mask, and the Figure-1 `∇L_seg × α` scaling is implemented as
differential learning rates (`model.lr_sr` ≪ `model.lr`, α = lr_sr/lr) via two
Adam param groups in [`configure_optimizers`](model.py).

```
# once, on a node with internet (compute nodes have none):
python -c "from sr.sen2sr_loader import download_sen2sr; download_sen2sr('<model_dir>')"
# once, after building the V2 dataset:
python -m sentinel2data.cli norm-stats --dataset-dir <root> --out src/unet/configs/norm_stats.yaml
# for OSM HR labels (mask_source: raster), in OpenStreetMapTest:
python dataset_hr_masks.py --dataset-dir <root> --scale 4

# train / test (R2 by default; override model.upsampler / freeze_sr for R0 / R1):
python -m sr.cli fit  --config src/sr/configs/joint_sr.yaml --config src/unet/configs/norm_stats.yaml
python -m sr.cli test --config ... --ckpt_path checkpoints/unet_s2rosa_jointsr_best.ckpt

# hyperparameter search (Optuna, joint LR pair) + HPC two-stage flow:
python -m sr.tune --base-config src/sr/configs/joint_sr.yaml \
                  --base-config src/unet/configs/norm_stats.yaml --dataset-dir <root>
sbatch scripts/hpc/train.sbatch --SCRIPT=sr_tune_only.sh        # search  (SEN2SR: R1/R2)
sbatch scripts/hpc/train.sbatch --SCRIPT=sr_tune_r0.sh          # search  (bicubic R0; no lr_sr)
sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr_fit_only.sh STUDY_TAG=<tag>   # refit + test
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
gradients reach the SR parameters.

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
   as a buffer. Its fixed 512×512 size also pins LR patches to **128×128**
   (checked in `JointSRUNetLightning.forward`).

## Normalisation adapter

SEN2SR consumes/produces surface reflectance (DN/10000); the baseline U-Net
consumes per-band z-scored DN using the frozen train stats
(`norm_stats.yaml`). The adapter in `JointSRUNetLightning.forward` is exactly
that transform — `(reflectance × 10000 − mean) / std` with the `[B4,B3,B2,B8]`
slice of the same stats — as a differentiable buffer-based affine, so the
U-Net's input distribution matches the baseline **and** gradients flow through
it into SEN2SR. The `JointSRDataModule` therefore feeds **raw DN** (no
normalisation in the dataloader); `normalize` / `norm_mean` / `norm_std` are
declared on the datamodule only so the shared `UNetCLI` links hand them to the
model, which normalises *after* super-resolution.

## Assumptions about the on-disk layout (flag if wrong)

Reads the V2 tiled layout written by `sentinel2data generate` (split CSVs under
`<root>/splits/`, 512×512 tile COGs under `<root>/<split>/imagery/`), sharing
the split definition with the baseline/benchmarking. The dataloader
([sentinel2data.dataset.joint_sr_dataset](../sentinel2data/dataset/joint_sr_dataset.py))
returns **native 128 px** crops (raw DN) paired with **512 px** (2.5 m) masks
from one of two sources (`data.mask_source`):

* `graph` (default): the tile's `masks_graph` parquet (the pipeline's CDNGI
  labels) rasterised on the fly at the 2.5 m transform.
* `raster`: pre-generated HR mask COGs at `<root>/<split>/<mask_dirname>/{tile}.tif`
  (default `mask_osm_2pt5`), dims **exactly `upscale`× the tile's** —
  e.g. the OSM masks from `OpenStreetMapTest/dataset_hr_masks.py --scale 4`.
  Asserted per tile at dataset init. Enables the OSM-vs-CDNGI label comparison.

Two more contracts:

* The V2 COGs store bands `[B4, B3, B2, B8, ...]` (`S2_V2_BANDS`; SR uses bands
  1–4 = `S2_10M`), which is **already** SEN2SR's required `[B04,B03,B02,B08]` =
  R, G, B, NIR order — used as-is, no permutation. Do **not** substitute the
  CLAHE+gamma enhanced-RGB bands (21–23): SEN2SR expects raw reflectance.
* Band stats come from `sentinel2data norm-stats` (`norm_stats.yaml`,
  full-stack 1-based DN-unit mean/std; the `[B4,B3,B2,B8]` slice is taken for
  the adapter). These are **required** — the post-SR adapter has no per-image
  fallback, so `JointSRUNetLightning` raises without them.

## Segmentation net & loss

Inherited unchanged from `unet.model.UNetLightning` (smp U-Net, default encoder
`resnet34`; the baseline's own loss and metrics). Keep `--model.encoder_name`
identical across R0/R1/R2 — encoder constancy is the point of the ablation.
No auxiliary / reconstruction losses are added anywhere.

## Hyperparameter search

[tune.py](tune.py) runs Optuna over the **joint LR pair** (`lr` for the U-Net,
`lr_sr` for SEN2SR, both log-uniform with `lr_sr` well below `lr`), plus
`pos_weight` and `batch_size`; the encoder is held constant. For the **R0**
baseline (`--upsampler bicubic`) there are no SR params, so `lr_sr` is **not
searched** — use [scripts/hpc/sr_tune_r0.sh](../../scripts/hpc/sr_tune_r0.sh)
(or pass `--upsampler bicubic` to `sr.tune`). The best trial is written as a
`best_params.yaml` overlay (carrying the resolved `upsampler`) that you layer
onto the base config for the full-length refit.

## Two protocols: `_all` (holdout) and `_new` (final)

The same arms run under two split protocols. They are kept in separate script
families so their results can never end up in the same benchmark row.

| | `r*_all.sh` → `_stages.sh` | `r*_new.sh` → `_stages_tv.sh` |
| --- | --- | --- |
| dataset | `ROSA_all` | `ROSA_New` (final) |
| tune | train → score on val | *identical* |
| refit | train only | **train + val** |
| stopping | EarlyStopping on `val_iou` | **fixed, pre-registered budget** |
| checkpoint | `..._best.ckpt` (argmax over val) | `..._final.ckpt` (end of budget) |
| reported on | test | test |

The `_new` series exists because the val split is a *budget*, not a permanent
reservation. Optuna spends it choosing hyperparameters; after that, holding
those tiles out of the fit costs ~17% of the training data and buys no extra
inferential guarantee — test is still untouched either way. So the final models
are refit on train+val.

What that costs, and how it is paid: with val folded in there is no honest
signal left to early-stop or select a checkpoint on, so
[`configs/joint_sr_trainval.yaml`](configs/joint_sr_trainval.yaml) removes that
machinery rather than letting it peek at training data — `limit_val_batches: 0`,
no `EarlyStopping`, `monitor: null`. The epoch budget is therefore a
**between-arm constant**: change `REFIT_EPOCHS` for one arm and the comparison
is void. Recipe v2's cosine has `T_max = max_epochs`, so a fixed budget still
ends at LR 0 — the schedule completes rather than being cut off.

The checkpoint is deliberately named `unet_s2rosa_jointsr_final.ckpt`, never
`..._best.ckpt`: `_best` means "argmax over a holdout" everywhere else in this
repo, and the staged arms (`r6`/`r7`) warm-start from stage 1's checkpoint, so
a name collision there would quietly reintroduce val-based selection into the
final numbers. `_warm_tv.sh` looks for `_final` and says so if it finds a stray
`_best` instead.

```sh
# per arm: search on train/val, refit on train+val, score test
bash scripts/hpc/submit.sh sr/r0_new.sh  STAGE=tune
bash scripts/hpc/submit.sh sr/r0_new.sh  STAGE=fit
bash scripts/hpc/submit.sh sr/r0_new.sh  STAGE=bench
# or all three in one allocation
bash scripts/hpc/submit.sh --both sr/r0_new.sh
```

`TRAIN_SPLITS=train` reverts a `_new` arm to the holdout protocol on the same
dataset (tagged `_holdout` in the run dir, study and benchmark name) if you
want the merge itself as an ablation.

### Norm stats come from the dataset

`sentinel2data.cli norm-stats` writes `<dataset_dir>/norm_stats.yaml` by
default, so each dataset already carries the stats computed from **its own**
`splits/train.csv`. `_stages_tv.sh` reads that file, and echoes the resolved
path plus its provenance at the top of every run log.

It falls back to the repo copy at `src/unet/configs/norm_stats.yaml` only when
the dataset has none, and then refuses to start unless you pass
`NORM_FALLBACK_OK=1`. That copy belongs to whatever dataset it was last
hand-copied from — its header still names `ROSA_RandomSampling110zones` — so on
any other dataset it is a wrong mean/std applied to every input the model ever
sees, with nothing downstream to flag it.

The older `_stages.sh` (and the `unet` / `loss` engines) still hard-code the
repo copy. Worth confirming it matched `ROSA_all` when the `_all` series ran.
