# RL-series — linear-probe read-out on the SR front-ends (`rl0`–`rl4_new`)

Status: **plan, not implemented.** Nothing in this document has been run.
Sections are marked *observed* (read out of the current code), *derived*
(follows from the code), *inferred* (expected but untested) and
*recommended* (a decision to take).

## 1. What the series is for

The R-series compares SR treatments with a 24 M-parameter U-Net sitting on
top. The U-Net can compensate for a great deal — an SR front-end that
degrades the image can still be read successfully by a decoder with enough
capacity and receptive field. So `r2a − r1a` and `r4b − r5` measure "does
joint task-driven fine-tuning help *this decoder*", not "does the SR output
carry more road evidence".

Replacing the decoder with a per-pixel logistic regression removes the
compensation channel. The head has **no spatial context and no capacity**:
every bit of structure in its prediction has to have been put there by the
SR stage. Two things follow:

* **Frozen arms (`rl0`/`rl1`/`rl3`)** measure *linear spectral separability
  of road vs non-road in the SR output* — a property of the image, cleanly
  attributable to the upsampler.
* **Joint arms (`rl2`/`rl4`)** measure how much segmentation capability the
  pretrained generator can be *repurposed into* when the only read-out is
  linear. This upper-bounds the share of the R-series joint gain that is
  the SR net acting as a segmenter rather than as an image enhancer — which
  is the open question left by R2/R4 and is not answerable from those arms
  alone.

That second framing is the load-bearing one for the thesis. It is also the
one most likely to be misread, so §7 makes the degeneracy measurable rather
than asserted.

## 2. Arms

Five arms, all on `ROSA_New` under the `_new` (train+val refit) protocol,
sourcing `_stages_tv.sh` exactly like the R-series.

| tag | upsampler | `FREEZE_SR` | `SR_PAD` | R-series twin | what it isolates |
| --- | --- | --- | --- | --- | --- |
| `rl0_new` | `bicubic` | – | 0 | `r0_new` | separability with no learned SR — the series anchor |
| `rl1_new` | `sen2sr` | `true` | 8 | `r1a_new` | `rl1 − rl0` = separability added by frozen SEN2SR-Lite |
| `rl2_new` | `sen2sr` | `false` | 8 | `r2a_new` | `rl2 − rl1` = what task-driven adaptation writes into the image |
| `rl3_new` | `sr4rs` | `true` | 0 | `r5_new` | `rl3 − rl0` = separability added by the frozen WGAN-GP generator |
| `rl4_new` | `sr4rs` | `false` | 0 | `r4b_new` | `rl4 − rl3`, the unconstrained-generator counterpart of `rl2 − rl1` |

`SR_PAD` mirrors the twin arms (SEN2SR padded, SR4RS not). *Inferred:* a
purely spectral probe is the worst possible consumer of an FFT border
ring — it cannot contextually discount it — so if the padded/unpadded
contrast is ever worth rerunning, this series is where it would show up
largest. Not in scope; keep `rl1`/`rl2` padded and say so.

**Joint-arm init: warm-start the head from the frozen twin.** `rl2` takes
`rl1`'s final head; `rl4` takes `rl3`'s. This is exact LP-FT — the linear
probe is fully converged on the frozen-SR input distribution *before* any
gradient reaches the generator.

The cold-init alternative (head from scratch, relying on the `lr_sr` ramp to
delay SR adaptation) rests on an assumption that is *inferred and
unverified*: `lr_sr` is non-zero from step 2 of the ramp, so "the head
converges first" is a hope about relative timescales, not a guarantee.
Warm-starting removes the assumption rather than testing it.

It also cleans up the contrast. Under cold init, `rl2 − rl1` mixes "SR
unfrozen" with "the head followed a different early trajectory" — and since
early head gradients are precisely what shape the SR's adaptation, that
trajectory difference propagates into the quantity being measured.
Warm-starting makes SR trainability the only difference between the arms.

Take the **final** head, not an intermediate epoch: a 5-parameter
near-convex problem is converged long before epoch 30, and picking a
mid-training snapshot would introduce an arbitrary constant needing its own
justification.

Two implementation points:

* **Keep the SR warmup ramp** (`sr_warmup_epochs=1.0`). It costs nothing and
  preserves recipe similarity with the r-series joint arms.
* **Add a separate `warm_start_head` hparam — do not reuse
  `warm_start_unet`.** *Observed:* `model.py` sets
  `self._sr_warmup_epochs = … if (sr_train and not warm_start_unet) else 0.0`,
  so routing head warm-starting through that flag would silently zero the
  ramp. The auto-disable is correct for its own case (a staged U-Net warm
  start *is* the warmup) and wrong for this one.

Initialisation is a nuisance parameter here, not a research question, so
**do not run cold and warm as separate arms** — that is two extra joint arms
of GPU time before 4 September for a variable nobody is asking about. Pick
warm-start and state it in the protocol. If `rl2` shows instability, a
cold-init run becomes a diagnostic worth having — but only then.

## 3. The head

*Recommended:* `nn.Conv2d(in_channels, classes, kernel_size=1, bias=True)` —
4 weights + 1 bias — applied **after** the existing z-score adapter.

The adapter is affine and the head is linear, so keeping it does not change
the function class; it only conditions the optimisation, and it keeps the
`forward` path byte-identical to the U-Net arms. Bias initialised to
`logit(road base rate)`, weights zero — the probe starts at the class prior
instead of at an arbitrary point, which matters when a compound loss with a
skeleton slot is applied to a 5-parameter model.

It stays as `self.model` on `JointSRUNetLightning`, behind a new
`head: "unet" | "linear"` hparam. *Derived:* `benchmarking/runner.py:219`
calls `JointSRUNetLightning.load_from_checkpoint`, so the bench path, the
viz path and the `reflectance_scale` guard all keep working with no changes
if the head lives inside the existing module. Building a separate
LightningModule would fork all three.

When `head="linear"`, skip `build_model` entirely (don't construct and
discard a 24 M-param U-Net) and record `encoder_name`/`encoder_weights` as
`None` in hparams so no run can later be mistaken for an encoder ablation.

## 4. Code changes

| file | change |
| --- | --- |
| `src/sr/model.py` | `head` hparam; `LinearProbeHead`; skip U-Net construction; `warm_start_head` (§2, separate from `warm_start_unet`); fp32 region keyed on `head=="linear"` (§6.3); `configure_gradient_clipping` override for per-group clipping (§6.2); add `val_ap` metric (§5) |
| `src/sr/tune.py` | `--head`; drop `suggest_categorical("batch_size", …)` in favour of a scalar `--batch-size` (§6.1); don't `suggest_categorical("encoder_name", …)` when linear; configurable `MONITOR` (§5); widened default `lr` band |
| `scripts/hpc/sr/_stages_tv.sh` | `HEAD="${HEAD:-unet}"`; pass through to tune and fit; **`HEAD_TAG` into `RUN_DIR` / `STUDY_NAME` / `MODEL_NAME`**; `BENCH_THRESHOLD` → `--threshold`; stop passing `gradient_clip_val` (§6.2) |
| `scripts/hpc/sr/_warm_head_tv.sh` | resolve `STAGE1_TAG`'s `_final.ckpt` → `WARM_START_HEAD` for `rl2`/`rl4`; modelled on `_warm_tv.sh` but must **not** set `warm_start_unet` |
| `scripts/hpc/sr/rl{0..4}_new.sh` | five ~20-line arm scripts, same shape as `r0_new.sh` |
| `src/benchmarking/cli.py` | expose `--sweep-thresholds` on `eval` (the runner already implements it — `runner.py:338–402`) |

The `HEAD_TAG` is not cosmetic. *Observed:* the benchmark store is
append-only and `MODEL_NAME` defaults to
`sr_${EXP_TAG}${LOSS_TAG}${REG_TAG}${PROTO_TAG}` — no head field. A distinct
`EXP_TAG` (`rl*` vs `r*`) is already sufficient, but adding the tag makes the
head visible in the report without joining back to the run dir.

## 5. Metrics and the operating point — the blocking issue

**Observed, and this is the one that would silently invalidate the series:**
`UNetLightning._eval_step` binarises at `self.hparams.threshold` (0.5), and
`sr.tune` uses `MONITOR = val_iou` for the Optuna objective, the pruner
*and* early stopping. A logistic regression on 4 reflectance bands has no
reason to be calibrated at 0.5. Left as-is, the search would select the
learning rate that happens to put the decision boundary near 0.5 — the
hyperparameter search itself becomes a calibration search, and every arm's
result depends on an arbitrary constant.

*Recommended, in order of necessity:*

1. **Tune objective → `val_ap`.** Add `BinaryAveragePrecision` to the
   module and make `MONITOR` a CLI argument (default `val_iou`, so the
   R-series and the loss pilot are untouched). All five rl arms use
   `val_ap`. This makes model selection threshold-free.
2. **Headline test metric → AP.** *Derived:* `evaluate(...,
   sweep_thresholds=[...])` scores the same cached probabilities at every θ
   in one pass, so a dense θ grid (19 points, 0.05…0.95) yields the PR curve
   and its integral **at no extra forward cost**. AP on test involves no
   selection, so it is honest without a val set.
3. **θ\* for the reported IoU/APLS row.** Choose on val at the tune stage
   (θ is a hyperparameter, so this is protocol-legal), persist as
   `best_threshold` in `sweep.json` — which `benchmarking.cli` already reads
   (`cli.py:277–281`) — and pass it at bench via `BENCH_THRESHOLD`.

   **State the caveat rather than hiding it:** θ\* is selected from a
   15-epoch train-only tune model and applied to a 100-epoch train+val
   refit. Calibration moves with training length, so this transfer is
   approximate. That is exactly why AP is the headline and IoU@θ\* is
   secondary — the main claim must not rest on the transfer. Report
   IoU@0.5 alongside IoU@θ\* so the size of the calibration effect is
   visible.

APLS still requires a binarisation, so topology metrics are reported at θ\*
only, with the same caveat.

*Note for the write-up:* the r-arm rows in the existing store are all at
θ=0.5 and were selected on `val_iou@0.5`. rl-vs-r comparisons are therefore
qualitative unless the R-series is re-benched at θ\*. That re-bench is a
separate decision and is **not** a prerequisite for this series — every
contrast that carries the argument (`rl1 − rl0`, `rl2 − rl1`, `rl3 − rl0`,
`rl4 − rl3`) is internal to the rl series and consistently thresholded.

## 6. Search space and recipe

*Recommended* changes from the R-series defaults, with the reason each is
needed:

| knob | R-series | rl-series | why |
| --- | --- | --- | --- |
| `ENCODERS` | `resnet34` | n/a | no encoder; don't leave a dead categorical in the Optuna space |
| `LR_MIN`/`LR_MAX` | `1e-5`/`1e-2` | `1e-3`/`3e-1` | *inferred:* a 5-parameter model on z-scored inputs converges at a far higher LR than a 24 M-param U-Net; the old band may not contain the optimum |
| `LR_SR_MIN`/`MAX` | `1e-7`/`1e-3` | unchanged | the generator is the same net under the same task loss |
| `BATCH_SIZES` | `2 4 8` | **`4`, pinned — every arm, both series** | see §6.1 |
| `N_TRIALS` | 60 | 20 frozen / 40 joint | frozen arms search 1 dimension (`lr`), joint arms 2 (`lr`, `lr_sr`) |
| `LOSS_ARM` + θs, `pos_weight` | frozen control | **identical, pinned** | R-series rule. `SEARCH_THETAS=false`, `SEARCH_MIX_W=false`, `POS_WEIGHT_MIN==MAX`. See §9 — the values to pin need confirming against what the `r*_new` arms actually ran |
| `REFIT_EPOCHS` | 100 | 100 | between-arm constant; do not change for one arm |

### 6.1 Batch size is a pinned constant, not a searched hyperparameter

`BATCH_SIZES=4`, pinned, for **every rl arm and every r arm**. Not searched
anywhere.

*Observed:* `JointSRRoadTileDataset`'s `length` is fixed per epoch
independently of batch size, so a trial at `bs=2` takes twice the optimiser
steps of a trial at `bs=4` within the same epoch budget. Optuna's `val_iou`
therefore rewards small batches for a reason that has nothing to do with
batch size, and the "best" batch size a study reports is an artefact of the
step-count coupling. *Derived:* the only way the comparison across arms
stays clean is for batch size to be a between-arm constant, exactly like
`REFIT_EPOCHS` and the loss arm.

`4` is the value: it is inside the feasible set for the heaviest arm
(`r4b`/`r5`/`rl3`/`rl4` cap at 4 — SR4RS runs 256-channel convs, one of them
9×9, at the full 512 px grid, and 8 OOMs on 44 GB), so one number covers
both series with no per-arm exception.

**Consequence for the existing R-series, stated plainly.** The `r*_new` and
`r*_all` runs searched batch size, so each arm's `best_params.yaml` carries
whatever value its study landed on — plausibly different values for
different arms. Pinning to 4 makes those refits inconsistent with the new
rule, and re-running them means re-running tune **and** fit per arm. Two
things follow, and they should be decided before any rl arm is launched:

1. Whether the R-series is re-run under the pin, or the pin applies to new
   work only and the existing rows are reported with their searched batch
   sizes and a stated caveat.
2. If re-run: `BATCH_SIZES=4` must be set on the **tune** stage, not patched
   into `best_params.yaml` afterwards — the tune stage's LR search is
   conditioned on the batch size it saw, and a resumed fit at a changed
   batch size also breaks the cosine `T_max`.

Recording the actual searched values from the existing runs' `best_params.yaml`
before overwriting anything is worth doing regardless: if every arm already
chose 4, the question is moot.

### 6.2 Gradient clipping: per-group, not global, and never searched

`CLIP` is **not** a searched hyperparameter. Searching it would reintroduce
§6.1's defect under a different name: arms would land on different clip
values, so the head's treatment would again vary across the contrast.

But the problem is not that `1.0` is the wrong number. *Observed:* Lightning's
`gradient_clip_val` applies a **single global L2 norm** across all trainable
parameters. In `rl1` that group is 5 parameters; in `rl2` it is those 5 plus
~240 k SEN2SR parameters. The same setting therefore means something
different in each arm — the head's effective step size in `rl1` is
determined by the head's own gradient norm, and in `rl2` by a norm that
SEN2SR dominates. The nuisance variable moves with the treatment, which is
exactly the condition that makes a comparison unfair.

*Recommended:* clip **per parameter group**, by overriding
`configure_gradient_clipping` on the module rather than passing
`gradient_clip_val` to the Trainer:

* SR group: L2 norm clipped at `1.0` — matching what the r-series joint arms
  did, so `rl2`'s SR is treated as `r2a`'s SR was.
* Head group: unclipped.

Then `rl1`'s head and `rl2`'s head are treated identically, the SR group's
stability mechanism is preserved where it does work, and there is no search
and no per-arm variation.

Turning clipping off entirely would also be a valid constant, but it removes
the stability mechanism recipe-v2 added specifically for the joint arms.
Per-group keeps it.

### 6.3 Precision: key the fp32 region on the head, not on the SR stage

Mixed precision across stages is not itself unfair — the same policy applied
to every arm is part of the fixed recipe, not a treatment. The asymmetry is
elsewhere.

*Observed:* the `autocast(device_type=…, enabled=False)` block in
`JointSRUNetLightning.forward` is keyed on the **SR stage** (it exists
because `torch.fft` has no BFloat16 kernels). `rl0` has no SR network — it is
bicubic upsampling with a head on top. So under the plan as first written,
`rl0`'s head would run in bf16 while `rl1`–`rl4`'s heads run in fp32.

*Derived:* bf16 carries 8 mantissa bits, which produces far more tied logit
values than fp32. AP is a ranking statistic, so ties degrade the estimate.
`rl0` is the series anchor and appears in two of the four load-bearing
contrasts (`rl1 − rl0`, `rl3 − rl0`), so a systematically degraded AP there
biases both.

*Recommended:* key the fp32 region on `head == "linear"`, not on the presence
of an SR stage. All five arms then run the head in fp32 and the precision
policy is genuinely constant across the series. Negligible cost — it is a
1×1 convolution.

## 7. Degeneracy diagnostics for `rl2` / `rl4`

The joint arms have an obvious failure mode that is also, arguably, the
result: with a linear read-out and a segmentation-only loss, the generator's
lowest-resistance solution is to **paint the road mask into its output
channels** — the "SR image" becomes a road-probability map in reflectance
coordinates. If that happens, `rl2 − rl1` measures conv-net capacity, not
image improvement.

This must be measured. *Recommended instruments:*

* **Image-space drift, always on.** Mean absolute difference and PSNR
  between `SR_ft(x)` and `SR_0(x)` on a fixed held-out probe batch, logged
  per epoch. *Observed:* the existing `sr_drift_rel` is weight-space and
  cosine-confounded (the `r4b`/`r2a` curves differ by a near-constant 1.47×),
  so it cannot answer this question. Image-space can.
* **Band-statistics shift.** Per-band mean/std and inter-band correlation of
  the fine-tuned output vs the pretrained output. Mask-painting shows up as
  the four bands collapsing toward mutual correlation ≈ 1 — a single road
  channel replicated — which is a sharp, falsifiable signature.
* **Snapshots.** `SR_SNAPSHOT_EVERY=2` (already implemented, weights-only)
  plus `viz_sr.py` gives a frame-by-frame replay. For `rl2`/`rl4` this is
  evidence, not a demo.

No HR reference exists for these tiles, so PSNR against ground-truth imagery
is not available — which is the same reason R2 uses task loss alone. The
pretrained output is the only available reference, and that is sufficient
for a *drift* claim (not for a *fidelity* claim). Say it that way in the
write-up.

## 8. Confounds to state explicitly

* **Capacity, not just adaptation.** `rl2 − rl1` = task-driven adaptation
  **+** ~240 k newly-trainable SEN2SR parameters against `rl1`'s 5. In the
  R-series the U-Net's 24 M dwarfs the generator, so `r2a − r1a` is mostly
  adaptation; here the generator *is* the model. Do not describe `rl2 − rl1`
  as "the value of adaptation" — describe it as "how much of a segmenter the
  generator becomes". Head-trajectory differences are *not* part of this
  confound: warm-starting (§2) controls for them, which is most of why it was
  chosen over cold init.
* **Two selection criteria.** rl arms selected on `val_ap`, r arms on
  `val_iou@0.5`. Internal rl contrasts are unaffected; cross-series ones are
  qualitative.
* **The linear head is not a fair segmentation baseline** and no claim
  should read as though it were. Absolute rl numbers will be poor. The
  series is diagnostic; the quantity of interest is always a difference
  between rl arms.

## 9. Open items to confirm before writing code

1. **Which loss the `r*_new` arms actually ran.** `_stages_tv.sh` defaults to
   `LOSS_ARM=gap_tl_ce`, but the loss pilot's final pair going into Phase C
   was `pstar_sdice` + `gap_t4_ce`. The rl series must pin whatever the
   `r*_new` runs pinned, or the frozen-control rule is broken between the two
   series. Check a `r*_new` run log or `train_meta.json`.
2. **`pos_weight`.** The default band is a pinned constant
   (`3.352251180486363`) — confirm that is the λ\* the R-series used.
3. **Base rate for the head bias init.** Road-pixel fraction of
   `ROSA_New`'s train+val at 2.5 m.
4. **What batch size each existing `r*_new` arm actually used.** Read
   `best_params.yaml` per run dir. Decides whether §6.1's re-run question is
   live or moot.

## 10. Order of operations

Gates first, cheap arms first, so a protocol error surfaces before the
expensive SR4RS arms burn walltime.

`rl2` warm-starts from `rl1` and `rl4` from `rl3`, so within each SR family
the frozen arm's **fit** must complete before the joint arm's **tune** can
start. Gates first, cheap arms first, so a protocol error surfaces before the
expensive SR4RS arms burn walltime.

1. Resolve §9. Implement §4 + §5.1 + §6 recipe fixes.
2. **Gate A — local, no cluster.** `rl0` for 2 epochs on a handful of tiles.
   Assert: head has exactly 5 trainable parameters; `sr_warmup` is off
   (frozen/bicubic); `val_ap` logs; **logits are fp32 even with no SR net
   present** (§6.3 — this is the assertion that catches the anchor bug);
   Trainer-level `gradient_clip_val` is unset and the per-group hook fires.
3. **Gate A2 — local.** `rl2` for 2 epochs from a throwaway `rl1` head.
   Assert: the head loads and matches the source; `sr_warmup_epochs` is
   still `1.0` (i.e. `warm_start_head` did *not* trip the
   `warm_start_unet` auto-disable — §2); the SR group is clipped and the
   head group is not.
4. **Gate B.** `rl0_new STAGE=tune` with `N_TRIALS=6`. Confirm the best-trial
   LR sits inside the widened band and not at an edge — if it pins to
   `3e-1`, the band is still too narrow. Confirm `batch_size` appears
   nowhere in the study's search space.
5. `rl0` and `rl1` full (tune → fit → bench), SEED=1. Then `rl2` full.
6. **Gate C — read `rl2` before launching SR4RS.** If §7's diagnostics show
   the generator has collapsed to mask-painting, that is the headline result
   and `rl4` becomes a confirmation rather than an exploration — which may
   change how much budget it deserves.
7. `rl3` full, then `rl4` full, SEED=1.
8. Seeds 2–3 only for arms carrying a reported contrast. *Derived:* the
   Wilcoxon signed-rank test runs over per-tile scores (n = test tiles), so
   the seeds buy training-variance error bars, not inferential power — one
   seed is enough to know whether an arm is worth three.
