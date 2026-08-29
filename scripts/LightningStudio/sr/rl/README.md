# RL-series campaign — Lightning Studio

Implements `docs/rl_lightning_campaign_plan.md` (rev 3, 2026-08-29).
`docs/sr_linear_probe.md` stays the authority on the linear head and the θ*/AP
protocol; the campaign plan overrides it where they conflict.

## What is here

| file | role |
|---|---|
| `_rl_common.sh` | every between-arm constant: head, loss, budget, batch size, anorm, rails, protocol. Sourced **first** by every arm. |
| `_rl_rung.sh`   | one rung of the `lr_sr` ladder — the pin, the hold, the `_ls…` tag. Sourced by the joint arms only. |
| `rl0.sh` | bicubic (the probe floor) |
| `rl1.sh` | frozen SEN2SR-Lite, bare |
| `rl2.sh` | joint SEN2SR-Lite, bare, one rung |
| `rl3.sh` | frozen SR4RS, bare (write-up's r3; run-tag r5's twin) |
| `rl4.sh` | joint SR4RS, bare, one rung |
| `lane.sh` | drives a whole lane (frozen arm, then rungs extremes-first) through tune→fit→bench |
| `read_head_lr.py` | reads the pinned head lr out of the cluster rl3 study |

Engine: `../\_stages_tv.sh`, **generated** from the HPC twin by
`scripts/LightningStudio/sync_engine.py`. Do not hand-edit it — add the feature
to `scripts/hpc/sr/_stages_tv.sh` and re-run the sync (`--check` tells you if it
is stale).

## Before the first fit (blocking)

1. **Upload `ROSA_New` to the Studio** — imagery, `splits/`, `norm_stats.yaml`
   and the `mask_new_2pt5` COGs. Batch jobs inherit the studio's files, so the
   dataset never moves per job and no cloud bucket is involved. Training reads
   local disk only; the teamspace Drive carries checkpoints and results.
2. **SR weights** under `$INSTAROAD_ROOT/models/`: `SEN2SRLite_RGBN` and
   `SR4RS_RGBN`. Parity-verify the SR4RS port locally (`python -m sr.sr4rs_torch`)
   — there is no TensorFlow on the job boxes.

## Running

```bash
# cheapest first: the anchor every other arm is reported against
bash scripts/LightningStudio/sr/rl/lane.sh LANE=base

# then the two lanes, concurrently (concurrency 2 on the current tier)
bash scripts/LightningStudio/job.sh run sr/rl/lane.sh LANE=sen2sr
bash scripts/LightningStudio/job.sh run sr/rl/lane.sh LANE=sr4rs
```

A lane **stops after its first tune** until you pass `GATE1_OK=1`. That tune is
one trial of one epoch and does double duty: it writes `best_params.yaml` (which
`STAGE=fit` requires) and it *is* the plan's §4 gate-1 timing/VRAM measurement.
Read h/epoch and peak VRAM off it before committing a lane's walltime.

`N_TRIALS=0` cannot replace that first pass: it re-emits the overlay from an
existing study's completed trials, and a fresh arm has none — `sr.tune` exits
with *"0 COMPLETE trials … nothing to write"*. It is the right flag **after**
the 1×1 pass has run, to regenerate an overlay without spending another epoch.

Single runs, if you want to drive them by hand:

```bash
bash scripts/LightningStudio/run.sh sr/rl/rl2.sh STAGE=fit LRSR=1e-4 SEED=0
```

## The shape of one run

Every arm-seed is **one 30-epoch job**. Frozen arms: 30 epochs head-only. Joint
arms: epochs 1–10 with `lr_sr` held at **exactly 0** (an LR gate on the SR
parameter group — Adam's update is identically zero while its moments warm on
the real gradients), then 20 joint epochs on the rung's `lr_sr`, cosine to 0.

That is what makes the design fair by construction: with `lr_sr = 0` the joint
arm's first ten epochs *are* a frozen-arm run, so the branches diverge only at
epoch 11 and the frozen arm's epochs 11–30 *are* the matched-budget control.
30 = 30 — no inheritance, no `warm_start_head`, no stage pairing.

Implementation: `sr_hold_epochs` in `src/sr/model.py`
(`configure_optimizers`), plumbed through `sr.tune` and both engines, pinned by
`tests/test_sr_hold_ramp.py`. `sr_hold_epochs=0` — every arm already in the
store — reproduces the previous schedule bit-for-bit, and appends nothing to any
command line.

## Decisions this implementation makes

* **Holdout, not merge-val** (`TRAIN_SPLITS=train`). The campaign's primary
  evidence is within-run trajectories: "frozen val-AP plateaus before epoch 10"
  is the falsifiable check on the hold length, and "the joint arms' hold phases
  reproduce the frozen curves" is the harness self-check. Both need a val loop
  every epoch, and the R-series' merged protocol sets `limit_val_batches: 0` —
  there would be no curve. Nothing is tuned on val here, so nothing is lost.
  Consequences, all wanted: a `_holdout` tag keeps rl rows away from R-series
  rows; θ* is swept on val, which under this protocol is genuinely unseen; test
  stays the single report split.

  `TRAIN_SPLITS=train` is **not sufficient on its own** — `joint_sr_trainval.yaml`
  is layered last at the fit stage and sets `limit_val_batches: 0`
  unconditionally, which would give a holdout *split* with no holdout *curve*.
  `FIT_VAL_LOOP=1` (added to both engines, default off) restores the loop for
  holdout fits only, and restores no selection with it: no EarlyStopping, and
  the checkpoint is still the end of the fixed 30-epoch budget
  (`monitor: null`, `save_on_train_epoch_end`). The val loop only logs.
* **The "tune" stage is not tuning.** One trial, one epoch, every band pinned
  (`LR_MIN == LR_MAX`, `POS_WEIGHT_MIN == POS_WEIGHT_MAX`, `LR_SR_MIN == LR_SR_MAX`).
  Optuna's log-uniform on a degenerate band suggests the constant, which lands
  in `best_params.yaml` exactly like a searched value — the only reason
  `STAGE=fit` needs no special case.
* **`SR_HC=off` on all five arms** — the b-series lane throughout:

  | probe arm | twins | upsampler | freeze | pad | constraint |
  |---|---|---|---|---|---|
  | `rl1` | `r1b_new` (`SR_HC=off`, pad 0) | sen2sr | true | 0 | **off** |
  | `rl2` | `r2b_new` (`SR_HC=off`, pad 0) | sen2sr | false | 0 | **off** |
  | `rl3` | `r5_new` (native, pad 0) | sr4rs | true | 0 | **off** |
  | `rl4` | `r4b_new` (native, pad 0) | sr4rs | false | 0 | **off** |

  Never the a-series: `r1a`/`r2a` leave `SR_HC` at `native`, which for SEN2SR
  resolves to the constraint **on**, and they run pad 8 with it. `r4a` forces it
  on with SEN2SR's mask.

  On the SR4RS row `off` and `native` are the same operator (`resolve_sr_hc`
  returns off for `sr4rs` either way), so `rl3`/`rl4` are behaviourally
  identical to `r5`/`r4b`. Forcing it explicitly rather than inheriting it costs
  one thing and buys one thing: the run dirs gain a `_nohc` tag that `r5`/`r4b`
  do not carry, and in exchange no arm in the series has an implicit constraint
  state and all five are tagged alike. Bicubic is forced `off` for the same
  reason (there is no generator to constrain either way).
* **Head lr = 3e-3**, pinned for all five arms. The cluster rl3 study found
  val_ap flat in lr across [~2.4e-3, ~7.9e-3] (≥30 trials, 0.043–0.049 spread),
  which is the licence to pin rather than search; 3e-3 sits at the low end of
  that evidenced-flat region. It is a chosen constant, not that study's argmax —
  the study is evidence only (wrong loss, wrong platform to be an arm), so
  quoting its best trial to four figures would imply a precision the flat
  objective does not support. `_rl_common.sh` refuses an `HEAD_LR` outside
  [1e-3, 1e-2], where the flatness evidence stops.
* **λ = 2.4789710497080004**, copied from `sr_r0_new_wbce_holdout_seed0`'s
  overlay — a wbce tune at `batch_size=4` under this same holdout protocol.
  Never re-searched.

## A band exit never ends a run

`STD_BAND_ACTION=warn` plus rails at `[0.01x, 100x]`. Two independent layers,
both deliberate:

* the rails make the check almost impossible to trip (production is `0.5x/4.0x`);
* if it trips anyway, the `warn` branch prints once, logs `adapt_band_exit=1`
  every epoch thereafter, and **returns** — training continues to the end of the
  budget. `joint_sr.yaml`'s own fit-stage default is `warn` for the same reason
  (the 2026-08-19 `r4b_new` kill at epoch 13 of 100, on a decelerating trend
  that crossed the bound by 0.004, is what set it).

`sr.tune` is the one place that defaults to `raise`, so `_rl_common.sh` passes
`--std-band-action warn` to that stage too — with `lr_sr` pinned, a pruned trial
would remove its rung from the record rather than record a bad result.

Nothing else aborts a fit on drift either: the trainval callback list has no
`EarlyStopping`, so a NaN loss does not stop the run — it just produces NaN
metrics from that step on. That is the expected endpoint at the top rung and it
is **data**: record the step, add no machinery. The envelope is then read post
hoc from the logged `adapt_std_b*` curves at the step each band crosses the
nominal `0.5x`/`4.0x`, which retains the trajectory *after* the crossing — 
strictly more information than the raise gave.

Do not reach for `adaptive_norm_check_every=0` to quieten it: that silences the
warn stream and the variance-floor diagnostic too, i.e. deletes the measurement.

## Store disjointness

New rows are e.g.

```
sr_rl2_new_ls1e-4_nohc_linear_wbce_anorm_recalpost_rails_holdout_ap_seed0
```

Loss tag (`_wbce` vs `_pstar_dice`), `_rails`, `_nohc` and `_holdout` all differ
from the 2026-08 cluster rl runs, so the append-only store cannot mix the two.
The rung is in `EXP_TAG`, so rungs are disjoint from each other by construction.

## Pre-registered, before the first Studio fit

1. Frozen arms plateau before epoch 10; joint arms' hold phases reproduce the
   frozen curves.
2. Decodability rises with `lr_sr` monotonically until instability; the top rung
   shows the strongest adaptation **or** numerical death — either is the
   unconstrained-degeneracy finding. Record the step at which it dies.
3. `rl2`/`rl4` gains (joint − hold-phase baseline) exceed the frozen − bicubic
   gaps. Falsification is reportable.
4. Even the best joint rung's probe AP stays far below the U-Net arms'
   segmentation quality — the degeneracy upper bound is small in absolute terms,
   i.e. the R-series gain is *not* mostly the SR acting as a segmenter.

Contingency: if the 1e-3 rung collapses within the first two joint epochs on
**both** rows, replace it with 3e-4. One dead rung is data; two is a wasted axis.

All claims are descriptive. At one seed per rung, between-rung metric gaps are
not interpreted against seed noise — trajectories and dose-response *ordering*
are the evidence, and dose (`lr_sr × joint epochs`) is what captions overlay
against the r2grid, never nominal `lr_sr`.
