# Running the loss pilot on Modal

Third port of the same harness. The arms are still
`scripts/LightningStudio/loss/l*_new.sh` on the `sr/_stages_tv.sh` engine; only
the wrapper changes.

| | Lightning Studio | Kaggle | **Modal** |
|---|---|---|---|
| orchestrator | `LightningStudio/pilot_seq.sh` | `kaggle/pilot_kaggle.sh` | **`modal/pilot_modal.sh`** |
| you get code in by | cloning once into the Studio | cloning every run (notebook) | **mounting your working tree at call time** |
| state survives via | the Studio's disk | attaching the previous Output | **a Modal Volume** |
| session cap | none | 12 h, 30 GPU-h/week | **24 h/call, $30/month** |
| precision | `bf16-mixed` (L4) | `16-mixed` (T4) | **`bf16-mixed` (L4)** |

That last row is the reason to prefer Modal over Kaggle for the remaining
Phase A arms — see [Protocol notes](#protocol-notes).

## One-time setup

```sh
pip install modal
modal setup                                    # browser auth
modal secret create wandb WANDB_API_KEY=<key>
modal volume create instaroad-data
modal volume create instaroad-runs
```

Then upload the dataset. It must be a **complete `ROSA_New`**: imagery, the
three `splits/*.csv`, `norm_stats.yaml`, and the pre-rasterised
`<split>/mask_new_2pt5/` COGs (`LABELS=new` reads those; on-the-fly
rasterisation is the slow path you already retired).

```sh
modal volume put instaroad-data /path/to/ROSA_New /ROSA_New
```

~16 GB, one upload, and it stays. Volumes are $0.09/GiB/month with 1 TiB/month
free, so the dataset and every checkpoint you will ever write for this pilot
cost nothing.

Verify before spending any GPU credit — this re-runs the engine's own dataset
preconditions plus an import check, on a CPU container, for a fraction of a
cent:

```sh
modal run scripts/modal/modal_app.py --action check
```

## Running

```sh
# time one 1-epoch trial on the target GPU (~2 cents). Do this once per GPU
# type; it is the only honest way to turn $30 into "how many arms".
modal run scripts/modal/modal_app.py --action probe --arms "l3_new"

# the real thing — --detach or closing your laptop kills it
modal run --detach scripts/modal/modal_app.py \
    --action run --arms "l3_new l4a_new l4b_new"

# fan out: one container per arm, all at once
modal run --detach scripts/modal/modal_app.py \
    --action run --arms "l3_new l4a_new l4b_new" --parallel

modal run scripts/modal/modal_app.py --action status
modal run scripts/modal/modal_app.py --action report
```

`--action run` is **idempotent**, exactly like the other two ports: re-invoke it
after any interruption and it skips finished work, tops a partial Optuna study
up to `--trials`, and resumes a killed fit from `last.ckpt`. Everything lives on
the `instaroad-runs` Volume, committed every 5 minutes and once more on exit, so
a preemption costs minutes.

Useful flags: `--gpu A10`, `--cpu 8`, `--memory 32768`, `--hours 23`,
`--trials 30`, `--workers 2`, `--seed 0`, `--no-wandb`.

## Benchmarking finished FINAL-protocol runs (`--action final-bench`)

Different job, different protocol, different entrypoint. `--action sweep` drives
the **loss pilot**: holdout arms (`TRAIN_SPLITS=train`), discovered by their
`sr_<exp>_<tag>_holdout_seed<N>` dir names, swept and benched on `val`.
`--action final-bench` drives **finished merge-val runs** — the ones whose dir
names carry no `_holdout` tag because `TRAIN_SPLITS='train val'`:

| | `--action sweep` | `--action final-bench` |
|---|---|---|
| arms come from | name discovery under `RUNS_ROOT` | the explicit table in `final_bench.sh` |
| θ\* swept on | `val` | `val` |
| benched on | `val` | **`test`** |
| SR weights | one `SEN2SR_DIR` for all | **per-arm** (`SEN2SRLite_RGBN` / `SR4RS_RGBN` / none) |
| store | `benchmarks_loss_pilot_theta` | `benchmarks_final_wbce` |

Both sweep θ\* on `val`, which for a merge-val run is *in-sample*. That is the
deliberate choice `_stages_tv.sh` documents at its bench stage: the only other
candidate is `test`, and selecting the operating point on the split you report
is the one thing that would actually invalidate the number. Record it as a
caveat — θ\* is fit on data the model saw — and note that it biases every arm
identically, so the ranking stands.

```sh
# once, on the account that will run it
bash scripts/modal/upload_final_bench.sh          # ~4.5 GB, skips what's there

# 4 tiles per arm, proves the plumbing (~10 min, a few cents)
modal run scripts/modal/modal_app.py --action final-bench --max-tiles 4

# the real thing
modal run --detach scripts/modal/modal_app.py --action final-bench

# re-report later without re-inference
modal run scripts/modal/modal_app.py --action report \
    --store /out/benchmarks_final_wbce --by-stratum
```

### Where the tables end up

The reports run **inside** the bench container, not chained from the local
client. That costs a few minutes of otherwise-idle GPU and buys a `.md` that is
on the Volume whether or not your laptop survived the run — the failure mode
that cost a run on 2026-08-11, when a `.remote()` call was cancelled by a
client disconnect.

Each bench writes, into the store dir **and into every `runs/` folder that
contributed an arm**:

```
report_<select_on>_all.md              whole test split
report_<select_on>_by_stratum_<S>.md   one per stratum (Urban/PeriUrban/Rural)
```

The `<select_on>` in the name is what keeps three tunings of the same arms —
`iou_mean`, `f1_mean`, `buffered_f1_mean` — from overwriting each other. The
copy into `runs_sr_wbce/` and `runs_gap_tl_ce/` means
`modal volume get instaroad-runs /runs_gap_tl_ce` brings the tables down beside
the runs they describe.

Re-report an existing store any time, on CPU, with no re-inference:

```sh
modal run scripts/modal/modal_app.py --action report \
    --store /out/benchmarks_final_wbce --by-stratum \
    --out /out/benchmarks_final_wbce/report_iou_mean_by_stratum.md \
    --copy-to "/out/runs_sr_wbce /out/runs_gap_tl_ce"
```

The bench itself is never repeated — the store's duplicate guard sees the arm
and skips it.

The stratified numbers are a **slice** of the same chips, not a second pass:
chip metrics are per-chip and strata are per-tile, so `report --by-stratum` is
the same arithmetic as a per-stratum eval with none of the GPU time. Score once
over the whole split, then slice.

Adding an arm means one line in `DEFAULT_ARMS` in `final_bench.sh` —
`run_dir|model_name|exp_tag|seed|sr_weights_subdir`, where `model_name` must
reproduce `_stages_tv.sh`'s `sr_${EXP_TAG}${LOSS_TAG}${REG_TAG}${PROTO_TAG}`
exactly, since that is what the store keys on.

`--action check` will report `train/` as MISSING after this upload. That is
correct and expected: `upload_final_bench.sh` skips the 7.3 GB training split
because nothing on the sweep → bench → report path reads it.

## Whole-tile figures (`--action viz`)

`src/sr/viz_single.py` renders one 128 px crop in a four-panel row. For a
figure you can put in a document you want the whole tile, which is a different
problem: SEN2SR's FFT mask pins its LR input to 128 px, so a 512 px tile cannot
go through in one pass at all. `src/sr/viz_tile.py` windows the tile the way
`runner._score_tile_sr` does — `_required_lr` for a pinned model, the bench's
256 px footprint cell otherwise — so the picture shows **the same prediction
that produced the numbers in the store**, not a similar-looking one rendered
with different convolution borders.

```sh
modal run scripts/modal/modal_app.py --action viz \
    --tiles "Durban_IndianCoastal_-29p851_30p94_Urban_r0_c3 Grassland_-27p78_30p54_Rural_r3_c4"

modal volume get instaroad-runs /figures ./figures      # bring the PNGs down
```

Per arm x tile it writes `_sr.png` (the SR image the UNet saw — bicubic ×4 for
r0), `_pred.png`, `_gt.png` and a `_pair.png` contact sheet, all at the native
2048 × 2048. The three single-panel files are borderless, one pixel per array
element, so panel spacing stays a document-layout decision.

θ\* is read from each run's `sweep.json`, so the figures binarise at the arm's
own operating point rather than 0.5. **Run this after the bench**, both because
the sweep must exist and because the two would race on the same file.

The percentile stretch is computed once from the original 10 m reflectance and
reused for every arm, so a panel looking sharper is the SR net differing rather
than an autoscale artefact. `--stretch-from-sr` opts out.

## Sequential or parallel?

**Cost is the same.** Modal bills per GPU-second; four arms on four containers
for 5 h each bills identically to one container for 20 h. What differs:

- *Parallel is slightly worse on paper* — each container pays its own cold
  start (image pull, imports, volume attach), so N arms pay N startups instead
  of one. A couple of minutes each. Noise against a 5-hour arm.
- *Parallel is better in practice* — you see all the val benches on the same
  day instead of at the end of the week, and the credit you save by killing a
  hopeless arm early dwarfs the startup overhead.
- *Sequential is safer for the first run* — if something is wrong, it burns one
  GPU's worth of credit before you notice, not four. On $30 that matters.

So: **first arm sequential as a canary, then `--parallel` for the rest.** The
Starter plan allows 10 concurrent GPUs, which is more than the pilot has arms.

Parallel is safe for the store, and this is worth knowing rather than assuming:
each arm writes its own `runs/sr_r0_new_<tag>_holdout_seed0/`, and
`benchmarking.store` writes **one parquet shard per `run_id`** (`_write_shard`),
not a shared appended file. No two containers ever write the same path.

## What $30 buys

Modal bills GPU **plus** CPU **plus** RAM. The CPU/RAM line is the part that
surprises people — at `cpu=4, memory=16 GiB` it adds $0.32/h, which is 40% on
top of an L4. Rates below are $/hour for the whole container:

| GPU | GPU | +cpu 4 | +ram 16 GiB | **total** | **hours on $30** |
|---|---|---|---|---|---|
| T4 | 0.59 | 0.19 | 0.13 | 0.91 | 33 |
| **L4** | **0.80** | 0.19 | 0.13 | **1.12** | **27** |
| A10 | 1.10 | 0.19 | 0.13 | 1.42 | 21 |
| L40S | 1.95 | 0.19 | 0.13 | 2.27 | 13 |
| A100 40 GB | 2.10 | 0.19 | 0.13 | 2.42 | 12 |
| H100 | 3.95 | 0.19 | 0.13 | 4.27 | 7.0 |

Going to `cpu=8` costs another $0.19/h (27 h → 23 h on an L4). Worth it only if
a probe shows the gap/tl skeleton pools starving the GPU.

Convert to arms with the probe, not with this table:

```
tune/arm ≈ trials × ~6 effective epochs × probe_seconds ÷ search_gpus
fit/arm  ≈ 50 × probe_seconds × (8830 ÷ TUNE_LENGTH)
```

`--action probe` prints exactly this with your measured number substituted in.

**H100 is a trap for these arms.** `gap_*` and `tl_*` build their weight maps
with CPU skimage skeletonisation every step, so an H100 would spend most of the
premium waiting on four cores. The $/work ratio only justifies the big cards
when the GPU is actually the bottleneck; here it often is not.

## Protocol notes

**Precision is a between-arm variable, and Kaggle already breaks it.**
`pilot_kaggle.sh` pins `PRECISION=16-mixed` because the T4 has no native bf16,
while the six finished Lightning arms ran `bf16-mixed`. fp16 and bf16 differ in
exponent range, not just speed — with `pos_weight` up to 40 and per-map mean-1
normalisation, the loss scales involved are exactly where that shows up. It is
probably a small effect, and the grad-scaler is there for precisely this, but it
is an uncontrolled hardware difference sitting inside a loss ablation, and a
reader can ask about it.

Modal's L4 removes the question: same card class, same `bf16-mixed`, same batch
8 as the Lightning arms. `pilot_modal.sh` **refuses to start** if
`PRECISION=bf16-mixed` is requested on a GPU without native bf16 — you have to
set `16-mixed` explicitly and log the amendment.

If some arms have already run at fp16 on Kaggle, that is worth one line in the
amendment log either way.

**Batch stays 8** (amendment 2026-08-03a). On Modal an arm that will not fit
batch 8 is fixed with `--gpu A10` or `--gpu L40S`, not by dropping the batch.

**Provenance.** The repo is mounted from your working tree, so the local git
SHA is stamped into each run's log and a dirty tree prints a warning before
launch. Commit before a run you intend to cite.

## Gotchas

- **`--detach` or it dies.** Without it, closing the terminal ends the run.
- **Module-level code runs in *both* places.** Modal re-imports `modal_app.py`
  inside the container to find the Function object, and it lands flat at
  `/root/modal_app.py` — not at `<repo>/scripts/modal/`. So anything at module
  scope that assumes the local directory layout (`__file__`, `parents[n]`,
  reading a file next to the script) has to survive that too. `_repo_root()`
  probes rather than assumes for exactly this reason.
- **The `wandb` secret must exist even with `--no-wandb`.** It is part of the
  Function definition, so Modal resolves it when the app registers; `--no-wandb`
  only swaps it out at call time and sets `WANDB_MODE=offline`. Create it with a
  dummy value if you genuinely don't want W&B.
- **The mount excludes big paths.** `models/`, `src/**/examples/`,
  `**/dummy_data/`, `*.ckpt`, `*.tif` are in `_IGNORE` in `modal_app.py` —
  ~7.8 GB of the repo that the training path never reads. SR weights for the
  later r-arms belong on the data Volume at `/data/models/SEN2SRLite_RGBN`.
- **Deps come from `pyproject.toml`** (`[unet, sentinel2, benchmarking]`, minus
  a `_DROP` list of cloud SDKs, docs tooling, and `sknw`). Edit an extra and the
  image rebuilds on the next run; nothing else needs touching. If an import
  fails in the container, delete a line from `_DROP`.
- **The `sr` extra is in the image, minus `mamba-ssm`.** It was absent until
  2026-08-11 because every loss-pilot arm is `UPSAMPLER=bicubic`, which builds
  no SR net — so the gap only surfaced when the first `sen2sr` arm was benched
  (`ModuleNotFoundError: No module named 'sen2sr'`, after the θ sweep had
  already run). `mamba-ssm` stays in `_DROP`: it backs only `sen2sr_full`, and
  it compiles CUDA kernels at install time. `sr4rs` arms need nothing extra —
  `src/sr/sr4rs_torch.py` is a local port with no dependency past torch.
- **`sknw` is dropped on purpose.** It pins `numba==0.53.1` → `llvmlite==0.36.0`,
  which will not build on Python ≥3.10, so a clean Linux/3.12 resolve fails at
  image build. Its only importer is
  `sentinel2data/generator/skeleton_graph.py` — dataset generation, which never
  runs here. `benchmarking/graph_metrics.py` already refuses to depend on it for
  the same reason. If you ever do need graph *generation* in a container, add a
  modern `numba` alongside it rather than un-dropping it bare.
  Worth fixing upstream at some point: `sknw` sits in the `sentinel2` extra, so
  every consumer of that extra inherits a build that only works on Python <3.10.
  A separate `datagen` extra would make the split honest.
- **Adding an arm means four files now**: the `TAG` table in `pilot_modal.sh`,
  `pilot_seq.sh`, `pilot_kaggle.sh`, and `TAG` in `modal_app.py`.
- **Retries are capped at 2.** They exist for preemption, where the idempotent
  resume makes them free. A deterministic bug 20 h in would otherwise retry at
  full GPU rate.
- **Academic credits exist.** Modal grants graduate students and labs up to
  $10k — <https://modal.com/academics>. Worth an application if this pilot
  works out on Modal, since it would cover the whole R-series.

## Files

- `modal_app.py` — image, volumes, secrets, container shape, entrypoint.
- `pilot_modal.sh` — the tune → fit → bench loop. Port of `pilot_kaggle.sh`
  minus the state-restore block, plus a bf16 guard.
