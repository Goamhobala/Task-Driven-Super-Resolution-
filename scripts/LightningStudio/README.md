# Running InstaRoad experiments on Lightning Studio

This folder is the [Lightning Studio](https://lightning.ai) counterpart of
`scripts/hpc/`. Same experiments, same engines, same replication contract — the
only things that change are the ones tied to the cluster: SLURM is gone, the
`/scratch/$USER` layout becomes a single configurable data root, and the Optuna
search defaults to one GPU.

The experiment logic (the `unet/`, `sr/`, `loss/` engines and arm scripts) is a
faithful copy of the HPC versions. Only paths, the environment, and the GPU
fan-out were adapted.

## 1. One-time setup

Clone the repo into your Studio (with submodules), then run setup:

```bash
git clone --recursive https://github.com/InstaRoad/InstaRoadPrototype.git
cd InstaRoadPrototype

bash scripts/LightningStudio/setup.sh              # UNet baseline + loss ablation
bash scripts/LightningStudio/setup.sh --sr         # + joint-SR arms (SEN2SR / SR4RS)
bash scripts/LightningStudio/setup.sh --sr --mamba # + full (Mamba) SEN2SR / r3 arms
```

`setup.sh` installs `uv`, builds the repo-local `.venv`, installs the right
optional-dependency groups, and scaffolds the data root. The engines activate
that `.venv` for you — you don't need to activate anything by hand.

Then add your data and log in to W&B:

There is no `/scratch` on Lightning, so the data folders sit **right next to the
repo** — `INSTAROAD_ROOT` defaults to the repo's parent directory:

```
<parent>/                        # = INSTAROAD_ROOT (the repo's parent dir)
├── InstaRoadPrototype/          # the repo you cloned
├── ROSA_all/                    # loss ablation + the _all SR series
├── ROSA_Dense_CDNGI/            # unet baseline + the _cdngi SR series
├── models/
│   ├── SEN2SRLite_RGBN/model.safetensor          # sr r1/r2/r7
│   └── SR4RS_RGBN/gen_weights.safetensors ...     # sr r4/r5/r6
├── runs/         benchmarks/    benchmarks_loss/  # created by setup.sh
```

```bash
wandb login          # or:  export WANDB_MODE=offline
```

## 2. Configuration — `env.sh`

`env.sh` is the single source of truth (the SLURM-header analogue). Everything
is overridable from your shell or as a `KEY=VALUE` on the command line:

| Variable | Default | Meaning |
| --- | --- | --- |
| `INSTAROAD_ROOT` | the repo's parent dir | Datasets, weights, `runs/`, benchmark stores — placed beside the repo. Override to point at a Lightning Drive. |
| `VENV_DIR` | `<repo>/.venv` | uv virtualenv the engines activate. |
| `SEARCH_GPUS` | `1` | Optuna workers (one per GPU). Bump on a multi-GPU machine. |
| `PRECISION` | `bf16-mixed` | **On a free-tier T4, set `PRECISION=16-mixed`** — Turing has no native bf16. |
| `WANDB_MODE` | `online` | `offline` keeps logs local. |

## 3. Running experiments

| Task | Command |
| --- | --- |
| One stage | `bash scripts/LightningStudio/run.sh <arm> STAGE=tune\|fit\|bench [KEY=VALUE ...]` |
| Full chain | `bash scripts/LightningStudio/run_both.sh <arm> [KEY=VALUE ...]` |
| Two arms / a sweep | `bash scripts/LightningStudio/run_pair.sh --A=<arm> --B=<arm> [A.K=V] [B.K=V] [K=V]` |
| Detach (keep running) | `bash scripts/LightningStudio/job.sh <run\|run_both\|run_pair> <args...>` |
| Separate Lightning Job | `python scripts/LightningStudio/submit_job.py --machine L4 -- run_both <arm> ...` |

`STAGE` is `tune` → `fit` → `bench` (UNet/SR default to `tune`; loss arms to
`fit`). `run_both.sh` chains all three. Every run tees its own log into
`$INSTAROAD_ROOT/runs/…`.

### Command mapping from `scripts/hpc/`

```
sbatch scripts/hpc/train.sbatch --SCRIPT=unet/cdngi.sh STAGE=tune
  ->   bash scripts/LightningStudio/run.sh unet/cdngi.sh STAGE=tune

sbatch --gres=gpu:1 scripts/hpc/train.sbatch --SCRIPT=sr/r2a_all.sh STAGE=fit SEED=1
  ->   bash scripts/LightningStudio/run.sh sr/r2a_all.sh STAGE=fit SEED=1

sbatch scripts/hpc/train_both.sbatch --SCRIPT=loss/la0_all.sh SEED=2
  ->   bash scripts/LightningStudio/run_both.sh loss/la0_all.sh SEED=2

sbatch scripts/hpc/train_pair.sbatch --A=loss/l1_all.sh --B=loss/la0_all.sh
  ->   bash scripts/LightningStudio/run_pair.sh --A=loss/l1_all.sh --B=loss/la0_all.sh
```

### The experiment families

- **`unet/`** — UNet baseline: `cdngi.sh` (CDNGI labels), `osm.sh` (OSM labels).
- **`sr/`** — resolution / joint-SR arms: `r0` bicubic, `r1` frozen SEN2SR,
  `r2` cold joint SEN2SR, `r4` SR4RS, `r5` frozen SR4RS, `r6`/`r7` staged
  (warm-started) SEN2SR/SR4RS. `_all` = ROSA_all, `_cdngi` = ROSA_Dense_CDNGI;
  `a`/`b` = reflect-pad on/off.
- **`loss/`** — loss-function ablation: `l1`…`l10`, `la0`/`la1` (see each arm's
  header for the loss it maps to).

## 4. Notes for the free tier

- **Single GPU.** `SEARCH_GPUS=1`, `REFIT_GPUS=1`, and `run_pair.sh` runs its two
  sides sequentially. On a multi-GPU machine everything fans back out.
- **Watch the 80 GPU-hours.** A `tune` search (100–200 trials × 8 epochs) plus a
  100-epoch refit is the expensive part. To economise: lower `N_TRIALS` /
  `TUNE_EPOCHS` for smoke tests, run `STAGE` by `STAGE` rather than the full
  chain, and use `job.sh` / `submit_job.py` so a dropped connection doesn't waste
  a run. Prototype on a CPU machine, switch to GPU only for the actual training.
- **Smoke test first:**
  ```bash
  bash scripts/LightningStudio/run.sh unet/cdngi.sh STAGE=tune N_TRIALS=2 TUNE_EPOCHS=1
  ```

Some inline comments in the arm/engine scripts still describe the original SLURM
workflow (mentions of `sbatch`, `scancel`, `/scratch`, "the cluster"). They're
documentation only — the executable defaults and the "Next:" hints all target
the Lightning `run.sh` commands above.
