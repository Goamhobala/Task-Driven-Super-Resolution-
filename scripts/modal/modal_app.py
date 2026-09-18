"""Modal driver for the loss pilot — the third port of the same harness.

Kaggle and Lightning Studio are *machines you sit on*: you clone/upload once and
run bash. Modal is a *dispatcher*: this file describes the container (image,
GPU, CPU, RAM, disks, secrets) and then runs exactly the same bash inside it.
Nothing about the experiment protocol is defined here — the arms are still
``scripts/LightningStudio/loss/l*_new.sh`` on the ``_stages_tv.sh`` engine, and
the orchestration is ``scripts/modal/pilot_modal.sh``.

    modal setup                                   # once, on your laptop
    modal secret create wandb WANDB_API_KEY=...    # once
    modal volume create instaroad-data
    modal volume put instaroad-data /path/to/ROSA_New /ROSA_New

    modal run scripts/modal/modal_app.py --action check          # verify the data
    modal run scripts/modal/modal_app.py --action probe          # ~2c: time one trial
    modal run --detach scripts/modal/modal_app.py \
        --action run --arms "l3_new l4a_new l4b_new"
    modal run scripts/modal/modal_app.py --action status
    modal run scripts/modal/modal_app.py --action report

``--detach`` is not optional for real runs: without it, closing the terminal
kills the job.

Two Volumes, deliberately split:
  ``instaroad-data`` -> /data   the dataset + SR weights. Written once, by you,
                               from the laptop. Never written by a run.
  ``instaroad-runs`` -> /out    runs/, checkpoints, Optuna study.db, the
                               benchmark store. Written constantly, committed
                               every few minutes so a preemption costs minutes,
                               not hours.

The repo is *mounted from your laptop at call time* (``add_local_dir``), not
cloned from GitHub like the Kaggle notebook does. So there is no push-then-run
loop and no image rebuild when you edit a shell script — but it also means the
code that ran is your working tree, dirty or not. The local git SHA (with a
``-dirty`` suffix when it applies) is stamped into every run's log, and a dirty
tree prints a warning before launch.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import modal

try:                            # stdlib on 3.11+; the repo targets 3.12+
    import tomllib
except ModuleNotFoundError:     # pragma: no cover - older interpreter
    import tomli as tomllib     # pip install tomli

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
APP_NAME = "instaroad-loss-pilot"

REPO_REMOTE = "/root/InstaRoadPrototype"
DATA_MNT = "/data"
OUT_MNT = "/out"
DATASET_SUBDIR = "ROSA_New"                          # <data vol>/ROSA_New


def _readable_file(path: Path) -> bool:
    """`Path.is_file()` that answers False instead of raising on a denied stat."""
    try:
        return path.is_file()
    except OSError:
        return False


def _repo_root() -> Path:
    """The repo, as seen by whichever machine is executing this file.

    Module-level code runs TWICE: once on your laptop, to define the App, and
    again inside the container, because Modal re-imports this module to find
    the Function object. Locally the file sits at
    ``<repo>/scripts/modal/modal_app.py``; in the container Modal copies it to
    ``/root/modal_app.py``, where ``parents[2]`` does not exist. Anything at
    module scope has to survive both, so this probes instead of assuming.
    """
    here = Path(__file__).resolve()
    if len(here.parents) > 2 and _readable_file(here.parents[2] / "pyproject.toml"):
        return here.parents[2]           # laptop: <repo>/scripts/modal/ -> <repo>
    return Path(REPO_REMOTE)             # container: the mounted copy


REPO_LOCAL = _repo_root()

DATA_VOL_NAME = "instaroad-data"
RUNS_VOL_NAME = "instaroad-runs"

# ---------------------------------------------------------------------------
# Compute defaults
#
# L4 is the default for a reason that is protocol, not price: it has native
# bf16, so arms run here are numerically identical to the finished Lightning L4
# arms. The T4 is cheaper and would silently change precision — see the guard
# in pilot_modal.sh. cpu=4 matches Kaggle/Lightning so throughput comparisons
# carry over; bump it to 8 if the gap/tl skeleton pools are starving.
# ---------------------------------------------------------------------------
DEFAULT_GPU = "L4"
DEFAULT_CPU = 4.0            # physical cores (Modal bills 2 vCPU per core)
DEFAULT_MEMORY = 16384       # MiB

# --action final-bench is inference, not training, and its container shape is a
# COST decision rather than a protocol one — no numbers change with it.
#
# cpu=2: the bench alternates GPU forward passes with tile metrics, and both
# APLS and clDice are serial Python loops over chips (see
# benchmarking.tile_metrics — no thread/process pool anywhere in the package).
# There is no DataLoader either; rasterio reads on the main thread. So cores 3
# and 4 sit idle at $0.047/core/h while one core does the graph work. Two
# leaves headroom for BLAS/GDAL without paying for parallelism that does not
# exist. Raise it only if a profile shows numpy threading actually helping.
#
# memory=8 GiB: the working set is a stitched tile (~4k x 4k at 2.5 m -> tens
# of MB per canvas), the chip batch, and the CUDA context. 16 GiB was sized for
# training minibatches, which this path never allocates.
FINAL_BENCH_CPU = 2.0
FINAL_BENCH_MEMORY = 8192
DEFAULT_TIMEOUT_H = 23.0     # Modal's hard ceiling is 24 h
BUDGET_MARGIN_S = 1800       # bash stops this long before the timeout

COMMIT_EVERY_S = 300         # Volume commit cadence during a run

# Published Modal rates, $/sec (modal.com/pricing, checked 2026-08-04). Used
# only for the pre-launch estimate printed below — never for billing.
GPU_RATE = {
    "T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542,
    "A100-40GB": 0.000583, "A100-80GB": 0.000694, "A100": 0.000583,
    "H100": 0.001097, "H200": 0.001261, "B200": 0.001736, "B300": 0.001972,
}
CPU_RATE = 0.0000131         # $/physical-core/sec
MEM_RATE = 0.00000222        # $/GiB/sec
FREE_CREDIT = 30.0           # Starter plan, $/month

# Arm -> loss tag. Mirrors the TAG table in pilot_modal.sh / pilot_seq.sh /
# pilot_kaggle.sh — used here only to locate run dirs for the status report.
# Extend all four together.
TAG = {
    "l1_new": "bce", "l2_new": "gap_ce", "l3_new": "tl_ce", "l9_new": "gap_tl_ce",
    "l10_new": "wbce", "l11_new": "sdice", "l12_new": "lcdice",
    "l15_new": "balance_ce", "l16_new": "dice",
    "l4a_new": "t2_ce", "l4b_new": "t4_ce",
    "l17_new": "gap_t2_ce", "l18_new": "gap_t4_ce", "l19_new": "gap_t2t4_ce",
    # Phase B compounds. These lagged pilot_seq.sh by a week — the "adding an
    # arm means four files" gotcha, caught 2026-08-12.
    "l5_new": "pstar_dice", "l13_new": "pstar_sdice", "l14_new": "pstar_lcdice",
    # Phase B compounds with P* pinned in the arm script: EXP_TAG=r0_new_<pstar>
    # + LOSS_TAG=_pstar_<region>, so the tag below IS the whole chain.
    "l20_new": "wbce_pstar_sdice", "l21_new": "wbce_pstar_lcdice",
    "l22_new": "tl_pstar_sdice", "l23_new": "tl_pstar_lcdice",
}


# ---------------------------------------------------------------------------
# Image
#
# Dependencies come from pyproject.toml's [unet, sentinel2, benchmarking]
# extras — the same three groups scripts/LightningStudio/setup.sh installs — so
# the container and the Studio drift together or not at all. The list is read
# on YOUR machine when Modal loads this file, and Modal hashes it: the image
# rebuilds when pyproject changes, and only then.
# ---------------------------------------------------------------------------
_EXTRAS = ("unet", "sentinel2", "benchmarking", "sr")

# Not needed by tune/fit/bench, and each one is either large, slow to resolve,
# or pulls cloud SDKs the container has no business having. If an import ever
# fails in the container, the fix is to delete a line here, not to pin a
# version somewhere else.
_DROP = (
    "earthengine-api",   # GEE export — dataset build, done long before this
    "kagglehub", "awscli", "eotdl",
    "mkdocs", "mkdocstrings", "mkdocs-gen-files", "mkdocs-literate-nav",
    "instaroadprototype", "instageo",
    # sknw -> numba==0.53.1 -> llvmlite==0.36.0, which refuses to build on
    # anything past Python 3.9 (it hard-guards the version in setup.py). The
    # only importer is sentinel2data/generator/skeleton_graph.py — dataset
    # *generation*, which never runs on Modal; the labels arrive pre-rasterised
    # on the Volume. The bench path already avoids sknw on purpose:
    # benchmarking/graph_metrics.py traces skeleton chains in pure
    # numpy/scipy/networkx/skimage "precisely so the eval environment does not
    # carry numba" (its own docstring). Dropping it here follows that decision
    # rather than inventing a new one.
    "sknw",
    # The `sr` extra exists for the r-arms' SR nets. Of its members only
    # `sen2sr` + `mlstac` + `safetensors` are reachable here:
    #   upsampler=sen2sr  -> sen2sr_loader.load_trainable_sen2sr -> CNNSR +
    #                        HardConstraint, loaded from safetensors
    #   upsampler=sr4rs   -> sr/sr4rs_torch.py, a local port with no deps past
    #                        torch (no TensorFlow anywhere)
    #   upsampler=bicubic -> nothing at all
    # mamba-ssm is the odd one out: it backs ONLY `sen2sr_full`
    # (sen2sr.models.opensr_baseline.mamba, imported lazily inside
    # load_trainable_sen2sr_full), and it compiles CUDA kernels at install
    # time — minutes of image build, needing a toolchain, for a code path no
    # arm on this account uses. Un-drop it if you ever bench a sen2sr_full arm.
    "mamba-ssm",
)


_REQ_RE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?:\[(?P<extras>[^\]]*)\])?(?P<spec>.*)$")


def _requirements() -> list[str]:
    """Base deps + the three extras, minus _DROP, with headless cv2.

    A package can appear in more than one extra with different floors (torch is
    `>=2.1.0` under `unet` and `>=2.11.0` under `sentinel2`). Rather than pick
    one and hope, the specifiers are concatenated — `torch>=2.1.0,>=2.11.0` is
    valid PEP 508 and lets the resolver intersect them, which is its job.
    """
    pyproject = REPO_LOCAL / "pyproject.toml"
    if not _readable_file(pyproject):
        # Container-side re-import with the repo not (yet) mounted. The image
        # was already built from the laptop's list, so returning nothing here
        # only keeps the import alive; it cannot change what is installed.
        return []

    data = tomllib.loads(pyproject.read_text())
    project = data["project"]
    extras = project.get("optional-dependencies", {})

    reqs = list(project.get("dependencies", []))
    for group in _EXTRAS:
        reqs += extras[group]

    specs: dict[str, list[str]] = defaultdict(list)
    xtras: dict[str, set[str]] = defaultdict(set)
    for req in reqs:
        m = _REQ_RE.match(req.strip())
        if not m:
            raise ValueError(f"cannot parse requirement from pyproject.toml: {req!r}")
        name, spec = m["name"], m["spec"].strip()
        if name.lower() in _DROP:
            continue
        if name == "opencv-python":
            # No X11/GL in the container; the headless wheel is the same cv2.
            name = "opencv-python-headless"
        if m["extras"]:
            xtras[name].update(e.strip() for e in m["extras"].split(",") if e.strip())
        if spec and spec not in specs[name]:
            specs[name].append(spec)
        specs.setdefault(name, [])

    out = []
    for name in sorted(specs):
        extra = f"[{','.join(sorted(xtras[name]))}]" if xtras.get(name) else ""
        out.append(f"{name}{extra}{','.join(specs[name])}")
    return out


# Everything the training path does not read. src/**/examples and
# **/dummy_data hold ~3.5 GB of checkpoints and GeoTIFFs; models/ holds 4.2 GB
# of SR weights (those belong on the data Volume, under /data/models). Mounting
# any of it would re-upload gigabytes on every single `modal run`.
_IGNORE = [
    ".git", ".git/**", ".venv", ".venv/**", ".pytest_cache", ".pytest_cache/**",
    "**/__pycache__", "**/__pycache__/**", "**/*.pyc",
    "**/examples", "**/examples/**", "**/dummy_data", "**/dummy_data/**",
    "models", "models/**", "dataset", "dataset/**", "processed", "processed/**",
    "runs", "runs/**", "sam_road", "sam_road/**", "instageo", "instageo/**",
    "docs/image", "docs/image/**", "site", "site/**",
    "*.png", "**/*.png", "*.tif", "**/*.tif", "*.ckpt", "**/*.ckpt",
    "*.zip", "**/*.zip", "*.safetensors", "**/*.safetensors",
    "*.parquet", "**/*.parquet",
    ".DS_Store", "**/.DS_Store",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    # libgl/libglib: cv2's runtime deps even in the headless wheel.
    # The GDAL/PROJ libs come bundled in the rasterio/pyogrio manylinux wheels.
    .apt_install("libgl1", "libglib2.0-0", "git")
)
_REQS = _requirements()
if _REQS:                       # empty only on a container-side re-import
    image = image.uv_pip_install(*_REQS)
image = (
    image
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(REPO_LOCAL), REPO_REMOTE, ignore=_IGNORE)
)

data_vol = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
runs_vol = modal.Volume.from_name(RUNS_VOL_NAME, create_if_missing=True)
VOLUMES = {DATA_MNT: data_vol, OUT_MNT: runs_vol}

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Container-side helpers
# ---------------------------------------------------------------------------
def _base_env(overrides: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        REPO_DIR=REPO_REMOTE,
        INSTAROAD_ROOT=OUT_MNT,
        DATASET_DIR=f"{DATA_MNT}/{DATASET_SUBDIR}",
        RUNS_ROOT=f"{OUT_MNT}/runs",
        STORE_DIR=f"{OUT_MNT}/benchmarks_loss_pilot",
        SEN2SR_DIR=f"{DATA_MNT}/models/SEN2SRLite_RGBN",
        VENV_DIR="/nonexistent",              # system python; engine tolerates it
        PYTHONPATH=f"{REPO_REMOTE}/src",
        PYTHONUNBUFFERED="1",
    )
    env.update({k: str(v) for k, v in overrides.items()})
    return env


def _arm_status(arms: list[str], seed: int) -> list[dict]:
    """Read the same on-disk markers the bash orchestrator skips on."""
    runs = Path(OUT_MNT) / "runs"
    out = []
    for arm in arms:
        tag = TAG.get(arm, arm)
        d = runs / f"sr_r0_new_{tag}_holdout_seed{seed}"
        out.append({
            "arm": arm,
            "tag": tag,
            "tuned": (d / "best_params.yaml").exists(),
            "fitted": (d / "checkpoints" / "unet_s2rosa_jointsr_final.ckpt").exists(),
            "benched": (d / ".bench_done").exists(),
        })
    return out


def _print_status(rows: list[dict]) -> None:
    print(f"{'arm':<10} {'loss':<12} {'tune':<6} {'fit':<6} {'bench':<6}")
    print("-" * 44)
    for r in rows:
        mark = lambda b: "  ok  " if b else "  --  "  # noqa: E731
        print(f"{r['arm']:<10} {r['tag']:<12} {mark(r['tuned'])} {mark(r['fitted'])} {mark(r['benched'])}")


# ---------------------------------------------------------------------------
# The training Function
#
# retries: preemption and transient CUDA/network faults are what we want to
# survive; the bash is idempotent, so a retry resumes from the Volume rather
# than restarting the arm. Kept low (2) because a *deterministic* bug 20 h into
# a run would otherwise burn the retry budget at full GPU rate.
# ---------------------------------------------------------------------------
@app.function(
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb")],
    timeout=int(DEFAULT_TIMEOUT_H * 3600),
    retries=modal.Retries(max_retries=2, initial_delay=0.0),
    single_use_containers=True,
)
def pilot(overrides: dict[str, str]) -> list[dict]:
    """Run pilot_modal.sh inside the container, committing /out as it goes."""
    data_vol.reload()
    runs_vol.reload()

    env = _base_env(overrides)
    arms = env.get("ARMS", "").split()
    seed = int(env.get("SEED", "0"))

    stop = threading.Event()

    def _committer() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            try:
                runs_vol.commit()
            except Exception as exc:            # never let a commit kill training
                print(f"[volume] commit failed (will retry): {exc}", flush=True)

    t = threading.Thread(target=_committer, daemon=True)
    t.start()

    started = time.time()
    try:
        proc = subprocess.run(
            ["bash", f"{REPO_REMOTE}/scripts/modal/pilot_modal.sh"],
            env=env,
            check=False,
        )
    finally:
        stop.set()
        t.join(timeout=30)
        try:
            runs_vol.commit()
        except Exception as exc:
            print(f"[volume] FINAL commit failed: {exc}", flush=True)

    mins = (time.time() - started) / 60
    print(f"\n[modal] pilot_modal.sh exited {proc.returncode} after {mins:.1f} min", flush=True)
    rows = _arm_status(arms, seed) if arms else []
    if rows:
        _print_status(rows)
    if proc.returncode != 0:
        # Raise so Modal's retry logic sees it; the run resumes from the Volume.
        raise RuntimeError(f"pilot_modal.sh failed with exit code {proc.returncode}")
    return rows


# ---------------------------------------------------------------------------
# Cheap CPU-only helpers (fractions of a cent)
# ---------------------------------------------------------------------------
@app.function(volumes=VOLUMES, timeout=900)
def check_data(seed: int = 0) -> None:
    """Re-run _stages_tv.sh's dataset preconditions against the Volume.

    Everything here is a guard the engine would hit ~40 s into a GPU run.
    Checking on a CPU container instead costs nothing and fails in seconds.
    """
    data_vol.reload()
    ds = Path(DATA_MNT) / DATASET_SUBDIR
    ok = True

    def need(path: Path, why: str) -> None:
        nonlocal ok
        hit = path.exists()
        print(f"[{'ok ' if hit else 'MISSING'}] {path}   {'' if hit else '<- ' + why}")
        ok = ok and hit

    print(f"== dataset: {ds} ==")
    need(ds, "modal volume put instaroad-data <local ROSA_New> /ROSA_New")
    for split in ("train", "val", "test"):
        need(ds / "splits" / f"{split}.csv", "all three split CSVs are required")
    need(ds / "norm_stats.yaml",
         "python -m sentinel2data.cli norm-stats --dataset-dir <local ROSA_New>, then re-upload")

    print("\n== HR labels (LABELS=new -> mask_source=raster, mask_dirname=mask_new_2pt5) ==")
    for split in ("train", "val", "test"):
        d = ds / split / "mask_new_2pt5"
        n = len(list(d.glob("*.tif"))) if d.is_dir() else 0
        print(f"[{'ok ' if n else 'MISSING'}] {d}  ({n} tif)")
        ok = ok and bool(n)
        img = ds / split / "imagery"
        print(f"        {img}  ({len(list(img.glob('*.tif'))) if img.is_dir() else 0} tif)")

    print("\n== output volume ==")
    out = Path(OUT_MNT)
    for sub in ("runs", "benchmarks_loss_pilot"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        print(f"[ok ] {out / sub}")
    runs_vol.commit()

    print("\n== imports ==")
    import importlib
    import sys
    sys.path.insert(0, f"{REPO_REMOTE}/src")
    for mod in ("torch", "lightning", "segmentation_models_pytorch", "optuna",
                "rasterio", "skimage", "networkx", "wandb", "jsonargparse",
                "sr.tune", "sr.cli", "unet.losses", "benchmarking.cli"):
        try:
            importlib.import_module(mod)
            print(f"[ok ] {mod}")
        except Exception as exc:
            ok = False
            print(f"[FAIL] {mod}: {type(exc).__name__}: {exc}")

    print("\n" + ("ALL CHECKS PASSED — safe to spend GPU credit." if ok
                  else "FAILURES ABOVE — fix before launching a GPU run."))
    if not ok:
        raise SystemExit(1)


@app.function(volumes=VOLUMES, timeout=900)
def status(arms: str = "", seed: int = 0) -> None:
    runs_vol.reload()
    wanted = arms.split() or sorted(TAG)
    _print_status(_arm_status(wanted, seed))
    used = shutil.disk_usage(OUT_MNT)
    print(f"\n/out volume: {used.used / 2**30:.1f} GiB written")


@app.function(volumes=VOLUMES, timeout=3600)
def report(metrics: str = "f1,iou,apls,cldice", store: str = "",
           stratum: str = "", by_stratum: bool = False, out: str = "",
           copy_to: str = "", aggregation: str = "micro") -> None:
    """Report over a store; optionally persist the markdown to the Volume.

    ``out`` is a path ON THE VOLUME (e.g. /out/benchmarks_x/report_all.md).
    ``copy_to`` is a space-separated list of directories the finished .md files
    are copied into — the runs/ folders, so the tables travel with the runs
    they describe.
    """
    runs_vol.reload()
    env = _base_env({})
    # Aggregation matters for comparability, not just presentation: `f1` and
    # `iou` are count-derivable so they default to MICRO (pooled tp/fp/fn),
    # while apls/cldice/buffered_* are not and always fall back to MACRO
    # (cli.py:578). Reading a micro f1 beside a macro buffered_f1 as though
    # they were the same quantity suggests the buffer LOWERS F1, which is
    # impossible. Pass aggregation="macro" for a table where every column is
    # the same kind of average.
    # Keep only the metrics this store actually carries. `report` REJECTS an
    # absent metric (cli._load_metric_table raises BadParameter) so a typo fails
    # loudly — correct for a human, fatal for a default list applied to stores
    # benched with and without --buffer-px. Also skips cleanly when the store
    # does not exist yet, e.g. after a --dry-run sweep that wrote nothing.
    wanted = [m.strip() for m in metrics.split(",") if m.strip()]
    store_dir = store or env["STORE_DIR"]
    try:
        # _base_env sets PYTHONPATH for the SUBPROCESS; this import runs in the
        # modal_app process, which has no repo on sys.path of its own.
        import sys
        if f"{REPO_REMOTE}/src" not in sys.path:
            sys.path.insert(0, f"{REPO_REMOTE}/src")
        from benchmarking.store import load_chips, load_tiles
        cols = set(load_chips(Path(store_dir)).columns)
        try:
            cols |= set(load_tiles(Path(store_dir)).columns)
        except Exception:
            pass
    except Exception as exc:
        print(f"[report] store {store_dir} unreadable ({type(exc).__name__}); "
              "nothing to report")
        return
    keep = [m for m in wanted if m in cols]
    drop = [m for m in wanted if m not in cols]
    if drop:
        print(f"[report] skipping absent metric(s): {', '.join(drop)}")
    if not keep:
        print(f"[report] none of {wanted} are columns of {store_dir}")
        return
    print(f"[report] metrics: {', '.join(keep)}")
    args = [a for m in keep for a in ("--metric", m)]
    args += ["--aggregation", aggregation]
    if stratum:
        args += ["--stratum", stratum]
    if by_stratum:
        args += ["--by-stratum"]
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        args += ["--out", out]
    subprocess.run(
        ["python", "-m", "benchmarking.cli", "report",
         "--store-dir", store_dir, *args],
        env=env, check=True,
    )
    if out:
        # --by-stratum fans `out` into one file per stratum, so glob the stem.
        produced = sorted(Path(out).parent.glob(Path(out).stem + "*.md"))
        for dest in (d for d in copy_to.split() if d):
            Path(dest).mkdir(parents=True, exist_ok=True)
            for md in produced:
                shutil.copy(md, Path(dest) / md.name)
                print(f"  report -> {Path(dest) / md.name}")
        runs_vol.commit()
        for md in produced:
            print(f"  wrote {md}")


# ---------------------------------------------------------------------------
# θ sweep (GPU). One inference pass per arm scores the whole θ grid — see
# benchmarking.runner.evaluate(sweep_thresholds=...) — so this is far cheaper
# than the arm's training run and fits comfortably in a short container.
#
# Writes to ${STORE_DIR}_theta, NOT STORE_DIR: the store is append-only and the
# existing shards are all at θ=0.5. Mixing the two protocols in one store would
# make every per-model mean average incomparable rows.
# ---------------------------------------------------------------------------
@app.function(
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY,
    volumes=VOLUMES,
    timeout=int(6 * 3600),
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
)
def sweep(overrides: dict[str, str]) -> None:
    """θ sweep + bench at θ* for every arm on the Volume, committing as it goes."""
    data_vol.reload()
    runs_vol.reload()
    env = _base_env(overrides)

    stop = threading.Event()

    def _committer() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            try:
                runs_vol.commit()
            except Exception as exc:
                print(f"[volume] commit failed (will retry): {exc}", flush=True)

    t = threading.Thread(target=_committer, daemon=True)
    t.start()

    extra = [a for a in str(overrides.get("SWEEP_ARGS", "")).split() if a]
    started = time.time()
    try:
        proc = subprocess.run(
            ["python", "-u", f"{REPO_REMOTE}/scripts/local/theta_sweep_bench.py",
             "--device", "cuda", *extra],
            env=env, check=False,
        )
    finally:
        stop.set()
        t.join(timeout=30)
        try:
            runs_vol.commit()
        except Exception as exc:
            print(f"[volume] FINAL commit failed: {exc}", flush=True)

    mins = (time.time() - started) / 60
    print(f"\n[modal] theta_sweep_bench.py exited {proc.returncode} after {mins:.1f} min",
          flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"theta_sweep_bench.py failed with exit code {proc.returncode}")


# ---------------------------------------------------------------------------
# θ* sweep + FINAL-protocol bench (GPU) for already-trained run dirs.
#
# Separate from sweep() above because the two obey different protocols. sweep()
# drives the LOSS PILOT: holdout arms (TRAIN_SPLITS=train), discovered by their
# `_holdout_seed<N>` dir names, benched on val. The runs this drives are
# merge-val FINAL runs — no `_holdout` tag, and val is training data for them,
# so the bench has to be on test. scripts/modal/final_bench.sh holds the arm
# table and does sweep-on-val -> bench-on-test -> report, per arm.
# ---------------------------------------------------------------------------
@app.function(
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY,
    volumes=VOLUMES,
    timeout=int(6 * 3600),
    # No retries, unlike pilot(). A retry pays for the whole container again,
    # and every failure this path can hit is deterministic (missing module,
    # missing weights, OOM on an arm) — the smoke run burned one proving it.
    # Resuming by hand is nearly free instead: sweep.json is reused and the
    # store's duplicate guard skips arms already benched, so re-invoking after
    # a fix picks up exactly where it stopped.
    retries=0,
)
def final_bench(overrides: dict[str, str]) -> None:
    """Run final_bench.sh inside the container, committing /out as it goes."""
    data_vol.reload()
    runs_vol.reload()
    env = _base_env(overrides)

    stop = threading.Event()

    def _committer() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            try:
                runs_vol.commit()
            except Exception as exc:
                print(f"[volume] commit failed (will retry): {exc}", flush=True)

    t = threading.Thread(target=_committer, daemon=True)
    t.start()

    started = time.time()
    try:
        proc = subprocess.run(
            ["bash", f"{REPO_REMOTE}/scripts/modal/final_bench.sh"],
            env=env, check=False,
        )
    finally:
        stop.set()
        t.join(timeout=30)
        try:
            runs_vol.commit()
        except Exception as exc:
            print(f"[volume] FINAL commit failed: {exc}", flush=True)

    mins = (time.time() - started) / 60
    print(f"\n[modal] final_bench.sh exited {proc.returncode} after {mins:.1f} min",
          flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"final_bench.sh failed with exit code {proc.returncode}")


# ---------------------------------------------------------------------------
# Whole-tile figures (GPU, minutes).
#
# The arm table is duplicated from DEFAULT_ARMS in final_bench.sh and must be
# edited alongside it — the same convention the TAG tables already follow
# across the four pilot ports. Fields: run dir, filename tag, SR weights subdir
# ('' = none, bicubic).
# ---------------------------------------------------------------------------
FINAL_ARMS_VIZ = [
    ("runs_sr_wbce/sr_r0_new_wbce_seed0", "r0_wbce", ""),
    ("runs_sr_wbce/sr_r2a_new_wbce_seed0", "r2a_wbce", "SEN2SRLite_RGBN"),
    ("runs_sr_wbce/sr_r2a_new_wbce_noreg_seed333", "r2a_wbce_noreg", "SEN2SRLite_RGBN"),
    ("runs_sr_wbce/sr_r4b_new_wbce_seed0", "r4b_wbce", "SR4RS_RGBN"),
    ("runs_gap_tl_ce/sr_r0_new_gap_tl_ce_seed0", "r0_gaptl", ""),
    ("runs_gap_tl_ce/sr_r2a_new_gap_tl_ce_seed0", "r2a_gaptl", "SEN2SRLite_RGBN"),
    ("runs_gap_tl_ce/sr_r2a_new_gap_tl_ce_noreg_seed333", "r2a_gaptl_noreg", "SEN2SRLite_RGBN"),
]


@app.function(gpu=DEFAULT_GPU, cpu=FINAL_BENCH_CPU, memory=FINAL_BENCH_MEMORY,
              volumes=VOLUMES, timeout=3600)
def viz_tiles(tiles: str, split: str = "val", runs_dir: str = "",
              out_dir: str = "", arms: str = "",
              select_on: str = "iou_mean,f1_mean,buffered_f1_mean",
              stretch_from_sr: bool = False, tag_suffix: str = "") -> None:
    """Render whole-tile SR + prediction PNGs for every arm x tile."""
    data_vol.reload()
    runs_vol.reload()
    env = _base_env({})
    runs = Path(runs_dir or OUT_MNT)
    out = Path(out_dir or f"{OUT_MNT}/figures")
    out.mkdir(parents=True, exist_ok=True)

    want = set(arms.split()) if arms else None
    tile_list = [t for t in tiles.split() if t]
    failed = []
    for run_name, tag, sr_sub in FINAL_ARMS_VIZ:
        if want and tag not in want and run_name not in want:
            continue
        ckpt = runs / run_name / "checkpoints" / "unet_s2rosa_jointsr_final.ckpt"
        if not ckpt.is_file():
            print(f"[{tag}] SKIP — no checkpoint at {ckpt}", flush=True)
            failed.append(f"{tag}: no ckpt")
            continue
        for tile in tile_list:
            print(f"\n=== {tag} x {tile} ===", flush=True)
            cmd = ["python", "-m", "sr.viz_tile",
                   "--ckpt", str(ckpt),
                   "--dataset-dir", env["DATASET_DIR"],
                   "--split", split,
                   "--tile", tile,
                   "--tag", tag,
                   "--out-dir", str(out),
                   "--select-on", select_on,
                   "--device", "cuda"]
            if stretch_from_sr:
                cmd.append("--stretch-from-sr")
            if tag_suffix:
                cmd[cmd.index("--tag") + 1] = tag + tag_suffix
            # r0 is bicubic and builds no SR net; the others need the weights
            # dir because their hparams point at the training node's /scratch.
            if sr_sub:
                cmd += ["--sr-dir", f"{DATA_MNT}/models/{sr_sub}"]
            rc = subprocess.run(cmd, env=env, check=False).returncode
            if rc != 0:
                print(f"[{tag} x {tile}] FAILED (rc={rc})", flush=True)
                failed.append(f"{tag} x {tile}")
    runs_vol.commit()
    print(f"\nfigures -> {out}")
    for f in sorted(p.name for p in out.glob("*.png")):
        print(f"  {f}")
    if failed:
        raise RuntimeError(f"viz failed for: {failed}")


# ---------------------------------------------------------------------------
# Phase B matrix (GPU, HOURS per arm — this one actually trains).
#
# Drives scripts/local/phase_b_matrix.py, which reads each parent's
# best_params.yaml and pins pos_weight / tl_theta / gap_theta into the child so
# the compound measures "what does adding a region term do to THIS attention
# arm" rather than a fresh joint search. Only lr and mix_w stay searched.
#
# The arms run on loss/l*_new.sh -> _pilot_new.sh -> sr/_stages_tv.sh, which is
# where SEARCH_THETAS / MIX_W_* / POS_WEIGHT_* are consumed — they reach it by
# ENVIRONMENT INHERITANCE through run.sh, not by any orchestrator forwarding
# them, which is why no TAG-table entry is required for the pinning to work.
# ---------------------------------------------------------------------------
@app.function(
    gpu=DEFAULT_GPU, cpu=DEFAULT_CPU, memory=DEFAULT_MEMORY, volumes=VOLUMES,
    secrets=[modal.Secret.from_name("wandb")],
    timeout=int(DEFAULT_TIMEOUT_H * 3600),
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
)
def phase_b(overrides: dict[str, str]) -> None:
    """Run phase_b_matrix.py --run inside the container, committing as it goes."""
    data_vol.reload()
    runs_vol.reload()
    env = _base_env(overrides)

    stop = threading.Event()

    def _committer() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            try:
                runs_vol.commit()
            except Exception as exc:
                print(f"[volume] commit failed (will retry): {exc}", flush=True)

    t = threading.Thread(target=_committer, daemon=True)
    t.start()
    started = time.time()
    try:
        args = [a for a in str(overrides.get("PHASE_B_ARGS", "")).split() if a]
        proc = subprocess.run(
            ["python", "-u", f"{REPO_REMOTE}/scripts/local/phase_b_matrix.py",
             "--runs-root", env["RUNS_ROOT"], "--run", *args],
            env=env, cwd=REPO_REMOTE, check=False,
        )
    finally:
        stop.set()
        t.join(timeout=30)
        try:
            runs_vol.commit()
        except Exception as exc:
            print(f"[volume] FINAL commit failed: {exc}", flush=True)

    print(f"\n[modal] phase_b_matrix.py exited {proc.returncode} after "
          f"{(time.time() - started) / 60:.1f} min", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"phase_b_matrix.py failed with {proc.returncode}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO_LOCAL), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO_LOCAL), "status", "--porcelain"], text=True
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def _rate(gpu: str, cpu: float, memory_mib: int) -> float:
    """$/hour for this container shape (GPU + CPU + RAM, as Modal bills it)."""
    g = GPU_RATE.get(gpu.upper(), GPU_RATE.get(gpu, 0.0))
    per_sec = g + cpu * CPU_RATE + (memory_mib / 1024) * MEM_RATE
    return per_sec * 3600


@app.local_entrypoint()
def main(
    action: str = "run",          # check | probe | run | sweep | final-bench | status | report
    arms: str = "l3_new l4a_new l4b_new",
    seed: int = 0,
    gpu: str = DEFAULT_GPU,
    cpu: float = DEFAULT_CPU,
    memory: int = DEFAULT_MEMORY,
    hours: float = DEFAULT_TIMEOUT_H,
    trials: int = 30,
    workers: int = 2,
    precision: str = "bf16-mixed",
    parallel: bool = False,
    tune_only: bool = False,
    wandb: bool = True,
    metrics: str = "f1,iou,apls,cldice,buffered_f1,buffered_precision,buffered_recall",
    stratum: str = "",
    by_stratum: bool = False,
    sweep_args: str = "",
    store: str = "",
    runs_dir: str = "",
    max_tiles: int = 0,
    refresh_sweep: bool = False,
    tiles: str = "",
    split: str = "val",
    select_on: str = "iou_mean",
    buffer_px: float = 0.0,
    out: str = "",
    copy_to: str = "",
    sweep_only: bool = False,
    aggregation: str = "micro",
    stretch_from_sr: bool = False,
    tag_suffix: str = "",
    phase_b_parents: str = "gap_tl_ce",
    phase_b_stage: str = "all",
    phase_b_no_bench: bool = False,
    phase_b_regions: str = "",
    sweep_lo: float = 0.0,
    sweep_hi: float = 0.0,
    sweep_step: float = 0.0,
) -> None:
    if action == "check":
        check_data.remote(seed=seed)
        return
    if action == "status":
        status.remote(arms=arms, seed=seed)
        return
    if action == "report":
        report.remote(metrics=metrics, store=store, stratum=stratum,
                      by_stratum=by_stratum, out=out, copy_to=copy_to,
                      aggregation=aggregation)
        return
    if action == "sweep":
        # θ* rows go to a store of their own — see the sweep() docstring.
        theta_store = f"{OUT_MNT}/benchmarks_loss_pilot_theta"
        rate = _rate(gpu, cpu, memory)
        print(f"θ sweep on {gpu} (~${rate:.3f}/h) -> {theta_store}")
        print(f"extra args: {sweep_args or '(none)'}\n")
        # .spawn(), not .remote(): a .remote() call is cancelled when its client
        # dies, even under --detach. That killed a sweep 38 min in on
        # 2026-08-14 because the launching shell exited. final_bench and
        # phase_b were already fixed; this path was the one left behind.
        call = sweep.with_options(
            gpu=gpu, cpu=cpu, memory=memory,
            timeout=int(hours * 3600), volumes=VOLUMES).spawn(
            {"SEED": seed, "INSTAROAD_GIT_SHA": _git_sha(),
             "THETA_STORE_DIR": theta_store, "SWEEP_ARGS": sweep_args}
        )
        print(f"spawned: {call.object_id}")
        print("  Safe to disconnect — the sweep survives this client dying.\n")
        call.get()
        report.remote(metrics=metrics, store=theta_store,
                      stratum=stratum, by_stratum=by_stratum)
        return
    if action == "phase-b":
        rate = _rate(gpu, cpu, memory)
        # Per-stage costs, measured on sr_r0_new_gap_t2_ce (L4): a Phase B trial
        # is ~16 min (8 epochs, pruned), a 50-epoch fit ~2.5 h, a bench ~0.4 h.
        # Quoting the "all" total for a tune-only run overstates it by 4x, which
        # is the difference between fitting in a $30 budget and not.
        TUNE_H, FIT_H, BENCH_H = trials * 0.27, 2.5, 0.4
        if phase_b_stage == "tune":
            est = TUNE_H
        elif phase_b_stage == "fit":
            est = FIT_H
        elif phase_b_stage == "bench":
            est = BENCH_H
        else:
            est = TUNE_H + FIT_H + BENCH_H
        reg_list = phase_b_regions.split() or ["pstar_dice", "pstar_sdice",
                                               "pstar_lcdice"]
        n = len(reg_list) * len(phase_b_parents.split())
        print(f"Phase B on {gpu}  cpu={cpu} mem={memory}MiB  ~${rate:.3f}/h")
        print(f"  parents : {phase_b_parents}")
        print(f"  regions : {phase_b_regions or '(all three)'}")
        if phase_b_no_bench and phase_b_stage == "all":
            est -= BENCH_H
            print("  stages  : tune -> fit ONLY (not swept, not benched)")
        else:
            print(f"  stage   : {phase_b_stage}")
        print(f"  trials  : {trials}   (only lr and mix_w are searched;")
        print(f"            pos_weight/tl_theta/gap_theta are pinned from the parent)")
        print(f"  estimate: ~{est:.1f} h/arm x {n} arm(s) = ~{est * n:.1f} GPU-h "
              f"~= ${est * n * rate:.0f}\n")
        env = {
            "INSTAROAD_GIT_SHA": _git_sha(),
            "RUNS_ROOT": runs_dir or f"{OUT_MNT}/runs",
            "WANDB_MODE": "online" if wandb else "offline",
            "PHASE_B_ARGS": (
                f"--parents {phase_b_parents} --trials {trials} --seed {seed}"
                + f" --stage {phase_b_stage}"
                + (f" --regions {phase_b_regions}"
                   if phase_b_regions and not parallel else "")
                + (" --no-bench" if phase_b_no_bench else "")
            ),
        }
        fn = phase_b.with_options(
            gpu=gpu, cpu=cpu, memory=memory, timeout=int(hours * 3600),
            volumes=VOLUMES,
            secrets=[modal.Secret.from_name("wandb")] if wandb else [],
        )
        # --parallel gives each REGION its own container, the same shape the
        # `run` action uses for arms. One container per region is the only way
        # three tunes finish in ~8 h instead of ~24; the COST is identical
        # because Modal bills container-seconds either way. Each region writes
        # to its own run dir, so the concurrent volume commits do not collide.
        if parallel and len(reg_list) > 1:
            calls = []
            for r in reg_list:
                e = dict(env)
                e["PHASE_B_ARGS"] = env["PHASE_B_ARGS"] + f" --regions {r}"
                calls.append((r, fn.spawn(e)))
            for r, c in calls:
                print(f"spawned [{r}]: {c.object_id}")
            print("  Safe to disconnect — training survives this client dying.\n")
            for r, c in calls:
                try:
                    c.get()
                except Exception as exc:
                    print(f"[{r}] FAILED: {exc}")
            return
        call = fn.spawn(env)
        print(f"spawned: {call.object_id}")
        print("  Safe to disconnect — training survives this client dying.\n")
        call.get()
        return
    if action == "viz":
        if not tiles:
            raise SystemExit(
                "--action viz needs --tiles '<stem> <stem>' (tile stems, no .tif)")
        # `select_on` is shared with --action final-bench, whose default is a
        # single criterion. For figures the point is the COMPARISON, so an
        # untouched flag means all three.
        if select_on == "iou_mean":
            select_on = "iou_mean,f1_mean,buffered_f1_mean"
        print(f"whole-tile figures on {gpu} -> /out/figures")
        print(f"  tiles: {tiles}")
        print(f"  split: {split}")
        print(f"  θ* on: {select_on}\n")
        viz_tiles.remote(tiles=tiles, split=split, select_on=select_on,
                         runs_dir=runs_dir, out_dir=store,
                         stretch_from_sr=stretch_from_sr, tag_suffix=tag_suffix,
                         arms=arms if arms != "l3_new l4a_new l4b_new" else "")
        return
    if action == "final-bench":
        # A store of its own. Every existing store holds either θ=0.5 rows or
        # holdout/val rows; these are θ* rows on test, and the store is
        # append-only, so mixing them would average incomparable protocols.
        bench_store = store or f"{OUT_MNT}/benchmarks_final_wbce"
        runs = runs_dir or OUT_MNT
        # Inference-shaped container unless you asked for something else.
        if cpu == DEFAULT_CPU:
            cpu = FINAL_BENCH_CPU
        if memory == DEFAULT_MEMORY:
            memory = FINAL_BENCH_MEMORY
        rate = _rate(gpu, cpu, memory)
        print(f"final bench on {gpu}  cpu={cpu} mem={memory}MiB  ~${rate:.3f}/h")
        print(f"           (gpu ${GPU_RATE.get(gpu.upper(), 0)*3600:.2f} "
              f"+ cpu ${cpu*CPU_RATE*3600:.2f} + ram ${memory/1024*MEM_RATE*3600:.2f})")
        print(f"  runs  : {runs}")
        print(f"  store : {bench_store}")
        print(f"  arms  : {arms if arms != 'l3_new l4a_new l4b_new' else '(all in final_bench.sh)'}")
        print(f"  select: {select_on}"
              + (f"   buffer_px={buffer_px}" if buffer_px else ""))
        print("  report: in-container -> store + each contributing runs/ folder, "
              f"as report_{select_on}_*.md")
        if max_tiles:
            print(f"  MAX_TILES={max_tiles} — SMOKE TEST, the rows are not a result\n")
        env: dict[str, str] = {
            "INSTAROAD_GIT_SHA": _git_sha(),
            "RUNS_ROOT": runs,
            "STORE_DIR": bench_store,
            "METRICS": metrics,
            "SELECT_ON": select_on,
            # Reports run in-container now. They cost a few minutes of idle GPU,
            # and buy a .md that is on the Volume whether or not this client
            # survives — see the REPORT block in final_bench.sh.
            "REPORT": "1",
        }
        if buffer_px:
            env["BUFFER_PX"] = str(buffer_px)
        # The default arm string belongs to the loss pilot; only forward --arms
        # when it was actually overridden, otherwise final_bench.sh would
        # reject "l3_new" as an unknown run dir.
        if arms and arms != "l3_new l4a_new l4b_new":
            env["ARMS"] = arms
        if max_tiles:
            env["MAX_TILES"] = str(max_tiles)
        if refresh_sweep:
            env["REFRESH_SWEEP"] = "1"
        if sweep_only:
            env["SWEEP_ONLY"] = "1"
        # The θ grid has to be forwarded EXPLICITLY. _base_env copies the
        # CONTAINER's os.environ, not the laptop's, so `SWEEP_LO=... modal run`
        # silently does nothing — the sweep would run the default grid and look
        # like it worked. Widen the grid when an arm's θ* lands on an endpoint.
        for name, val in (("SWEEP_LO", sweep_lo), ("SWEEP_HI", sweep_hi),
                          ("SWEEP_STEP", sweep_step)):
            if val:
                env[name] = str(val)
        # .spawn(), NOT .remote() — the same rule the pilot path follows below,
        # for a second reason that cost a run on 2026-08-11: `modal run
        # --detach` detaches the APP, but a .remote() call stays bound to this
        # client, and Modal cancels the input when the client goes away
        # ("Received a cancellation signal"). Closing a laptop mid-bench killed
        # the container despite --detach. .spawn() hands the call to Modal and
        # lets it outlive the client entirely.
        call = final_bench.with_options(
            gpu=gpu, cpu=cpu, memory=memory,
            timeout=int(hours * 3600), volumes=VOLUMES).spawn(env)
        print(f"spawned: {call.object_id}\n"
              "  Safe to disconnect — the bench now survives this client dying.\n"
              "  Re-attach or read results with:\n"
              f"    modal app logs {APP_NAME}\n"
              f"    modal run scripts/modal/modal_app.py --action report "
              f"--store {bench_store} --by-stratum\n")
        call.get()
        return
    if action not in ("run", "probe"):
        raise SystemExit(
            f"unknown --action {action!r} "
            "(check|probe|run|sweep|final-bench|status|report)")

    arm_list = arms.split()
    unknown = [a for a in arm_list if a not in TAG]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown} — extend the TAG table in all four ports")

    sha = _git_sha()
    if sha.endswith("-dirty"):
        print("WARNING: working tree is dirty. The mounted code is whatever is on disk\n"
              "         right now, so this run is not reproducible from a commit.\n")

    rate = _rate(gpu, cpu, memory)
    n = len(arm_list)
    print(f"app      : {APP_NAME}")
    print(f"repo     : {REPO_LOCAL}  @ {sha}")
    print(f"container: gpu={gpu} cpu={cpu} mem={memory}MiB timeout={hours}h")
    print(f"rate     : ${rate:.3f}/h  (gpu ${GPU_RATE.get(gpu.upper(), 0)*3600:.2f} "
          f"+ cpu ${cpu*CPU_RATE*3600:.2f} + ram ${memory/1024*MEM_RATE*3600:.2f})")
    print(f"           ${FREE_CREDIT:.0f} credit ~= {FREE_CREDIT/rate:.1f} container-hours"
          + (f", i.e. ~{FREE_CREDIT/rate/n:.1f} h per arm across {n} arms" if n else ""))
    print(f"arms     : {' '.join(arm_list)}  (seed {seed})")
    print(f"mode     : {'PARALLEL — ' + str(n) + ' containers at once' if parallel else 'sequential — 1 container'}")
    print(f"stages   : {'tune only (fit/bench deferred)' if tune_only else 'tune -> fit -> bench'}")
    print(f"trials   : {trials} per arm")
    print()

    overrides = {
        "SEED": seed,
        "TARGET_TRIALS": trials,
        "NUM_WORKERS": workers,
        "PRECISION": precision,
        "WANDB_MODE": "online" if wandb else "offline",
        "INSTAROAD_GIT_SHA": sha,
        # Stop the bash orchestrator cleanly before Modal's timeout, so the
        # final Volume commit always happens.
        "MAX_SECONDS": max(600, int(hours * 3600) - BUDGET_MARGIN_S),
    }
    if tune_only:
        # Stop each container after its Optuna search. The fit is the expensive
        # half; --tune-only buys a look at best_params.yaml before committing to
        # it. Re-invoke the same command without the flag to resume at fit.
        overrides["TUNE_ONLY"] = "1"
    if action == "probe":
        overrides["PROBE"] = "1"
        hours = min(hours, 1.0)

    fn = pilot.with_options(
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        timeout=int(hours * 3600),
        volumes=VOLUMES,
        secrets=[modal.Secret.from_name("wandb")] if wandb else [],
    )

    # .spawn(...).get() rather than .remote(): Modal expires .remote calls after
    # 24 h, and these outlive that.
    if parallel and action == "run":
        calls = [fn.spawn({**overrides, "ARMS": a}) for a in arm_list]
        rows: list[dict] = []
        for call, arm in zip(calls, arm_list):
            try:
                rows += call.get() or []
            except Exception as exc:
                print(f"[{arm}] FAILED: {exc}")
        if rows:
            print()
            _print_status(rows)
    else:
        fn.spawn({**overrides, "ARMS": " ".join(arm_list)}).get()
