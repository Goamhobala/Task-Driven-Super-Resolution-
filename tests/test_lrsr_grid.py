"""Unit tests for the lr_sr x hard-constraint mechanism grid (docs/lrsr_grid_ablation_plan.md).

The grid is a 2x4 ablation on the SEN2SR row — HC on/off x PINNED lr_sr — and
its two load-bearing pieces of machinery are both plumbing:

  * §3, §7.1  the std-band rails reach BOTH stages. The tune runs the model
        with ``std_band_action="raise"``, so the band decides which trials can
        be scored at all: at pinned lr_sr=1e-4 in the bare lane the production
        band (0.5x-4x) would prune every trial, no ``best_params.yaml`` would
        be written, and the GUARD rather than the design would decide which
        cells exist. The refit must then run the same band the search ran, or a
        cell searched under loosened rails is refitted under the band it exists
        to leave.
  * §7.2  the cell coordinates (HC, lr_sr) derive an EXP_TAG that keeps run
        dirs, Optuna studies and benchmark rows disjoint from the formal
        r2a/r2b arms — and from each other — BY CONSTRUCTION. The store is
        append-only, so a tag collision is unrecoverable, not merely untidy.

Everything here runs on CPU with the parameter-free ``bicubic`` upsampler or on
a stubbed engine: no SR weights, no dataset, no network, no GPU.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from sr.model import (
    _STD_RAISE_HI, _STD_RAISE_LO, AdaptiveNormBandExit, JointSRUNetLightning)

REPO = Path(__file__).resolve().parents[1]
GRID_SCRIPT = REPO / "scripts" / "hpc" / "sr" / "r2grid_new.sh"
TRAIN_SBATCH = REPO / "scripts" / "hpc" / "train.sbatch"

NORM_MEAN = [0.10, 0.12, 0.14, 0.20]
NORM_STD = [0.04, 0.05, 0.06, 0.07]
BANDS = (1, 2, 3, 4)
C, P, UP = 4, 16, 4
HR = P * UP


# --------------------------------------------------------------------------
# §3 — the rails, at the model level: what the tune stage prunes on
# --------------------------------------------------------------------------
def make_model(**kw):
    """Bicubic-front-end JointSR model on CPU (as in tests/test_adaptive_norm)."""
    kwargs = dict(
        encoder_name="resnet18",
        encoder_weights=None,           # never hits the network
        classes=1,
        bands=BANDS,
        in_channels=C,
        norm_mean=NORM_MEAN,
        norm_std=NORM_STD,
        upsampler="bicubic",
        upscale=UP,
        reflectance_scale=1.0,
        image_size=HR,
        adaptive_norm=True,
        adaptive_norm_momentum=1.0,     # one batch = the whole EMA
        adaptive_norm_check_every=1,
    )
    kwargs.update(kw)
    return JointSRUNetLightning(**kwargs)


def hr_tensor(mean: float, std: float, seed: int):
    import torch
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, C, HR, HR, generator=g) * std + mean


def test_loosened_rails_stop_the_tune_from_pruning_the_cell():
    """The grid's whole point is to RUN the corner the guard exists to kill.

    Same model, same collapse, same ``std_band_action="raise"`` the tune uses —
    only the rails differ. Under the production band this trial is pruned (and
    with lr_sr pinned, so is every other trial of the cell, leaving the study
    with nothing to write); under the grid's rails it survives to be scored.
    """
    assert (_STD_RAISE_LO, _STD_RAISE_HI) == (0.5, 4.0), (
        "the production band moved; the grid's rails are described relative to "
        "it in docs/lrsr_grid_ablation_plan.md §3")
    collapsed = hr_tensor(mean=0.30, std=0.002, seed=7)    # 0.05x band 0's 0.04

    production = make_model(std_band_action="raise")
    with pytest.raises(AdaptiveNormBandExit, match="COLLAPSED"):
        production._adapt_update(collapsed)

    grid = make_model(std_band_action="raise",
                      std_band_raise_lo=0.01, std_band_raise_hi=100)
    grid._adapt_update(collapsed)                          # must NOT raise


def test_loosened_rails_keep_the_diagnostics_the_grid_is_for(capsys):
    """Loosened, NOT disabled (§3): the warn stream and the variance-floor
    diagnostic keep printing, and adapt_band_exit still latches on the warn
    bands. This is what separates the sanctioned protocol from the tempting
    shortcut, ``adaptive_norm_check_every=0``, which silences both."""
    grid = make_model(std_band_raise_lo=0.01, std_band_raise_hi=100)
    grid._adapt_update(hr_tensor(mean=0.30, std=0.004, seed=8))   # 0.1x
    out = capsys.readouterr().out
    assert "WARN adaptive_norm" in out
    # ...and the run is still alive to be measured post hoc.
    assert grid._adapt_band_exited is False   # inside the loosened hard band
    assert float(grid.band_std.reshape(-1)[0]) == pytest.approx(0.004, rel=1e-2)


def test_rails_are_a_band_not_a_pair_of_numbers():
    """A crossed or non-positive band would trip every check on step one."""
    from sr import tune as sr_tune
    args = sr_tune.parse_args([
        "--base-config", "x.yaml", "--std-band-raise-lo", "5",
        "--std-band-raise-hi", "1"])
    with pytest.raises(SystemExit, match="not a band"):
        sr_tune.build_objective(args, _base_cfg())


# --------------------------------------------------------------------------
# §7.1 — the rails reach the tune's model, and the overlay carries them on
# --------------------------------------------------------------------------
def _base_cfg(**model_extra) -> dict:
    cfg = {
        "data": {
            "dataset_dir": "/nonexistent",
            "norm_mean": NORM_MEAN,
            "norm_std": NORM_STD,
            "bands": list(BANDS),
            "batch_size": 4,
            "crop_size": 128,
            "image_size": 512,
            "upscale": 4,
        },
        "model": {"upsampler": "bicubic", "classes": 1},
    }
    cfg["model"].update(model_extra)
    return cfg


class _Sentinel(Exception):
    """Aborts the objective at the model construction we want to inspect."""


def _model_kwargs_from_a_trial(monkeypatch, argv, base_cfg) -> dict:
    """Run one objective up to model construction and return its kwargs."""
    import optuna

    from sr import tune as sr_tune

    captured: dict = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        raise _Sentinel

    monkeypatch.setattr(sr_tune, "JointSRUNetLightning", _capture)
    monkeypatch.setattr(sr_tune, "JointSRDataModule", lambda **kw: object())

    args = sr_tune.parse_args(["--base-config", "x.yaml", *argv])
    objective = sr_tune.build_objective(args, base_cfg)
    trial = optuna.trial.FixedTrial({
        "lr": 1e-4, "lr_sr": 1e-5, "pos_weight": 5.0,
        "encoder_name": "resnet34", "batch_size": 4,
    })
    with pytest.raises(_Sentinel):
        objective(trial)
    return captured


def test_the_tune_passes_the_forced_rails_into_every_trial(monkeypatch):
    kwargs = _model_kwargs_from_a_trial(
        monkeypatch,
        ["--std-band-raise-lo", "0.01", "--std-band-raise-hi", "100"],
        _base_cfg())
    assert kwargs["std_band_raise_lo"] == 0.01
    assert kwargs["std_band_raise_hi"] == 100.0


def test_the_tunes_pruning_action_is_unchanged_unless_asked(monkeypatch):
    """`raise` stays the search's default for every arm — including one that
    only loosens the rails — so no existing tune changes behaviour. The base
    config's `warn` (which is the FIT-stage setting) must not leak in here."""
    kwargs = _model_kwargs_from_a_trial(
        monkeypatch, ["--std-band-raise-lo", "0.01", "--std-band-raise-hi", "100"],
        _base_cfg(std_band_action="warn"))
    assert kwargs["std_band_action"] == "raise"


def test_the_grid_can_ask_for_a_collapsing_trial_to_be_SCORED_not_pruned(monkeypatch):
    """With lr_sr pinned, pruning is the wrong response twice over: the trial's
    lr vanishes from the ranking, and if every trial of a cell prunes, the study
    writes no best_params at all — the guard deciding which cells exist, which
    is what §3 exists to prevent. MedianPruner still prunes on the objective."""
    kwargs = _model_kwargs_from_a_trial(
        monkeypatch, ["--std-band-action", "warn"], _base_cfg())
    assert kwargs["std_band_action"] == "warn"


def test_the_base_configs_rails_are_honoured_when_no_flag_is_given(monkeypatch):
    """The ``adapt_rest`` problem: a config setting that reaches the refit but
    not the trials means the search scores a different adapter than it ships."""
    kwargs = _model_kwargs_from_a_trial(
        monkeypatch, [],
        _base_cfg(std_band_raise_lo=0.25, std_band_raise_hi=8.0))
    assert kwargs["std_band_raise_lo"] == 0.25
    assert kwargs["std_band_raise_hi"] == 8.0


def test_a_flag_beats_the_config(monkeypatch):
    kwargs = _model_kwargs_from_a_trial(
        monkeypatch, ["--std-band-raise-hi", "100"],
        _base_cfg(std_band_raise_lo=0.25, std_band_raise_hi=8.0))
    assert kwargs["std_band_raise_lo"] == 0.25     # untouched by the flag
    assert kwargs["std_band_raise_hi"] == 100.0


def test_arms_that_force_nothing_are_unchanged(monkeypatch):
    """Every arm already in the append-only store must keep its exact recipe:
    with no flag and no config key, the model's own defaults apply and no rail
    kwarg is passed at all."""
    kwargs = _model_kwargs_from_a_trial(monkeypatch, [], _base_cfg())
    assert "std_band_raise_lo" not in kwargs
    assert "std_band_raise_hi" not in kwargs


def _fake_study():
    """The three attributes write_best_overlay reads, and nothing else."""
    class _Study:
        best_params = {"lr": 1e-4, "lr_sr": 1e-5, "batch_size": 4,
                       "encoder_name": "resnet34"}
        best_value = 0.5
        study_name = "test"
        trials: list = []

        class best_trial:                      # noqa: N801 - stands in for optuna's
            number = 0
    return _Study()


def test_the_overlay_pins_forced_rails_so_the_refit_runs_the_searched_band(tmp_path):
    from sr import tune as sr_tune

    path = sr_tune.write_best_overlay(
        _fake_study(), tmp_path, "imagenet", "sen2sr",
        std_band_raise_lo=0.01, std_band_raise_hi=100)
    model = yaml.safe_load(path.read_text())["model"]
    assert model["std_band_raise_lo"] == 0.01
    assert model["std_band_raise_hi"] == 100.0


def test_the_overlay_is_byte_identical_for_arms_that_force_nothing(tmp_path):
    """Written only when FORCED — never when inherited from the base config —
    so every overlay already on disk still resolves to the recipe it ran."""
    from sr import tune as sr_tune

    path = sr_tune.write_best_overlay(
        _fake_study(), tmp_path, "imagenet", "sen2sr")
    model = yaml.safe_load(path.read_text())["model"]
    assert "std_band_raise_lo" not in model
    assert "std_band_raise_hi" not in model


# --------------------------------------------------------------------------
# §7.2 — the cell coordinates derive a disjoint EXP_TAG and the right lane
# --------------------------------------------------------------------------
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _stub_engine(tmp_path: Path) -> Path:
    """Plant a STUB ``_stages_tv.sh`` beside a copy of the grid script.

    The real engine needs /scratch, a venv and a GPU; what is under test here is
    the 20 lines of derivation above the `source`, so the engine is replaced by
    an echo of the variables the cell is supposed to have set. Returns the copied
    grid script; ``tmp_path`` doubles as REPO_DIR.
    """
    stub_dir = tmp_path / "scripts" / "hpc" / "sr"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "_stages_tv.sh").write_text(
        "for k in STAGE EXP_TAG SR_HC SR_PAD UPSAMPLER FREEZE_SR LABELS \\\n"
        "  SR_SNAPSHOT_EVERY LR_SR_MIN LR_SR_MAX N_TRIALS TUNE_EPOCHS PATIENCE \\\n"
        "  BATCH_SIZES REFIT_EPOCHS STD_BAND_RAISE_LO STD_BAND_RAISE_HI \\\n"
        "  STD_BAND_ACTION \\\n"
        "  LOSS_ARM PSTAR GAP_THETA POS_WEIGHT_MIN POS_WEIGHT_MAX \\\n"
        "  SEARCH_THETAS SEARCH_MIX_W TRAIN_SPLITS MONITOR SWEEP_CRITERION; do\n"
        # ${!k-}, not ${!k}: STAGE is the engine's own default, unset when the
        # cell script is run directly, and `set -u` would abort on it.
        "  printf '%s=%s\\n' \"$k\" \"${!k-}\"\n"
        "done\n")
    shutil.copy(GRID_SCRIPT, stub_dir / GRID_SCRIPT.name)
    return stub_dir / GRID_SCRIPT.name


def _read_back(proc: subprocess.CompletedProcess) -> dict[str, str]:
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines()
                if "=" in line and not line.startswith("==="))


def _run_cell(tmp_path: Path, hc: str, lrsr: str) -> dict[str, str]:
    """Run the grid script itself against the stub engine, as one cell."""
    script = _stub_engine(tmp_path)
    return _read_back(subprocess.run(
        ["bash", str(script)],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "REPO_DIR": str(tmp_path), "HC": hc, "LRSR": lrsr},
        capture_output=True, text=True))


@needs_bash
@pytest.mark.parametrize("lrsr,tag", [
    ("1e-4", "1e-4"), ("1E-04", "1e-4"), ("0.0001", "1e-4"),
    ("1e-7", "1e-7"), ("0.0000001", "1e-7"), ("1.4e-5", "1.4e-5"),
])
def test_the_cell_tag_normalises_the_pinned_lr_sr(tmp_path, lrsr, tag):
    """Three spellings of one decade must name ONE cell (or two cells share a
    run dir and an append-only study), and a genuinely different rate must name
    a different one."""
    assert _run_cell(tmp_path, "on", lrsr)["EXP_TAG"] == f"r2grid_on_ls{tag}"


@needs_bash
def test_the_lanes_are_the_formal_arms_configurations(tmp_path):
    """HC on = r2a's lane (native constraint, pad 8); HC off = r2b's (bare
    generator, pad 0). The pad travels with the constraint: no FFT splice means
    no Gibbs ringing at the patch border for a pad to mitigate."""
    on = _run_cell(tmp_path, "on", "1e-5")
    off = _run_cell(tmp_path, "off", "1e-5")
    assert (on["SR_HC"], on["SR_PAD"]) == ("native", "8")
    assert (off["SR_HC"], off["SR_PAD"]) == ("off", "0")
    assert on["EXP_TAG"] != off["EXP_TAG"]
    for cell in (on, off):
        assert cell["UPSAMPLER"] == "sen2sr" and cell["FREEZE_SR"] == "false"


@needs_bash
def test_lr_sr_is_pinned_not_searched(tmp_path):
    """min == max is what makes Optuna suggest a CONSTANT that still lands in
    best_params.yaml — the reason the fit stage needs no special-casing."""
    cell = _run_cell(tmp_path, "off", "1e-4")
    assert cell["LR_SR_MIN"] == cell["LR_SR_MAX"] == "1e-4"
    assert cell["POS_WEIGHT_MIN"] == cell["POS_WEIGHT_MAX"]   # same pattern for λ*
    assert (cell["N_TRIALS"], cell["TUNE_EPOCHS"], cell["PATIENCE"]) == ("15", "10", "5")
    assert cell["BATCH_SIZES"] == "4" and cell["REFIT_EPOCHS"] == "100"


@needs_bash
def test_every_cell_carries_the_same_loosened_rails_and_snapshots_each_epoch(tmp_path):
    """§3's pair rule, and §7.3: a cell that dies at step ~2k must still leave
    its snapshot sequence — the strips ARE the mechanism figure."""
    cells = [_run_cell(tmp_path, hc, ls)
             for hc in ("on", "off") for ls in ("1e-4", "1e-7")]
    assert {(c["STD_BAND_RAISE_LO"], c["STD_BAND_RAISE_HI"]) for c in cells} == {
        ("0.01", "100")}
    assert {c["SR_SNAPSHOT_EVERY"] for c in cells} == {"1"}


@needs_bash
def test_no_cell_can_be_ended_by_a_band_exit_at_either_stage(tmp_path):
    """The rails make an exit unlikely; the action is what makes it harmless.
    A cell that truly collapses past 0.01x must still be scored (tune) and must
    still finish its 100 epochs (fit) — a numerical death at step X is an
    observation to record, not a run to bin."""
    cells = [_run_cell(tmp_path, hc, ls)
             for hc in ("on", "off") for ls in ("1e-4", "1e-7")]
    assert {c["STD_BAND_ACTION"] for c in cells} == {"warn"}


@needs_bash
def test_the_loss_is_a_frozen_control_across_the_grid(tmp_path):
    """Re-searching the loss per cell would confound lr_sr with a different
    loss surface per cell (the R-series rule)."""
    cells = [_run_cell(tmp_path, hc, ls)
             for hc in ("on", "off") for ls in ("1e-4", "1e-5")]
    assert {(c["LOSS_ARM"], c["PSTAR"], c["GAP_THETA"]) for c in cells} == {
        ("gap_ce", "gap_ce", "0.55836")}
    assert {c["SEARCH_THETAS"] for c in cells} == {"false"}
    assert {c["SEARCH_MIX_W"] for c in cells} == {"false"}


@needs_bash
def test_the_grid_refuses_a_cell_it_cannot_name(tmp_path):
    stub = tmp_path / "scripts" / "hpc" / "sr"
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "_stages_tv.sh").write_text("echo ENGINE RAN\n")
    shutil.copy(GRID_SCRIPT, stub / GRID_SCRIPT.name)
    base = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
            "REPO_DIR": str(tmp_path)}
    for env, why in (({"HC": "maybe", "LRSR": "1e-4"}, "HC must be on|off"),
                     ({"HC": "on", "LRSR": "abc"}, "positive number"),
                     ({"HC": "on", "LRSR": "-1e-4"}, "positive number"),
                     ({"HC": "on"}, "LRSR")):
        proc = subprocess.run(["bash", str(stub / GRID_SCRIPT.name)],
                              env={**base, **env}, capture_output=True, text=True)
        assert proc.returncode != 0, f"{env} should not have run"
        assert "ENGINE RAN" not in proc.stdout
        assert why in proc.stderr


@needs_bash
def test_train_sbatch_carries_the_cell_coordinates_into_the_script(tmp_path):
    """The launch path: ``sbatch scripts/hpc/train.sbatch --SCRIPT=sr/r2grid_new.sh
    STAGE=... HC=... LRSR=...``. All eight cells share ONE script, so the
    dispatcher's KEY=VALUE forwarding is what distinguishes them — and a
    coordinate that fails to arrive does not fail loudly, it runs the wrong
    cell (or, for LRSR, refuses). `--SCRIPT` itself must NOT leak through as an
    env var or a positional argument.
    """
    _stub_engine(tmp_path)                      # tmp_path is REPO_DIR
    cell = _read_back(subprocess.run(
        ["bash", str(TRAIN_SBATCH), "--SCRIPT=sr/r2grid_new.sh",
         "STAGE=fit", "HC=off", "LRSR=1e-4"],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "REPO_DIR": str(tmp_path)},
        capture_output=True, text=True))
    assert cell["STAGE"] == "fit"
    assert cell["EXP_TAG"] == "r2grid_off_ls1e-4"
    assert (cell["SR_HC"], cell["SR_PAD"]) == ("off", "0")
    assert cell["LR_SR_MIN"] == cell["LR_SR_MAX"] == "1e-4"


@needs_bash
def test_train_sbatch_refuses_a_cell_whose_script_it_cannot_find(tmp_path):
    """A typo'd --SCRIPT must not fall through to some other arm's defaults."""
    (tmp_path / "scripts" / "hpc").mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["bash", str(TRAIN_SBATCH), "--SCRIPT=sr/r2grid_typo.sh", "HC=on", "LRSR=1e-5"],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "REPO_DIR": str(tmp_path)},
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "inner script not found" in proc.stderr


# --------------------------------------------------------------------------
# The lane pools: one SLURM job per HC lane, all four cells, resume-guarded
# --------------------------------------------------------------------------
# A pool runs unattended for 48 h and is MEANT to be re-submitted after a
# timeout, so its whole value is the guards: skip what is finished, resume what
# is partial, and never silently redo a 100-epoch refit. Those guards read the
# cell's run dir, which is why the engine is ASKED for the name (PRINT_RUN_DIR)
# rather than the pool rebuilding the tag chain and inspecting a directory the
# run will never write.
POOL_DIR = REPO / "scripts" / "hpc" / "sr" / "grid"
RUN_POOL = POOL_DIR / "run_pool.sh"

STUB_CELL = """#!/bin/bash
set -euo pipefail
RUN_DIR="${RUNS_ROOT}/cell_${HC}_${LRSR}_seed${SEED}"
if [ "${PRINT_RUN_DIR:-0}" = "1" ]; then
  echo "RUN_DIR=${RUN_DIR}"
  echo "MODEL_NAME=model_${HC}_${LRSR}"
  exit 0
fi
echo "${STAGE} ${HC} ${LRSR} resume=${RESUME_FIT:-none}" >> "$TRACE"
[ "${FAIL_AT:-}" = "$LRSR" ] && exit 9
mkdir -p "$RUN_DIR/checkpoints"
case "$STAGE" in
  tune) echo 'model: {}' > "$RUN_DIR/best_params.yaml" ;;
  fit)  echo '{}' > "$RUN_DIR/sweep.json" ;;
esac
exit 0
"""


def _pool_repo(tmp_path: Path) -> Path:
    """A minimal REPO_DIR: the real _refit_lib.sh and src (the guards import
    torch and benchmarking.store for real), with a STUB cell script."""
    (tmp_path / "scripts" / "hpc" / "sr" / "grid").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "hpc" / "sr" / "refit").mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN_POOL, tmp_path / "scripts" / "hpc" / "sr" / "grid" / "run_pool.sh")
    shutil.copy(REPO / "scripts" / "hpc" / "sr" / "refit" / "_refit_lib.sh",
                tmp_path / "scripts" / "hpc" / "sr" / "refit" / "_refit_lib.sh")
    (tmp_path / "scripts" / "hpc" / "sr" / "r2grid_new.sh").write_text(STUB_CELL)
    if not (tmp_path / "src").exists():
        (tmp_path / "src").symlink_to(REPO / "src")
    return tmp_path


def _run_pool(tmp_path: Path, hc: str = "off", **env) -> tuple[str, list[str]]:
    """Drive the pool over the stub cells; return (stdout, the stage trace)."""
    import os
    repo = _pool_repo(tmp_path)
    trace = tmp_path / "trace"
    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "hpc" / "sr" / "grid" / "run_pool.sh")],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path),
             "REPO_DIR": str(repo), "RUNS_ROOT": str(tmp_path / "runs"),
             "STORE_DIR": str(tmp_path / "store"), "TRACE": str(trace),
             "VENV_DIR": str(tmp_path / "novenv"),   # falls through to PATH python
             "HC": hc, "LRSRS": "1e-4 1e-5", **env},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = trace.read_text().splitlines() if trace.exists() else []
    return proc.stdout, lines


needs_torch = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("torch") is None,
    reason="the pool's resume guards import torch for real")


@needs_bash
@needs_torch
def test_the_pool_runs_every_cell_through_the_whole_chain_in_order(tmp_path):
    """Extremes first (§8): the interaction and the probable collapses come
    before the cells that merely interpolate toward the formal arms."""
    _, trace = _run_pool(tmp_path)
    assert trace == [
        "tune off 1e-4 resume=none",
        "fit off 1e-4 resume=0",
        "bench off 1e-4 resume=none",
        "tune off 1e-5 resume=none",
        "fit off 1e-5 resume=0",
        "bench off 1e-5 resume=none",
    ]


@needs_bash
@needs_torch
def test_a_finished_tune_is_never_re_entered(tmp_path):
    """sr.tune writes best_params.yaml once, after the study finishes — but
    re-entering a FINISHED study would add another N_TRIALS to it. A cell that
    straddled a timeout would then carry a bigger search budget than its
    neighbours, which is not a grid any more."""
    run = tmp_path / "runs" / "cell_off_1e-4_seed0"
    run.mkdir(parents=True)
    (run / "best_params.yaml").write_text("model: {}\n")
    _, trace = _run_pool(tmp_path, LRSRS="1e-4")
    assert "tune off 1e-4 resume=none" not in trace
    assert trace[0].startswith("fit off 1e-4")


@needs_bash
@needs_torch
def test_a_partial_fit_resumes_instead_of_restarting(tmp_path):
    """The 48 h wall clock will cut a refit in half. last.ckpt is the only
    checkpoint carrying optimizer state, so its presence is what says 'resume'."""
    run = tmp_path / "runs" / "cell_off_1e-4_seed0" / "checkpoints"
    run.mkdir(parents=True)
    (run.parent / "best_params.yaml").write_text("model: {}\n")
    (run / "last.ckpt").write_bytes(b"")
    _, trace = _run_pool(tmp_path, LRSRS="1e-4")
    assert trace == ["fit off 1e-4 resume=1", "bench off 1e-4 resume=none"]


@needs_bash
@needs_torch
def test_a_completed_refit_is_not_retrained(tmp_path):
    """THE expensive mistake. The final ckpt is rewritten EVERY EPOCH, so its
    existence means 'a fit started' — only its recorded epoch says finished."""
    import torch
    run = tmp_path / "runs" / "cell_off_1e-4_seed0"
    (run / "checkpoints").mkdir(parents=True)
    (run / "best_params.yaml").write_text("model: {}\n")
    (run / "sweep.json").write_text("{}")
    torch.save({"epoch": 100}, run / "checkpoints" / "unet_s2rosa_jointsr_final.ckpt")
    _, trace = _run_pool(tmp_path, LRSRS="1e-4")
    assert trace == ["bench off 1e-4 resume=none"], "a finished fit was redone"

    # ...and one epoch short is NOT finished (the seed42 incident: a "final"
    # ckpt from epoch 14 of 100 whose test row was computed on it anyway).
    torch.save({"epoch": 99}, run / "checkpoints" / "unet_s2rosa_jointsr_final.ckpt")
    (run / "checkpoints" / "last.ckpt").write_bytes(b"")
    _, trace2 = _run_pool(tmp_path, LRSRS="1e-4")
    assert trace2[len(trace)] == "fit off 1e-4 resume=1", (
        "an epoch short of the budget must resume, not count as done")


@needs_bash
@needs_torch
def test_one_dead_cell_does_not_take_the_lane_with_it(tmp_path):
    """Numerical death at lr_sr=1e-4 is an EXPECTED outcome (§3) and a valid
    observation. It must not cost the other three cells their 48 h job."""
    out, trace = _run_pool(tmp_path, FAIL_AT="1e-4")
    assert [t for t in trace if t.startswith("tune off 1e-5")], "lane stopped early"
    assert "1e-4: TUNE FAILED" in out
    assert "1e-5: OK" in out


@needs_bash
@needs_torch
def test_the_guards_fail_closed_when_python_cannot_import(tmp_path):
    """ckpt_epoch returns -1 on ANY exception and in_store returns 'absent', so
    a broken venv would make every finished cell look unfinished — re-tuning,
    re-fitting and appending duplicate rows to an append-only store. Refuse."""
    repo = _pool_repo(tmp_path)
    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "hpc" / "sr" / "grid" / "run_pool.sh")],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "REPO_DIR": str(repo),
             "VENV_DIR": str(tmp_path / "novenv"), "HC": "off", "LRSRS": "1e-4"},
        capture_output=True, text=True)
    assert proc.returncode == 2
    assert "fail OPEN" in proc.stderr


@needs_bash
def test_print_run_dir_names_the_cell_without_running_or_creating_anything(tmp_path):
    """The pool's guards inspect this path, so it must be the engine's own —
    and the query must stay free of side effects (no run dir, no norm stats, no
    venv), since it is called once per stage of every cell."""
    runs = tmp_path / "runs"
    proc = subprocess.run(
        ["bash", str(GRID_SCRIPT)],
        cwd=REPO, env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
                       "REPO_DIR": str(REPO), "RUNS_ROOT": str(runs),
                       "HC": "off", "LRSR": "1e-4", "SEED": "0",
                       "PRINT_RUN_DIR": "1"},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    names = dict(line.split("=", 1) for line in proc.stdout.splitlines()
                 if line.startswith(("RUN_DIR=", "MODEL_NAME=")))
    # Exactly the run dir docs/lrsr_grid_ablation_plan.md §7 promises.
    assert names["RUN_DIR"] == str(
        runs / "sr_r2grid_off_ls1e-4_nohc_gap_ce_anorm_recalpost_rails_seed0")
    assert names["MODEL_NAME"] == \
        "sr_r2grid_off_ls1e-4_nohc_gap_ce_anorm_recalpost_rails_ap"
    assert not runs.exists(), "the name query must not create the run dir"


@needs_bash
@pytest.mark.parametrize("pool,hc,job", [
    ("pool_r2a.sh", "on", "r2agrid-pool"),
    ("pool_r2b.sh", "off", "r2bgrid-pool"),
])
def test_each_lane_pool_is_submittable_with_no_flags(tmp_path, pool, hc, job):
    """`sbatch scripts/hpc/sr/grid/pool_r2a.sh` and nothing else — sbatch reads
    #SBATCH from the file it is given, so the headers have to live here."""
    text = (POOL_DIR / pool).read_text()
    directives = dict(
        line.split("=", 1) for line in text.splitlines()
        if line.startswith("#SBATCH --") and "=" in line)
    assert directives["#SBATCH --job-name"] == job
    assert directives["#SBATCH --time"] == "48:00:00"
    assert directives["#SBATCH --gres"] == "gpu:1"
    assert directives["#SBATCH --cpus-per-task"] == "8"
    # %x = job name: two lanes running at once must not share a log file.
    assert directives["#SBATCH --output"] == "slurm-%x-%j.txt"
    assert f"export HC={hc}" in text
    assert 'LRSRS:-1e-4 1e-5 1e-6 1e-7' in text     # the grid's own bounds, §2


# --------------------------------------------------------------------------
# The arms that were here first: nothing about them may move
# --------------------------------------------------------------------------
@needs_bash
@pytest.mark.parametrize("arm,run_tag", [
    # These tags are not written out here for convenience — they are what
    # scripts/hpc/sr/refit/*.sh RECORDED for these arms before the rails, the
    # PRINT_RUN_DIR hook or the grid existed, and tuned-but-not-yet-fitted runs
    # are sitting in exactly these directories on /scratch right now.
    ("r2a_new.sh", "sr_r2a_new_gap_ce_anorm_recalpost"),
    ("r2b_new.sh", "sr_r2b_new_nohc_gap_ce_anorm_recalpost"),
])
def test_an_arm_that_forces_no_rails_keeps_its_exact_run_dir(tmp_path, arm, run_tag):
    """RAILS_TAG is interpolated into RUN_DIR, STUDY_NAME and MODEL_NAME, so if
    it were ever non-empty by default, a fit would look for best_params.yaml in
    a directory that does not exist and a bench row would land under a name that
    does not group with the tuned seed. It is empty unless STD_BAND_RAISE_LO/HI
    or STD_BAND_ACTION is set, and this pins that."""
    runs = tmp_path / "runs"
    proc = subprocess.run(
        ["bash", str(REPO / "scripts" / "hpc" / "sr" / arm)],
        cwd=REPO, env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
                       "REPO_DIR": str(REPO), "RUNS_ROOT": str(runs),
                       "LOSS_ARM": "gap_ce", "SEED": "0", "PRINT_RUN_DIR": "1"},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    names = dict(line.split("=", 1) for line in proc.stdout.splitlines()
                 if line.startswith(("RUN_DIR=", "MODEL_NAME=")))
    assert names["RUN_DIR"] == str(runs / f"{run_tag}_seed0")
    assert names["MODEL_NAME"] == f"{run_tag}_ap"
    assert "_rails" not in names["RUN_DIR"] + names["MODEL_NAME"]


@needs_bash
@needs_torch
def test_a_benched_cell_is_never_benched_again(tmp_path):
    """The unrecoverable one. The store is append-only with uuid run_ids and the
    bench stage has no duplicate check, so a second row for the same (model,
    seed, split) does not overwrite the first — it ADDS chips, and every mean
    over that arm silently averages them twice. A re-submitted pool walks past
    every finished cell, so this guard is on the hot path, not the edge."""
    import torch

    from benchmarking.store import append_run

    run = tmp_path / "runs" / "cell_off_1e-4_seed0"
    (run / "checkpoints").mkdir(parents=True)
    (run / "best_params.yaml").write_text("model: {}\n")
    (run / "sweep.json").write_text("{}")
    torch.save({"epoch": 100}, run / "checkpoints" / "unet_s2rosa_jointsr_final.ckpt")
    append_run({"run_id": "already-here", "model_name": "model_off_1e-4",
                "seed": 0, "dataset_split": "test"}, tmp_path / "store")

    out, trace = _run_pool(tmp_path, LRSRS="1e-4")
    assert trace == [], f"a finished cell did work: {trace}"
    assert "already in the store" in out


# --------------------------------------------------------------------------
# The per-arm pools: one SLURM job per R-series arm, tuned seed + refit seeds
# --------------------------------------------------------------------------
ARM_POOL = REPO / "scripts" / "hpc" / "sr" / "refit" / "run_arm_pool.sh"

STUB_ARM = """#!/bin/bash
set -euo pipefail
RUN_DIR="${RUNS_ROOT}/arm_${EXP_TAG:-stub}_seed${SEED}"
if [ "${PRINT_RUN_DIR:-0}" = "1" ]; then
  echo "RUN_DIR=${RUN_DIR}"; echo "MODEL_NAME=model_stub"; exit 0
fi
echo "arm ${STAGE} seed=${SEED} resume=${RESUME_FIT:-none}" >> "$TRACE"
mkdir -p "$RUN_DIR/checkpoints"
case "$STAGE" in
  tune) printf 'model:\\n  lr: 0.00042\\n' > "$RUN_DIR/best_params.yaml" ;;
  fit)  echo '{}' > "$RUN_DIR/sweep.json" ;;
esac
"""
STUB_REFIT = """#!/bin/bash
set -euo pipefail
echo "refit lr=${LR:-unset} seeds=${SEEDS:-default}" >> "$TRACE"
"""


def _arm_repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts" / "hpc" / "sr" / "refit").mkdir(parents=True, exist_ok=True)
    shutil.copy(ARM_POOL, tmp_path / "scripts" / "hpc" / "sr" / "refit" / "run_arm_pool.sh")
    shutil.copy(REPO / "scripts" / "hpc" / "sr" / "refit" / "_refit_lib.sh",
                tmp_path / "scripts" / "hpc" / "sr" / "refit" / "_refit_lib.sh")
    # Written only if absent: a test that plants a FAILING stub calls this
    # again (via _run_arm_pool) and must not have it overwritten underneath it.
    for rel, body in (("stubarm.sh", STUB_ARM), ("refit/stubrefit.sh", STUB_REFIT)):
        f = tmp_path / "scripts" / "hpc" / "sr" / rel
        if not f.exists():
            f.write_text(body)
    if not (tmp_path / "src").exists():
        (tmp_path / "src").symlink_to(REPO / "src")
    return tmp_path


def _run_arm_pool(tmp_path: Path, **env) -> tuple[str, list[str]]:
    import os
    repo = _arm_repo(tmp_path)
    trace = tmp_path / "trace"
    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "hpc" / "sr" / "refit" / "run_arm_pool.sh")],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "REPO_DIR": str(repo),
             "RUNS_ROOT": str(tmp_path / "runs"), "STORE_DIR": str(tmp_path / "store"),
             "TRACE": str(trace), "VENV_DIR": str(tmp_path / "novenv"),
             "ARM": "stubarm", "REFIT": "stubrefit", **env},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout, (trace.read_text().splitlines() if trace.exists() else [])


@needs_bash
@needs_torch
def test_the_arm_pool_runs_the_tuned_seed_then_seeds_the_refits_from_its_lr(tmp_path):
    """The two phases exist because the refit script can only PLANT an overlay,
    never make one: the tune writes it at the tuned seed and nowhere else. So
    phase 2's lr has to come from phase 1's output, not from thin air."""
    out, trace = _run_arm_pool(tmp_path)
    assert trace == [
        "arm tune seed=66 resume=none",
        "arm fit seed=66 resume=0",
        "arm bench seed=66 resume=none",
        "refit lr=0.00042 seeds=default",     # read off the tuned overlay
    ]
    # ...and it says so loudly, because the house style is to BAKE it in.
    assert "PASTE THIS" in out and "0.00042" in out


@needs_bash
@needs_torch
def test_an_already_tuned_arm_starts_at_the_fit(tmp_path):
    """r1a's case: the tune is done and waiting. Re-running it would spend a
    fresh N_TRIALS and overwrite the overlay the arm's identity rests on."""
    run = tmp_path / "runs" / "arm_stub_seed66"
    run.mkdir(parents=True)
    (run / "best_params.yaml").write_text("model:\n  lr: 0.00099\n")
    _, trace = _run_arm_pool(tmp_path)
    assert trace[0] == "arm fit seed=66 resume=0", "the tune was re-entered"
    assert trace[-1] == "refit lr=0.00099 seeds=default"


@needs_bash
@needs_torch
def test_the_refit_seeds_can_be_run_alone_and_the_seed_list_overridden(tmp_path):
    run = tmp_path / "runs" / "arm_stub_seed66"
    run.mkdir(parents=True)
    (run / "best_params.yaml").write_text("model:\n  lr: 0.00099\n")
    _, trace = _run_arm_pool(tmp_path, STAGES="refit", SEEDS="3 4")
    assert trace == ["refit lr=0.00099 seeds=3 4"]


@needs_bash
@needs_torch
def test_a_failed_tune_stops_the_arm_instead_of_refitting_on_nothing(tmp_path):
    """Everything downstream reads best_params.yaml. Carrying on would fit the
    engine's DEFAULT hyperparameters and file the row under the arm's name."""
    repo = _arm_repo(tmp_path)
    (repo / "scripts" / "hpc" / "sr" / "stubarm.sh").write_text(
        STUB_ARM.replace('case "$STAGE" in', 'exit 7\ncase "$STAGE" in'))
    out, trace = _run_arm_pool(tmp_path)
    assert trace == ["arm tune seed=66 resume=none"]
    assert "tune s66: FAILED" in out


@needs_bash
@pytest.mark.parametrize("pool,arm,refit,job", [
    ("pool_r1a.sh", "r1a_new", "r1a_new_gap_ce", "r1a-pool"),
    ("pool_r1b.sh", "r1b_new", "r1b_new_nohc_gap_ce", "r1b-pool"),
])
def test_each_arm_pool_is_submittable_with_no_flags(tmp_path, pool, arm, refit, job):
    text = (REPO / "scripts" / "hpc" / "sr" / "refit" / pool).read_text()
    directives = dict(
        line.split("=", 1) for line in text.splitlines()
        if line.startswith("#SBATCH --") and "=" in line)
    assert directives["#SBATCH --job-name"] == job
    assert directives["#SBATCH --time"] == "48:00:00"
    assert directives["#SBATCH --gres"] == "gpu:1"
    assert directives["#SBATCH --cpus-per-task"] == "8"
    assert directives["#SBATCH --output"] == "slurm-%x-%j.txt"
    assert f"export ARM={arm}" in text and f"export REFIT={refit}" in text
    # The refit script it names must exist, or the pool dies after phase 1.
    assert (REPO / "scripts" / "hpc" / "sr" / "refit" / f"{refit}.sh").is_file()
    assert (REPO / "scripts" / "hpc" / "sr" / f"{arm}.sh").is_file()


@needs_bash
@needs_torch
def test_the_pool_discovers_which_seed_was_tuned(tmp_path):
    """The r1b case: the tune runs at seed 42 while the pool's fallback seed is
    66. A wrong seed is NOT a loud failure — the pool would find no overlay,
    conclude the arm is untuned, and spend a fresh N_TRIALS search whose lr then
    seeds every refit. So it globs for the overlay instead of assuming."""
    run = tmp_path / "runs" / "arm_stub_seed42"
    run.mkdir(parents=True)
    (run / "best_params.yaml").write_text("model:\n  lr: 0.00042\n")
    out, trace = _run_arm_pool(tmp_path)
    assert "tuned seed: 42 (discovered" in out
    assert trace == [                       # no tune: it was found, not remade
        "arm fit seed=42 resume=0",
        "arm bench seed=42 resume=none",
        "refit lr=0.00042 seeds=default",
    ]


@needs_bash
@needs_torch
def test_discovery_survives_being_queued_before_the_tune_finishes(tmp_path):
    """`sbatch --dependency=afterok:<tune jobid>` submits this pool while the
    overlay does not exist yet. The glob runs at JOB START, not at submit, so
    the seed only has to exist by then."""
    repo = _arm_repo(tmp_path)             # nothing on disk yet, as at submit
    (tmp_path / "runs" / "arm_stub_seed42").mkdir(parents=True)   # ...tune lands
    (tmp_path / "runs" / "arm_stub_seed42" / "best_params.yaml").write_text(
        "model:\n  lr: 0.00042\n")
    out, _ = _run_arm_pool(tmp_path)
    assert "tuned seed: 42 (discovered" in out
    assert repo.is_dir()


@needs_bash
@needs_torch
def test_two_tuned_seeds_are_a_decision_not_a_default(tmp_path):
    """Two searches for one arm are two different hyperparameter sets. Picking
    one by glob order would let the filesystem decide the arm's identity."""
    for seed, lr in (("42", "0.00042"), ("66", "0.00066")):
        d = tmp_path / "runs" / f"arm_stub_seed{seed}"
        d.mkdir(parents=True)
        (d / "best_params.yaml").write_text(f"model:\n  lr: {lr}\n")
    repo = _arm_repo(tmp_path)
    import os
    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "hpc" / "sr" / "refit" / "run_arm_pool.sh")],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "REPO_DIR": str(repo),
             "RUNS_ROOT": str(tmp_path / "runs"), "STORE_DIR": str(tmp_path / "store"),
             "TRACE": str(tmp_path / "trace"), "VENV_DIR": str(tmp_path / "novenv"),
             "ARM": "stubarm", "REFIT": "stubrefit"},
        capture_output=True, text=True)
    assert proc.returncode == 2
    assert "2 tuned seeds found" in proc.stderr
    assert "seed 42" in proc.stderr and "seed 66" in proc.stderr
    assert not (tmp_path / "trace").exists(), "it ran work before deciding"

    # ...and naming one resolves it.
    out, trace = _run_arm_pool(tmp_path, TUNED_SEED="42")
    assert "set explicitly" in out
    assert trace[-1] == "refit lr=0.00042 seeds=default"


@needs_bash
@needs_torch
def test_an_untuned_arm_will_not_be_refitted_on_a_tune_it_was_told_not_to_run(tmp_path):
    """STAGES without `tune` and no overlay anywhere: refuse. Falling through
    would fit the engine's DEFAULT hyperparameters under the arm's name."""
    repo = _arm_repo(tmp_path)
    import os
    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "hpc" / "sr" / "refit" / "run_arm_pool.sh")],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "REPO_DIR": str(repo),
             "RUNS_ROOT": str(tmp_path / "runs"), "STORE_DIR": str(tmp_path / "store"),
             "TRACE": str(tmp_path / "trace"), "VENV_DIR": str(tmp_path / "novenv"),
             "ARM": "stubarm", "REFIT": "stubrefit", "STAGES": "fit bench refit"},
        capture_output=True, text=True)
    assert proc.returncode == 2
    assert "not in" in proc.stderr and "STAGES" in proc.stderr


@needs_bash
@needs_torch
def test_a_refit_seed_list_containing_the_tuned_seed_is_flagged(tmp_path):
    """run_seeds PLANTS the overlay per seed. Aimed at the tuned seed it would
    overwrite the one sr.tune wrote, making the tune's own output derived."""
    run = tmp_path / "runs" / "arm_stub_seed42"
    run.mkdir(parents=True)
    (run / "best_params.yaml").write_text("model:\n  lr: 0.00042\n")
    _run_arm_pool(tmp_path, SEEDS="42 888")   # stderr warn, does not abort
    import os
    repo = _arm_repo(tmp_path)
    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "hpc" / "sr" / "refit" / "run_arm_pool.sh")],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "REPO_DIR": str(repo),
             "RUNS_ROOT": str(tmp_path / "runs"), "STORE_DIR": str(tmp_path / "store"),
             "TRACE": str(tmp_path / "trace2"), "VENV_DIR": str(tmp_path / "novenv"),
             "ARM": "stubarm", "REFIT": "stubrefit", "SEEDS": "42 888"},
        capture_output=True, text=True)
    assert "contains the TUNED seed" in proc.stderr


# --------------------------------------------------------------------------
# The GENERATED per-cell seed-refit scripts (scripts/local/make_grid_refit_scripts.py)
#
# A seed-N run dir has never been tuned, so its overlay has to be planted, and
# the copy baked into the script is the only thing that says which
# hyperparameters that seed ran. Two ways for that to go quietly wrong: the
# baked overlay belongs to a different cell than the filename claims, or the
# engine's naming moves and the seeds land in a directory the append-only store
# will not group with seed 0. Both are checked here, and by the scripts
# themselves at run time.
# --------------------------------------------------------------------------
REFIT_DIR = REPO / "scripts" / "hpc" / "sr" / "grid" / "refit"


def _generated_cells():
    return sorted(REFIT_DIR.glob("r2grid_*_ls*.sh"))


def _script_field(text: str, name: str) -> str:
    line = next(ln for ln in text.splitlines() if ln.startswith(f'{name}='))
    return line.split("=", 1)[1].strip().strip('"')


def _baked_overlay(text: str) -> dict:
    # `read -r -d '' BEST_PARAMS <<'YAML' || true` — skip to the end of the
    # redirect line, then take everything up to the terminator.
    body = text.split("<<'YAML'", 1)[1].split("\n", 1)[1].split("\nYAML\n", 1)[0]
    return yaml.safe_load(body)


@pytest.mark.skipif(not _generated_cells(), reason="no generated cell scripts")
@pytest.mark.parametrize("script", _generated_cells(), ids=lambda p: p.stem)
def test_a_generated_cell_script_bakes_its_own_cells_overlay(script):
    """Filename, HC/LRSR, baked lr_sr and expected run tag must all name ONE
    cell. A mismatch would refit a seed of cell A under cell B's tune while
    filing the row under B — invisible in the store, fatal to the comparison."""
    text = script.read_text()
    lane, _, tag = script.stem[len("r2grid_"):].partition("_ls")
    model = _baked_overlay(text)["model"]

    mantissa, exponent = f"{float(model['lr_sr']):.1e}".split("e")
    assert f"{mantissa.removesuffix('.0')}e{int(exponent)}" == tag
    assert _script_field(text, "export HC") == lane
    assert float(_script_field(text, "export LRSR")) == float(model["lr_sr"])

    run_tag = _script_field(text, "EXPECTED_RUN_TAG")
    assert run_tag.startswith(f"sr_r2grid_{lane}_ls{tag}_")
    # The lane, spelled the way the engine tags it, and the band policy that
    # keeps every grid row out of the formal arms' rows.
    assert ("_nohc" in run_tag) is (lane == "off")
    assert run_tag.endswith("_rails")
    # The refit must run the band the search ran (§3), so the overlay carries it.
    assert model["std_band_raise_lo"] == 0.01
    assert model["std_band_raise_hi"] == 100.0
    # ...and the loss stays the frozen control it is for the whole grid.
    assert model["loss_arm"] == "gap_ce" and model["pstar"] == "gap_ce"


def _refit_repo(tmp_path: Path, script: Path, *, engine_tag: str | None = None) -> Path:
    """A REPO_DIR where r2grid_new.sh is a stub that answers PRINT_RUN_DIR and
    records the stages it was asked for."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "hpc" / "sr" / "refit").mkdir(parents=True)
    (repo / "scripts" / "hpc" / "sr" / "grid" / "refit").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "hpc" / "sr" / "refit" / "_refit_lib.sh",
                repo / "scripts" / "hpc" / "sr" / "refit" / "_refit_lib.sh")
    shutil.copy(script, repo / "scripts" / "hpc" / "sr" / "grid" / "refit" / script.name)
    # Reproduces the engine's tag for this cell (or a deliberately wrong one).
    name = engine_tag or (
        'sr_r2grid_${HC}_ls$(printf "%s" "$LRSR" | awk \'{'
        'split(sprintf("%.1e", $0), a, "e"); m = a[1]; sub(/\\.0$/, "", m);'
        ' printf "%se%d", m, a[2] + 0 }\')'
        '$( [ "$HC" = off ] && echo _nohc )_gap_ce_anorm_recalpost_rails')
    (repo / "scripts" / "hpc" / "sr" / "r2grid_new.sh").write_text(
        '#!/bin/bash\nset -euo pipefail\n'
        f'TAG="{name}"\n'
        'RUN_DIR="${RUNS_ROOT}/${TAG}_seed${SEED}"\n'
        'if [ "${PRINT_RUN_DIR:-0}" = "1" ]; then\n'
        '  echo "RUN_DIR=${RUN_DIR}"\n'
        '  echo "MODEL_NAME=${TAG}_ap"\n'
        '  exit 0\n'
        'fi\n'
        'echo "${STAGE} seed=${SEED} overlay=$(test -f "$RUN_DIR/best_params.yaml" '
        '&& echo planted || echo MISSING)" >> "$TRACE"\n')
    return repo


def _run_refit(repo: Path, script_name: str, tmp_path: Path, **env):
    import os
    return subprocess.run(
        ["bash", str(repo / "scripts" / "hpc" / "sr" / "grid" / "refit" / script_name)],
        env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "REPO_DIR": str(repo),
             "RUNS_ROOT": str(tmp_path / "runs"), "STORE_DIR": str(tmp_path / "store"),
             "TRACE": str(tmp_path / "trace"), **env},
        capture_output=True, text=True)


@needs_bash
@pytest.mark.skipif(not _generated_cells(), reason="no generated cell scripts")
def test_the_refit_plants_the_baked_overlay_before_each_seeds_fit(tmp_path):
    script = _generated_cells()[0]
    repo = _refit_repo(tmp_path, script)
    proc = _run_refit(repo, script.name, tmp_path, SEEDS="1 2")
    assert proc.returncode == 0, proc.stderr

    trace = (tmp_path / "trace").read_text().split("\n")
    # The overlay is on disk BEFORE the fit that needs it, for every seed.
    assert "fit seed=1 overlay=planted" in trace
    assert "fit seed=2 overlay=planted" in trace
    for seed in (1, 2):
        planted = next((tmp_path / "runs").glob(f"*_seed{seed}/best_params.yaml"))
        assert yaml.safe_load(planted.read_text()) == _baked_overlay(script.read_text())


@needs_bash
@pytest.mark.skipif(not _generated_cells(), reason="no generated cell scripts")
def test_the_refit_refuses_to_overwrite_the_tuned_seeds_own_overlay(tmp_path):
    """Seed 0 IS the tune this script was generated from; planting over it
    would make the tune's own output a derived copy of a copy."""
    script = _generated_cells()[0]
    repo = _refit_repo(tmp_path, script)
    proc = _run_refit(repo, script.name, tmp_path, SEEDS="0 1")
    assert proc.returncode == 0
    assert "SEED=0 skipped" in proc.stderr
    assert not list((tmp_path / "runs").glob("*_seed0"))
    assert "fit seed=1 overlay=planted" in (tmp_path / "trace").read_text()


@needs_bash
@pytest.mark.skipif(not _generated_cells(), reason="no generated cell scripts")
def test_a_refit_stops_when_the_engine_names_the_cell_differently(tmp_path):
    """The store is append-only: seeds filed under a tag seed 0 does not carry
    are not a replicate, they are a new arm nobody will ever notice."""
    script = _generated_cells()[0]
    repo = _refit_repo(tmp_path, script, engine_tag="sr_r2grid_renamed_somehow")
    proc = _run_refit(repo, script.name, tmp_path, SEEDS="1")
    assert proc.returncode == 2
    assert "would not join seed 0's rows" in proc.stderr
    assert not (tmp_path / "trace").exists()
