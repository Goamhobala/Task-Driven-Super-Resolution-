"""The RL campaign's CLUSTER arms — rl3/rl4 under scripts/hpc/sr/rl/.

The campaign (docs/rl_lightning_campaign_plan.md, rev 3) was written for
Lightning Studio. The two SR4RS arms moved to the l40s partition because rl4
OOMs a 24 GB L4, which creates the failure mode these tests exist to prevent:
one campaign, two platforms, and a constant that drifts between them would not
break a run — it would break the CONTROL, silently, and only the write-up would
notice.

  * §1  the cluster `_rl_common.sh` / `_rl_rung.sh` carry EXACTLY the Studio's
        constants. The three platform deltas are enumerated, and anything else
        is drift.
  * §2  rl3 early-stops (patience 5 on val_ap) and rl4 does not — the one
        deliberate difference from the Studio arms — and the store tags follow:
        an early-stopped row can never land beside a fixed-budget one.
  * §3  the arms' SR treatment and the engine plumbing behind FIT_EARLY_STOP.

Everything runs on bash alone: a stub engine for the arm scripts, and the real
engine in its PRINT_RUN_DIR=1 name-query mode, which touches no /scratch, no
venv and no GPU.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HPC_RL = REPO / "scripts" / "hpc" / "sr" / "rl"
LS_RL = REPO / "scripts" / "LightningStudio" / "sr" / "rl"
ENGINE = REPO / "scripts" / "hpc" / "sr" / "_stages_tv.sh"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


# --------------------------------------------------------------------------
# §1 — one campaign, two platforms: the constants must be the same constants
# --------------------------------------------------------------------------
def _assignments(path: Path) -> dict[str, str]:
    """VAR=value for every top-level assignment, comments and prose dropped."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        m = re.match(r'^([A-Z][A-Z0-9_]*)=(.*)$', line)
        if m:
            out[m.group(1)] = m.group(2).split("#")[0].strip()
    return out


# The ONLY differences allowed between the two `_rl_common.sh` files, each one
# platform (never scientific) and each documented in the cluster file's header.
ALLOWED_COMMON_DELTAS = {
    # env.sh exports NUM_WORKERS=0 in the Studio and has to be overridden there.
    # On SLURM the value arrives unset and the engine derives it from
    # --cpus-per-task, which is strictly better than a hard-coded 4.
    "NUM_WORKERS",
    # Reporting destination, not a treatment: the cluster arms log to
    # instaroad_rl_lightning, where this campaign's runs actually live. Nothing
    # a contrast depends on — but if the Studio file's sr_s2rosa_rl_campaign is
    # stale rather than deliberate, align the two and delete this entry.
    "WANDB_PROJECT",
}


def test_the_cluster_campaign_constants_are_the_studios():
    """Head, loss, λ, budget, batch size, rails, anorm, protocol, splits — if
    any of these differ, `rl4 - rl3` stops being a within-campaign contrast and
    the two platforms' rows must not share a model_name."""
    hpc = _assignments(HPC_RL / "_rl_common.sh")
    ls = _assignments(LS_RL / "_rl_common.sh")
    shared = (set(hpc) | set(ls)) - ALLOWED_COMMON_DELTAS
    differing = {k: (hpc.get(k), ls.get(k)) for k in shared if hpc.get(k) != ls.get(k)}
    assert not differing, f"campaign constants drifted between platforms: {differing}"


def test_the_ladder_is_the_same_ladder():
    """A rung that means one dose in the Studio and another on the cluster would
    put two different treatments under one `_ls` tag in an append-only store."""
    def code(p: Path) -> list[str]:
        return [ln for ln in p.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    assert code(HPC_RL / "_rl_rung.sh") == code(LS_RL / "_rl_rung.sh")


def test_early_stopping_is_the_only_knob_the_cluster_file_adds():
    """The cluster file may not quietly grow campaign knobs of its own. It is
    allowed to say nothing about FIT_EARLY_STOP (the arms own it) — and it must
    not ASSIGN it, or an arm's own default could never win."""
    code = "\n".join(ln for ln in (HPC_RL / "_rl_common.sh").read_text().splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "FIT_EARLY_STOP:=" not in code, (
        "`: ${FIT_EARLY_STOP:=0}` binds the variable, so rl3's own "
        "`${FIT_EARLY_STOP:-1}` would see 0 and keep it")
    assert "FIT_EARLY_STOP" not in _assignments(HPC_RL / "_rl_common.sh")


# --------------------------------------------------------------------------
# §2/§3 — the arm scripts, run against a stub engine
# --------------------------------------------------------------------------
STUB_VARS = (
    "STAGE EXP_TAG LABELS UPSAMPLER FREEZE_SR SR_HC SR_PAD HEAD MONITOR "
    "FIT_EARLY_STOP ES_PATIENCE ES_MONITOR ES_MODE "
    "N_TRIALS TUNE_EPOCHS PATIENCE BATCH_SIZES REFIT_EPOCHS SR_HOLD_EPOCHS "
    "SR_WARMUP_EPOCHS LR_MIN LR_MAX LR_SR_MIN LR_SR_MAX LOSS_ARM PSTAR "
    "POS_WEIGHT_MIN POS_WEIGHT_MAX SEARCH_THETAS SEARCH_MIX_W TRAIN_SPLITS "
    "FIT_VAL_LOOP BENCH_SPLIT SWEEP_SPLIT AP_BINS TILE_METRICS "
    "STD_BAND_RAISE_LO STD_BAND_RAISE_HI STD_BAND_ACTION SR_SNAPSHOT_EVERY "
    "ADAPTIVE_NORM NORM_RECALIBRATE SEARCH_GPUS REFIT_GPUS PRECISION "
    "SEN2SR_DIR SR4RS_GRAD_CKPT CHAIN_BENCH"
).split()


def _stub_arm(tmp_path: Path, arm: str) -> Path:
    """Copy the rl/ dir beside a STUB engine that just echoes the variables.

    The real engine wants /scratch, a venv and a GPU; what is under test is the
    arm script plus `_rl_common.sh`, i.e. everything above the `source`.
    """
    dst = tmp_path / "scripts" / "hpc" / "sr" / "rl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HPC_RL, dst, dirs_exist_ok=True)
    (dst.parent / "_stages_tv.sh").write_text(
        "for k in " + " ".join(STUB_VARS) + "; do\n"
        # ${!k-} not ${!k}: STAGE is the engine's own default and is unset when
        # an arm script is run directly, which `set -u` would abort on.
        "  printf '%s=%s\\n' \"$k\" \"${!k-}\"\n"
        "done\n"
        # The overlay is multi-line, so it goes between markers instead.
        "printf '<<<OVERLAY\\n%s\\nOVERLAY>>>\\n' \"${BEST_PARAMS-}\"\n")
    return dst / arm


def _run_arm(tmp_path: Path, arm: str, **env: str) -> dict[str, str]:
    proc = subprocess.run(
        ["bash", str(_stub_arm(tmp_path, arm))],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "USER": "tester", "REPO_DIR": str(tmp_path), **env},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines()
                if re.match(r'^[A-Z][A-Z0-9_]*=', line))


@needs_bash
def test_rl3_early_stops_on_val_ap_with_patience_5(tmp_path):
    """The requested cluster-only departure. `val_ap`, never `val_iou@0.5`: a
    5-parameter logistic regression has no reason to be calibrated at 0.5, so an
    IoU@0.5 stopper would fire on where the arm happens to put its boundary
    (probe doc §5.1)."""
    rl3 = _run_arm(tmp_path, "rl3.sh")
    assert rl3["FIT_EARLY_STOP"] == "1"
    assert rl3["ES_PATIENCE"] == "5"
    assert (rl3["ES_MONITOR"], rl3["ES_MODE"]) == ("val_ap", "max")


@needs_bash
def test_rl4_runs_the_whole_budget(tmp_path):
    """rl4's measured quantity IS its trajectory — how far task gradients walk
    an unanchored generator per unit dose, and where (or whether) it dies. A
    stopper would truncate the record exactly where the interesting part
    starts, and a rung that ended at epoch 17 could not be compared with one
    that ran to 30."""
    rl4 = _run_arm(tmp_path, "rl4.sh", LRSR="1e-3")
    assert rl4["FIT_EARLY_STOP"] == "0"
    assert rl4["REFIT_EPOCHS"] == "30" and rl4["SR_HOLD_EPOCHS"] == "10"


@needs_bash
def test_the_stopper_reaches_neither_arm_by_accident(tmp_path):
    """Each arm states its own policy rather than inheriting one, so a change to
    the shared file cannot silently switch either."""
    assert 'FIT_EARLY_STOP="${FIT_EARLY_STOP:-1}"' in (HPC_RL / "rl3.sh").read_text()
    assert 'FIT_EARLY_STOP="${FIT_EARLY_STOP:-0}"' in (HPC_RL / "rl4.sh").read_text()


@needs_bash
def test_both_arms_are_the_bare_sr4rs_row(tmp_path):
    """SR4RS ships no FFT constraint, so `off` and `native` are the same
    operator here; forcing `off` is what gives every arm in the series the same
    _nohc tag instead of an implicit constraint state."""
    rl3 = _run_arm(tmp_path, "rl3.sh")
    rl4 = _run_arm(tmp_path, "rl4.sh", LRSR="1e-4")
    for arm in (rl3, rl4):
        assert arm["UPSAMPLER"] == "sr4rs"
        assert (arm["SR_HC"], arm["SR_PAD"]) == ("off", "0")
        assert arm["LABELS"] == "new"
        assert arm["SEN2SR_DIR"].endswith("/models/SR4RS_RGBN")
    assert (rl3["FREEZE_SR"], rl4["FREEZE_SR"]) == ("true", "false")
    assert rl4["SR4RS_GRAD_CKPT"] == "0"   # superseded by bf16; still available


@needs_bash
def test_both_arms_carry_the_campaign_controls(tmp_path):
    """Everything except the SR treatment and the pinned lr_sr is a between-arm
    constant; the head lr is pinned (min == max) rather than searched, and the
    "tune" is a 1x1 config-writing + timing pass, not a search."""
    arms = [_run_arm(tmp_path, "rl3.sh"),
            _run_arm(tmp_path, "rl4.sh", LRSR="1e-6"),
            _run_arm(tmp_path, "rl2.sh", LRSR="1e-6")]
    for arm in arms:
        assert arm["HEAD"] == "linear" and arm["MONITOR"] == "val_ap"
        assert arm["LR_MIN"] == arm["LR_MAX"] == "3e-3"
        assert (arm["N_TRIALS"], arm["TUNE_EPOCHS"]) == ("1", "1")
        assert arm["LOSS_ARM"] == "wbce"
        assert arm["POS_WEIGHT_MIN"] == arm["POS_WEIGHT_MAX"]
        assert arm["BATCH_SIZES"] == "4" and arm["REFIT_EPOCHS"] == "30"
        assert arm["TRAIN_SPLITS"] == "train" and arm["FIT_VAL_LOOP"] == "1"
        assert (arm["STD_BAND_RAISE_LO"], arm["STD_BAND_RAISE_HI"],
                arm["STD_BAND_ACTION"]) == ("0.01", "100", "warn")
        assert arm["SR_SNAPSHOT_EVERY"] == "1"
        assert arm["ADAPTIVE_NORM"] == "1" and arm["NORM_RECALIBRATE"] == "post"
        assert arm["SEARCH_GPUS"] == arm["REFIT_GPUS"] == "1"


@needs_bash
def test_rl4_refuses_a_rung_it_cannot_name(tmp_path):
    """No LRSR = no dose = no run. (The tag normalisation itself is pinned by
    tests/test_lrsr_grid.py, on the same `_ls…` derivation.)"""
    proc = subprocess.run(
        ["bash", str(_stub_arm(tmp_path, "rl4.sh"))],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "USER": "tester", "REPO_DIR": str(tmp_path)},
        capture_output=True, text=True)
    assert proc.returncode != 0 and "LRSR" in proc.stderr


# --------------------------------------------------------------------------
# §3 — the engine: what FIT_EARLY_STOP does to the store, and what it refuses
# --------------------------------------------------------------------------
def _names(tmp_path: Path, arm: str, **env: str) -> dict[str, str]:
    """Ask the REAL engine for this arm's RUN_DIR / MODEL_NAME (PRINT_RUN_DIR
    prints and exits before it touches /scratch, the venv or a GPU)."""
    proc = subprocess.run(
        ["bash", str(HPC_RL / arm)],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "USER": "tester", "REPO_DIR": str(REPO), "PRINT_RUN_DIR": "1",
             "RUNS_ROOT": str(tmp_path / "runs"), **env},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return dict(line.split("=", 1) for line in proc.stdout.splitlines()
                if line.startswith(("RUN_DIR=", "MODEL_NAME=")))


@needs_bash
def test_an_early_stopped_row_cannot_land_beside_a_fixed_budget_one(tmp_path):
    """The store is append-only and `_es5` is the only thing separating this
    arm's rows from the Studio's rl3 rows, which ran the full 30 epochs. Same
    arm, same seed, different protocol — averaging them would be a silent
    protocol mix."""
    on = _names(tmp_path, "rl3.sh")
    off = _names(tmp_path, "rl3.sh", FIT_EARLY_STOP="0")
    assert "_es5" in on["RUN_DIR"] and "_es5" in on["MODEL_NAME"]
    assert "_es" not in off["RUN_DIR"] and "_es" not in off["MODEL_NAME"]
    # ...and with the stopper off, the cluster arm IS the Studio arm.
    assert off["MODEL_NAME"] == (
        "sr_rl3_new_nohc_linear_wbce_anorm_recalpost_rails_holdout_ap")


@needs_bash
def test_the_patience_is_in_the_tag_so_two_patiences_are_two_rows(tmp_path):
    assert "_es3" in _names(tmp_path, "rl3.sh", ES_PATIENCE="3")["MODEL_NAME"]


@needs_bash
def test_rl4_rows_merge_with_the_studios(tmp_path):
    """Nothing about running the same arm on l40s instead of an L4 changes what
    it is, so the rung's rows must group across platforms."""
    assert _names(tmp_path, "rl4.sh", LRSR="1e-4")["MODEL_NAME"] == (
        "sr_rl4_new_ls1e-4_nohc_linear_wbce_anorm_recalpost_rails_holdout_ap")


@needs_bash
def test_only_rl3_can_ever_early_stop(tmp_path):
    """The exception is rl3-shaped, not campaign-shaped: a frozen generator with
    a pre-registered plateau. Every other arm's budget is the fixed, identical
    one its comparisons assume, so the engine refuses the flag by EXP_TAG rather
    than trusting the caller — a truncated budget must not be reachable by a
    submit-time typo, and an _es row must not appear for an arm whose reported
    rows are fixed-budget."""
    for arm, env in (("rl4.sh", {"LRSR": "1e-3"}),
                     ("../r5_new.sh", {"TRAIN_SPLITS": "train", "FIT_VAL_LOOP": "1"})):
        proc = subprocess.run(
            ["bash", str(HPC_RL / arm)],
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
                 "USER": "tester", "REPO_DIR": str(REPO), "PRINT_RUN_DIR": "1",
                 "RUNS_ROOT": str(tmp_path / "runs"), "FIT_EARLY_STOP": "1", **env},
            capture_output=True, text=True)
        assert proc.returncode != 0, f"{arm} accepted FIT_EARLY_STOP=1"
        assert "allowed for rl3 ONLY" in proc.stderr
        assert "_es" not in proc.stdout


@needs_bash
@pytest.mark.parametrize("env,why", [
    ({"FIT_VAL_LOOP": "0"}, "FIT_VAL_LOOP"),
    ({"ES_MODE": "up"}, "ES_MODE must be min|max"),
])
def test_the_engine_refuses_a_stopper_it_cannot_honour(tmp_path, env, why):
    """No val loop = no `val_ap` = EarlyStopping(strict) aborting at the first
    check; a bad mode would stop on the wrong sign. Both must fail at submit,
    not in hour three of a queue slot."""
    proc = subprocess.run(
        ["bash", str(HPC_RL / "rl3.sh")],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "USER": "tester", "REPO_DIR": str(REPO), "PRINT_RUN_DIR": "1",
             "RUNS_ROOT": str(tmp_path / "runs"), **env},
        capture_output=True, text=True)
    assert proc.returncode != 0 and why in proc.stderr


def test_the_stopper_is_appended_to_the_fit_and_nothing_else():
    """`+=` appends to the trainval overlay's callback list (the checkpoint and
    snapshot callbacks survive), and the flags reach the fit command only — the
    tune stage keeps its own PATIENCE and the test/bench paths are untouched."""
    engine = ENGINE.read_text()
    assert '"--trainer.callbacks+=lightning.pytorch.callbacks.EarlyStopping"' in engine
    assert engine.count('${ES_ARGS_FIT[@]+"${ES_ARGS_FIT[@]}"}') == 1
    fit_cmd = engine.split("python -m sr.cli fit")[1].split("\n\n")[0]
    assert "ES_ARGS_FIT" in fit_cmd


def test_the_lightning_engine_is_still_in_sync_with_this_one():
    """The Studio engine is generated from the HPC one; FIT_EARLY_STOP was added
    here, so a stale copy would mean the same arm script means different things
    on the two platforms."""
    proc = subprocess.run(
        ["python3", str(REPO / "scripts/LightningStudio/sync_engine.py"), "--check"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# §4 — the planted overlay: no tune stage at all
# --------------------------------------------------------------------------
# What the campaign pins, in the form `sr.tune` would have seen them: the values
# a 1x1 "search" would have suggested from bands whose ends are equal.
PINNED_LOSS_HP = dict(pstar="bce", gap_r=4, gap_k=60.0, tl_ell=5, tl_theta=0.375,
                      gap_theta=0.55836, tversky_alpha=0.7, cl_alpha=0.3,
                      cl_iters=5, sr_w=1.0, sr_radius=1, warmup_start=30,
                      warmup_ramp=10, mix_w=0.6075946831862098)
PINNED_HEAD_LR = 0.003
PINNED_POS_WEIGHT = 2.4789710497080004


def _planted(tmp_path: Path, arm: str, **env: str) -> dict:
    """The overlay the arm script carries, parsed."""
    proc = subprocess.run(
        ["bash", str(_stub_arm(tmp_path, arm))],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "USER": "tester", "REPO_DIR": str(tmp_path), **env},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    body = proc.stdout.split("<<<OVERLAY\n")[1].split("\nOVERLAY>>>")[0]
    import yaml
    return yaml.safe_load(body)


def _what_a_tune_would_write(tmp_path: Path, *, frozen: bool, lr_sr=None,
                             hold=None, upsampler: str = "sr4rs") -> dict:
    """Rebuild the same overlay through the function that OWNS the schema."""
    import yaml
    from sr import tune as sr_tune

    class _Study:
        best_params = {"lr": PINNED_HEAD_LR, "batch_size": 4,
                       "pos_weight": PINNED_POS_WEIGHT,
                       **({} if lr_sr is None else {"lr_sr": lr_sr})}
        best_value, study_name, trials = 0.0, "x", []

        class best_trial:                      # noqa: N801 - stands in for optuna's
            number = 0

    out = tmp_path / f"ref_{upsampler}_{frozen}_{lr_sr}"
    out.mkdir(parents=True, exist_ok=True)
    path = sr_tune.write_best_overlay(
        _Study(), out, "imagenet", upsampler, freeze_sr=frozen, sr_pad=0,
        loss_arm="wbce", loss_hp=dict(PINNED_LOSS_HP), precision="bf16-mixed",
        mask_source="raster", mask_dirname="mask_new_2pt5",
        lr_schedule="cosine", sr_warmup_epochs=1.0, sr_hold_epochs=hold,
        l2sp_lambda=0.0, adaptive_norm=True, adaptive_norm_momentum=0.01,
        norm_recalibrate="post", head="linear", clip_sr=1.0, sr_hc="off",
        std_band_raise_lo=0.01, std_band_raise_hi=100, monitor="val_ap")
    return yaml.safe_load(path.read_text())


@needs_bash
@pytest.mark.parametrize("arm,up,frozen,env,lr_sr,hold", [
    ("rl3.sh", "sr4rs", True, {}, None, None),
    ("rl4.sh", "sr4rs", False, {"LRSR": "1e-3"}, 1e-3, 10.0),
    ("rl4.sh", "sr4rs", False, {"LRSR": "1e-6"}, 1e-6, 10.0),
    ("rl2.sh", "sen2sr", False, {"LRSR": "1e-3"}, 1e-3, 10.0),
    ("rl2.sh", "sen2sr", False, {"LRSR": "1e-6"}, 1e-6, 10.0),
])
def test_the_planted_overlay_is_what_a_tune_would_have_written(
        tmp_path, arm, up, frozen, env, lr_sr, hold):
    """These arms search nothing, so their 1x1 tune stage was pure ceremony and
    the overlay is written directly. The cost of that shortcut is a SECOND
    author for the overlay schema — `sr.tune.write_best_overlay` and a heredoc —
    and a divergence would not fail loudly: it would train a slightly different
    model. So the heredoc is checked against the function, key for key."""
    assert _planted(tmp_path, arm, **env) == _what_a_tune_would_write(
        tmp_path, frozen=frozen, lr_sr=lr_sr, hold=hold, upsampler=up)


@needs_bash
@pytest.mark.parametrize("lrsr,value", [
    ("1e-3", 1e-3), ("1e-6", 1e-6), ("0.0001", 1e-4), ("3e-4", 3e-4),
])
def test_the_rung_reaches_the_overlay_as_a_yaml_FLOAT(tmp_path, lrsr, value):
    """`lr_sr: 1e-3` is a STRING to PyYAML — its 1.1 resolver wants a dot and a
    signed exponent — so an un-normalised rung would reach the model as text and
    the ladder's treatment would arrive mistyped. Every spelling of a rung must
    land as the same number."""
    lr_sr = _planted(tmp_path, "rl4.sh", LRSR=lrsr)["model"]["lr_sr"]
    assert isinstance(lr_sr, float) and lr_sr == pytest.approx(value, rel=1e-9)


@needs_bash
def test_the_frozen_arm_carries_no_lr_sr_and_no_hold(tmp_path):
    """There is no SR parameter group to give a rate to, or to hold at zero."""
    model = _planted(tmp_path, "rl3.sh")["model"]
    assert "lr_sr" not in model and "sr_hold_epochs" not in model
    assert model["freeze_sr"] is True


@needs_bash
def test_the_overlay_carries_what_the_fit_belt_does_not(tmp_path):
    """The engine re-passes most of the config as --model.* after the config
    layers, so those keys cannot drift. These are the ones it does NOT pass:
    the overlay is their only source, and a missing one is not an error — it is
    joint_sr.yaml's default, silently."""
    for arm, env in (("rl3.sh", {}), ("rl4.sh", {"LRSR": "1e-5"})):
        ov = _planted(tmp_path, arm, **env)
        assert ov["model"]["lr"] == PINNED_HEAD_LR
        assert ov["model"]["pos_weight"] == PINNED_POS_WEIGHT
        assert ov["data"]["batch_size"] == 4
        assert ov["trainer"]["precision"] == "bf16-mixed"
        # head=linear builds no U-Net; a resnet34 here would let the run be read
        # back as an encoder ablation of a network that was never constructed.
        assert ov["model"]["encoder_name"] is None
        assert ov["model"]["encoder_weights"] is None


@needs_bash
def test_the_engine_plants_it_for_a_fit_but_never_for_a_tune(tmp_path):
    """A search writes its own overlay. Planting one first would leave a file
    the study is about to overwrite, and a tune that ran would be
    indistinguishable from one that did not.

    Driven through a MINIMAL arm rather than rl3.sh, because rl3 now refuses
    STAGE=tune outright (see the stage guard below) — the rule under test here
    is the engine's, and it protects every arm that ever plants an overlay."""
    def run(stage: str) -> Path:
        runs = tmp_path / stage
        arm = tmp_path / f"arm_{stage}.sh"
        arm.write_text(
            'EXP_TAG=planttest\nLABELS=new\nUPSAMPLER=bicubic\n'
            'FREEZE_SR=false\nSR_PAD=0\n'
            "BEST_PARAMS='model:\n  lr: 0.003'\n"
            f'source "{REPO}/scripts/hpc/sr/_stages_tv.sh"\n')
        subprocess.run(
            ["bash", str(arm)],
            env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
                 "USER": "tester", "REPO_DIR": str(REPO), "STAGE": stage,
                 "RUNS_ROOT": str(runs)},
            capture_output=True, text=True)   # fails later (no dataset) — fine
        return next(runs.glob("sr_planttest*"))

    assert (run("fit") / "best_params.yaml").is_file()
    assert not (run("tune") / "best_params.yaml").exists()


@needs_bash
def test_a_planted_overlay_never_silently_replaces_a_trained_ones(tmp_path):
    """If checkpoints exist, they were trained under the file on disk. Replacing
    it would leave the run dir describing a config that did not produce its
    weights — the one thing an overlay exists to prevent."""
    runs = tmp_path / "runs"
    run_dir = runs / ("sr_rl3_new_nohc_linear_wbce_anorm_recalpost_rails_es5"
                      "_holdout_seed0")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "last.ckpt").write_bytes(b"weights")
    (run_dir / "best_params.yaml").write_text("model:\n  lr: 0.999\n")
    proc = subprocess.run(
        ["bash", str(HPC_RL / "rl3.sh")],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "USER": "tester", "REPO_DIR": str(REPO), "STAGE": "fit",
             "RUNS_ROOT": str(runs)},
        capture_output=True, text=True)
    assert proc.returncode == 2 and "already holds checkpoints" in proc.stderr
    assert "0.999" in (run_dir / "best_params.yaml").read_text()   # untouched


# --------------------------------------------------------------------------
# §5 — one submission: fit -> test -> theta* sweep -> bench
# --------------------------------------------------------------------------
@needs_bash
@pytest.mark.parametrize("arm,env", [("rl3.sh", {}), ("rl4.sh", {"LRSR": "1e-4"}),
                                     ("rl2.sh", {"LRSR": "1e-4"})])
def test_one_submission_runs_the_whole_arm(tmp_path, arm, env):
    """No tune to be a first stage and no reason to queue twice for a bench that
    takes minutes next to the fit's hours."""
    got = _run_arm(tmp_path, arm, **env)
    assert got["STAGE"] == "fit"          # not the engine's `tune` default
    assert got["CHAIN_BENCH"] == "1"


@needs_bash
def test_an_explicit_stage_still_wins(tmp_path):
    """Re-benching an existing run dir on its own must stay possible — and must
    not re-enter the chain from the bench stage."""
    assert _run_arm(tmp_path, "rl3.sh", STAGE="bench")["STAGE"] == "bench"


# The engine's chain block, run as itself. The fit that precedes it needs a GPU
# and a dataset, so the block is extracted between its sentinels and exercised
# against a stand-in arm — the text under test is the engine's own.
CHAIN_START = "# >>> chain-bench"
CHAIN_END = "# <<< chain-bench"


def _chain_snippet() -> str:
    body = ENGINE.read_text().split(CHAIN_START)[1].split(CHAIN_END)[0]
    assert "exec env STAGE=bench" in body
    return body


@needs_bash
@pytest.mark.parametrize("env,expect_bench", [
    ({"CHAIN_BENCH": "1"}, True),
    ({}, True),                     # default since 2026-09-02: fit ends benched
    ({"CHAIN_BENCH": "0"}, False),                  # explicit opt-out
    ({"CHAIN_BENCH": "1", "SKIP_TEST": "1"}, False),
    ({"SKIP_TEST": "1"}, False),                    # ... and under the default
])
def test_the_chain_re_enters_the_arm_at_the_bench_stage(tmp_path, env, expect_bench):
    """SKIP_TEST=1 is the exception that matters: the bench reads the test split,
    which is exactly the split that flag exists to keep unseen. Chaining under it
    would spend a pilot's held-out set without anyone asking."""
    arm = tmp_path / "fake_arm.sh"
    arm.write_text('echo "BENCHED stage=${STAGE-unset}"\n')
    harness = tmp_path / "harness.sh"
    harness.write_text(f'ARM_SCRIPT="{arm}"\n' + _chain_snippet()
                       + 'echo "FIT ENDED WITHOUT CHAINING"\n')
    proc = subprocess.run(
        ["bash", str(harness)],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", **env},
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert ("BENCHED stage=bench" in proc.stdout) is expect_bench
    assert ("FIT ENDED WITHOUT CHAINING" in proc.stdout) is not expect_bench


def test_the_arm_path_is_absolute_before_anything_can_cd():
    """The fit stage cds into RUN_DIR, so a relative $0 would stop resolving —
    the chain would re-enter nothing. ARM_SCRIPT is therefore resolved at the top
    of the engine, above every cd in the file."""
    lines = ENGINE.read_text().splitlines()
    resolved = next(i for i, ln in enumerate(lines) if ln.startswith("ARM_SCRIPT="))
    cds = [i for i, ln in enumerate(lines) if ln.startswith("cd ")]
    assert cds and min(cds) > resolved


@needs_bash
@pytest.mark.parametrize("arm,env", [("rl3.sh", {}), ("rl4.sh", {"LRSR": "1e-4"}),
                                     ("rl2.sh", {"LRSR": "1e-4"})])
def test_a_stage_these_arms_cannot_honour_is_refused_loudly(tmp_path, arm, env):
    """SLURM exports the SUBMITTING shell's environment (--export=ALL by
    default), so a leftover `export STAGE=tune` in a login shell reaches the job
    and beats the arm's `${STAGE:-fit}`. Before this guard that bought a 1x1
    Optuna pass — one SR4RS epoch, ~20 min, producing an overlay the arm already
    carries — and then the job ended, looking like a successful run."""
    proc = subprocess.run(
        ["bash", str(HPC_RL / arm)],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path),
             "USER": "tester", "REPO_DIR": str(REPO), "STAGE": "tune", **env},
        capture_output=True, text=True)
    assert proc.returncode == 2
    assert "this arm has no such stage" in proc.stderr
    assert "OPTUNA" not in proc.stdout


@needs_bash
def test_rl2_is_the_SEN2SR_row_of_the_same_ladder(tmp_path):
    """The only thing separating rl2 from rl4 is the generator — same rungs,
    same hold, same everything else — which is what makes `rl2 vs rl4` a
    comparison of architectures rather than of recipes. It reads the SEN2SR-Lite
    model dir, not SR4RS's, and its rows merge with the Studio's rl2 rows (no
    platform tag, no _es tag)."""
    rl2 = _run_arm(tmp_path, "rl2.sh", LRSR="1e-3")
    rl4 = _run_arm(tmp_path, "rl4.sh", LRSR="1e-3")
    assert rl2["UPSAMPLER"] == "sen2sr" and rl2["FREEZE_SR"] == "false"
    assert rl2["SEN2SR_DIR"].endswith("/models/SEN2SRLite_RGBN")
    assert (rl2["SR_HC"], rl2["SR_PAD"]) == ("off", "0")
    assert rl2["FIT_EARLY_STOP"] == "0"       # its trajectory IS the result
    assert rl2["EXP_TAG"] == "rl2_new_ls1e-3"
    # identical treatment except the generator and where its weights live
    shared = {k: v for k, v in rl4.items() if k not in
              ("EXP_TAG", "UPSAMPLER", "SEN2SR_DIR", "SR4RS_GRAD_CKPT")}
    assert {k: rl2[k] for k in shared} == shared
