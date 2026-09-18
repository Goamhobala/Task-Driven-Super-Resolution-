"""Unit tests for the adaptive post-SR normalisation (docs/adaptive_norm_plan.md).

The post-SR z-score in :class:`sr.model.JointSRUNetLightning` uses frozen
dataset statistics, which are only valid while the SR output distribution stays
put. These tests pin the three things that make the adaptive version safe to
turn on:

  * §6.1a  flag OFF is byte-identical to the frozen-stats behaviour — every
           existing arm and benchmark row must stay comparable;
  * §4.1-4.3  the EMA tracks a shifted distribution at exactly the nominal rate,
           skips non-finite batches, never updates outside training, and RAISES
           (never silently clamps) when the std leaves its hard band;
  * §4.4-4.5  the exact recalibration reproduces population moments, and the
           adapted buffers survive a checkpoint round trip and a warm start.

Everything runs on the parameter-free ``bicubic`` upsampler so no SR weights,
no dataset and no network access are needed.

Note the split: gating (training vs eval, warmup, flag off) is exercised
through ``forward``, while the EMA arithmetic and the std guard are driven by
calling ``_adapt_update`` with a synthetic pre-normalisation tensor. Going
through bicubic for the arithmetic would make the assertions depend on how much
variance the interpolation kernel removes from the input — a property of
``F.interpolate``, not of the code under test.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from sr.model import AdaptiveNormBandExit, JointSRUNetLightning

# Full-stack stats (1-based bands 1-4 -> indices 0-3), deliberately NOT equal
# to the moments of any test tensor, so a stale adapter is visible.
NORM_MEAN = [0.10, 0.12, 0.14, 0.20]
NORM_STD = [0.04, 0.05, 0.06, 0.07]
BANDS = (1, 2, 3, 4)
C, P, UP = 4, 16, 4
HR = P * UP


def make_model(**kw):
    """Bicubic-front-end JointSR model on CPU: no SR weights, no downloads."""
    kwargs = dict(
        encoder_name="resnet18",
        encoder_weights=None,          # never hits the network
        classes=1,
        bands=BANDS,
        in_channels=C,
        norm_mean=NORM_MEAN,
        norm_std=NORM_STD,
        upsampler="bicubic",
        upscale=UP,
        reflectance_scale=1.0,         # ROSA V2 COGs already store reflectance
        image_size=HR,
    )
    kwargs.update(kw)
    return JointSRUNetLightning(**kwargs)


def lr_batch(b=2, mean=0.3, std=0.05, seed=0):
    """A low-resolution model input, (B, C, P, P)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, C, P, P, generator=g) * std + mean


def hr_tensor(b=4, mean=0.3, std=0.05, seed=0):
    """A synthetic PRE-NORMALISATION SR output, with exactly known moments."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(b, C, HR, HR, generator=g)
    # Standardise per band, then impose the requested moments exactly, so the
    # assertions never depend on sampling noise.
    m = y.mean(dim=(0, 2, 3), keepdim=True)
    s = y.std(dim=(0, 2, 3), unbiased=False, keepdim=True)
    return (y - m) / s * std + mean


def bicubic_up(x):
    """The exact call ``BicubicUpsampler`` makes."""
    return torch.nn.functional.interpolate(
        x, scale_factor=UP, mode="bicubic", antialias=True, align_corners=False)


class _TensorBatches(Dataset):
    """(image, mask, name) triples, like the real JointSR loaders."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, i):
        return self.tensors[i], torch.zeros(1, HR, HR), f"t{i}.png"


def loader_of(tensors, batch_size=1):
    return DataLoader(_TensorBatches(tensors), batch_size=batch_size)


# --------------------------------------------------------------------------- #
# §6.1a  flag off == legacy behaviour
# --------------------------------------------------------------------------- #
def test_flag_off_is_bit_identical_and_never_moves_the_buffers():
    model = make_model()
    assert model.hparams.adaptive_norm is False
    model.train()
    x = lr_batch(mean=5.0, std=3.0)       # wildly off the frozen stats
    mean0 = model.band_mean.clone()
    std0 = model.band_std.clone()

    with torch.no_grad():
        got = model(x)
        # The exact expression the pre-change forward computed.
        want = model.model((bicubic_up(x) * 1.0 - mean0) / std0)

    assert torch.equal(got, want)
    assert torch.equal(model.band_mean, mean0)
    assert torch.equal(model.band_std, std0)


def test_eval_mode_never_updates_even_with_the_flag_on():
    """SRPredictor, viz replays and the Lightning val loop must all consume
    frozen-at-that-moment buffers."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=0)
    model.eval()
    mean0 = model.band_mean.clone()
    with torch.no_grad():
        model(lr_batch(mean=5.0))
    assert torch.equal(model.band_mean, mean0)


def test_training_forward_does_update_when_enabled():
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=0.5,
                       adaptive_norm_check_every=0)
    model.train()
    mean0 = model.band_mean.clone()
    with torch.no_grad():
        model(lr_batch(mean=0.5, std=0.02))
    assert not torch.equal(model.band_mean, mean0)


def test_warmup_steps_delay_the_ema():
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_warmup_steps=5,
                       adaptive_norm_check_every=0)
    model.train()
    mean0 = model.band_mean.clone()
    with torch.no_grad():
        model(lr_batch(mean=0.9))         # global_step == 0 < 5
    assert torch.equal(model.band_mean, mean0)


# --------------------------------------------------------------------------- #
# §4.1-4.2  the EMA tracks at the nominal rate
# --------------------------------------------------------------------------- #
def test_ema_moves_toward_a_shifted_distribution_at_rate_m():
    m, steps = 0.25, 8
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=m,
                       adaptive_norm_check_every=0)   # band check off: the
    y = hr_tensor(mean=0.6, std=0.05, seed=7)         # shift is intentional
    target = y.mean(dim=(0, 2, 3))
    init = model.band_mean.reshape(-1).clone()

    for _ in range(steps):
        model._adapt_update(y)

    # Closed form for a constant target: mean_k = init + (target-init)(1-(1-m)^k)
    expect = init + (target - init) * (1.0 - (1.0 - m) ** steps)
    got = model.band_mean.reshape(-1)
    assert torch.allclose(got, expect, atol=1e-6), f"{got} vs {expect}"
    # Strictly between the two: tracking, not snapping, and not stuck.
    assert torch.all((got - init).abs() > 0)
    assert torch.all((got - target).abs() > 0)


def test_std_is_derived_from_the_moments_not_ema_d_directly():
    """With m=1 the buffers must equal this batch's exact moments."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=0)
    y = hr_tensor(mean=0.30, std=0.045, seed=3)
    model._adapt_update(y)
    assert torch.allclose(model.band_mean.reshape(-1), y.mean(dim=(0, 2, 3)),
                          atol=1e-6)
    assert torch.allclose(model.band_std.reshape(-1),
                          y.std(dim=(0, 2, 3), unbiased=False), atol=1e-6)


def test_ema_state_survives_many_steps_without_drifting_off():
    """E[x^2] is reconstructed from (mean, std) every step rather than stored.
    Over a long run that round trip must not accumulate error."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=0.01,
                       adaptive_norm_check_every=0)
    y = hr_tensor(mean=0.30, std=0.045, seed=23)
    for _ in range(2000):
        model._adapt_update(y)
    assert torch.allclose(model.band_mean.reshape(-1), y.mean(dim=(0, 2, 3)),
                          atol=1e-4)
    assert torch.allclose(model.band_std.reshape(-1),
                          y.std(dim=(0, 2, 3), unbiased=False), atol=1e-4)


def test_non_finite_batch_is_skipped_and_counted():
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=0.5,
                       adaptive_norm_check_every=0)
    y = hr_tensor(mean=0.30, std=0.045, seed=9)
    y[0, 0, 0, 0] = float("nan")
    mean0 = model.band_mean.clone()
    std0 = model.band_std.clone()
    model._adapt_update(y)
    assert torch.equal(model.band_mean, mean0), "NaN batch must not move the EMA"
    assert torch.equal(model.band_std, std0)
    assert float(model._adapt_skips) == 1.0


def test_lag_diagnostic_separates_the_fast_and_slow_emas():
    """`adapt_lag_max` is the gap between the slow (forward-path) EMA and a
    fast EMA of the raw batch moments — the plan's staleness readout."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=0.01,
                       adaptive_norm_check_every=0)
    y = hr_tensor(mean=0.60, std=0.05, seed=29)
    for _ in range(5):
        model._adapt_update(y)
    slow = model.band_mean.reshape(-1)
    fast = model._adapt_fast_mean
    target = y.mean(dim=(0, 2, 3))
    # The fast EMA is much closer to the target -> a non-trivial lag reading.
    assert torch.all((fast - target).abs() < (slow - target).abs())


# --------------------------------------------------------------------------- #
# §4.3  the std hard band fails loud — it is not a clamp
# --------------------------------------------------------------------------- #
def test_a_band_exit_NEVER_kills_a_fit_by_default(capsys):
    """THE regression test. On 2026-08-19 a raise-by-default band guard killed
    r4b_new at epoch 13 of 100, after a day in the SLURM queue, at lr_sr 13x
    BELOW the design default, on a decelerating trend that crossed the bound by
    0.004. Measuring what the task loss does to the SR generator IS the
    experiment; the instrument must never abort it."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1)
    assert model.hparams.std_band_action == "warn"
    model._adapt_update(hr_tensor(mean=0.30, std=0.01, seed=12))   # 0.25x
    out = capsys.readouterr().out
    assert "BAND EXIT (non-fatal, training continues)" in out
    assert model._adapt_band_exited is True
    # ...and it keeps tracking afterwards rather than wedging.
    model._adapt_update(hr_tensor(mean=0.30, std=0.045, seed=3))
    assert float(model.band_std.reshape(-1)[0]) == pytest.approx(0.045, rel=1e-3)


def test_band_exit_warns_only_once_but_the_metric_stays_latched():
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1)
    for _ in range(3):
        model._adapt_update(hr_tensor(mean=0.30, std=0.01, seed=12))
    assert model._adapt_band_exited is True


def test_std_collapse_raises_only_when_explicitly_asked():
    """`raise` is opt-in, for sr.tune, where pruning a bad corner saves budget."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1, std_band_action="raise")
    y = hr_tensor(mean=0.30, std=0.01, seed=12)   # 0.25x band 0's 0.04
    with pytest.raises(AdaptiveNormBandExit, match="COLLAPSED"):
        model._adapt_update(y)


def test_rejects_an_unknown_band_action():
    with pytest.raises(ValueError, match="std_band_action"):
        make_model(std_band_action="explode")


def test_resume_rebases_the_band_reference_instead_of_re_tripping(tmp_path):
    """Second half of the 2026-08-19 failure: `_adapt_init_*` are
    non-persistent, so a RESUME_FIT would restore the ADAPTED band_std while
    the reference still held the config's dataset stats — and the first check
    after the resume would re-trip on drift the resumed segment never caused.
    Without this, the run could not even be restarted."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1, std_band_action="raise")
    model._adapt_update(hr_tensor(mean=0.30, std=0.022, seed=4))   # ~0.55x
    p = tmp_path / "last.pt"
    torch.save(model.state_dict(), p)

    resumed = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                         adaptive_norm_check_every=1, std_band_action="raise")
    resumed.load_state_dict(torch.load(p, weights_only=True), strict=True)
    # Reference now equals the restored state, so the segment starts at 1.0x.
    assert torch.allclose(resumed._adapt_init_std, resumed.band_std)
    resumed._check_std_band()          # must NOT raise


def test_growth_past_the_old_symmetric_bound_is_tolerated():
    """Regression for 2026-08-13: a symmetric 2.0x upper bound killed a tune on
    a SEN2SR trial whose std was growing (means DC-pinned and steady). Growth is
    self-stabilising — rising std LOWERS the gradient gain — so 2x now warns."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1)
    y = hr_tensor(mean=0.30, std=0.088, seed=11)      # 2.2x band 0's 0.04
    model._adapt_update(y)                            # must NOT raise
    assert float(model.band_std.reshape(-1)[0]) == pytest.approx(0.088, rel=1e-3)


def test_extreme_growth_still_raises():
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1)
    y = hr_tensor(mean=0.30, std=0.30, seed=11)       # 7.5x band 0's 0.04
    with pytest.raises(AdaptiveNormBandExit, match="GREW"):
        model._adapt_update(y)
    # Diagnostics, not a silent clamp: the buffer still holds the real value.
    assert float(model.band_std.reshape(-1)[0]) == pytest.approx(0.30, rel=1e-3)


def test_band_limits_are_hparams():
    """An arm with a known reason to expect contrast growth can loosen the
    upper side WITHOUT touching the lower catastrophe bound."""
    loose = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1, std_band_raise_hi=20.0)
    loose._adapt_update(hr_tensor(mean=0.30, std=0.30, seed=11))   # no raise
    strict = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                        adaptive_norm_check_every=1, std_band_raise_lo=0.95)
    with pytest.raises(AdaptiveNormBandExit, match="COLLAPSED"):
        strict._adapt_update(hr_tensor(mean=0.30, std=0.035, seed=12))


def test_band_exit_is_catchable_without_catching_every_runtime_error():
    """`sr.tune` prunes the trial on this type specifically; it must not be so
    broad that a genuine bug is silently swallowed as 'bad hyperparameters'."""
    assert issubclass(AdaptiveNormBandExit, RuntimeError)
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1)
    try:
        model._adapt_update(hr_tensor(mean=0.30, std=0.01, seed=12))
    except AdaptiveNormBandExit as exc:
        assert len(exc.ratios) == C            # per-band diagnostics survive
        assert exc.band == (0.5, 4.0)
        assert "lr_sr" in str(exc)             # points at the first thing to check
    else:
        pytest.fail("expected AdaptiveNormBandExit")


def test_std_inside_the_warn_band_warns_once_and_continues(capsys):
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1)
    # 0.064 = 1.6x band 0's 0.04: inside the raise band [0.5, 4.0], outside
    # the warn band [0.7, 1.5] -> warn, keep training.
    y = hr_tensor(mean=0.30, std=0.064, seed=13)
    model._adapt_update(y)
    model._adapt_update(y)
    out = capsys.readouterr().out
    assert out.count("WARN adaptive_norm") == 1, "warn must fire exactly once"


def test_check_every_zero_disables_the_guard_everywhere():
    """Including the val-epoch-end path, where the traceback would be far less
    legible than one raised from the training step."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=0)
    model._adapt_update(hr_tensor(mean=0.30, std=0.20, seed=11))  # no raise
    model._log_adapt_stats()                                      # no raise


def test_variance_floor_warns_that_the_reconstruction_is_no_longer_exact(capsys):
    """The floor is the one path that breaks m2 = std^2 + mean^2, which the
    whole no-extra-buffer design rests on. It must never pass silently."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=1)
    y = torch.full((2, C, HR, HR), 0.02)      # a collapsed (constant) band
    with pytest.raises(RuntimeError, match="hard band"):
        model._adapt_update(y)
    assert "variance floor engaged" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# §4.4  exact recalibration
# --------------------------------------------------------------------------- #
def test_recalibration_reproduces_population_moments():
    model = make_model()
    tensors = [lr_batch(b=1, mean=0.35, std=0.06, seed=s)[0] for s in range(5)]
    stats = model.recalibrate_norm_stats(dataloader=loader_of(tensors),
                                         n_batches=len(tensors))

    with torch.no_grad():
        hr = torch.cat([bicubic_up(t[None]) for t in tensors])
    assert torch.allclose(model.band_mean.reshape(-1), hr.mean(dim=(0, 2, 3)),
                          atol=1e-5)
    assert torch.allclose(model.band_std.reshape(-1),
                          hr.std(dim=(0, 2, 3), unbiased=False), atol=1e-5)
    assert stats["batches"] == len(tensors)
    assert stats["pixels_per_band"] == len(tensors) * HR ** 2


def test_recalibration_honours_the_batch_budget():
    model = make_model()
    tensors = [lr_batch(b=1, seed=s)[0] for s in range(6)]
    stats = model.recalibrate_norm_stats(dataloader=loader_of(tensors), n_batches=2)
    assert stats["batches"] == 2


def test_recalibration_restores_training_mode():
    model = make_model()
    model.train()
    model.recalibrate_norm_stats(dataloader=loader_of([lr_batch(b=1)[0]]),
                                 n_batches=1)
    assert model.training


def test_recalibration_reseeds_the_lag_diagnostic():
    """Otherwise the first post-recalibration `adapt_lag_max` would read a
    spurious step change rather than real staleness."""
    model = make_model()
    model.recalibrate_norm_stats(dataloader=loader_of([lr_batch(b=1, mean=0.9)[0]]),
                                 n_batches=1)
    assert torch.allclose(model._adapt_fast_mean, model.band_mean.reshape(-1))


@pytest.mark.parametrize(
    "mode,expected",
    [("off", "off"), ("pre", "pre"), ("post", "post"), ("auto", "off")],
)
def test_explicit_modes_pass_through_and_auto_spares_r0(mode, expected):
    """Under `auto`, R0 (bicubic) stays the untouched deterministic baseline."""
    assert make_model(upsampler="bicubic",
                      norm_recalibrate=mode)._resolve_recalibrate() == expected


@pytest.mark.parametrize(
    "upsampler,n_sr_params,adaptive,expected",
    [
        # Frozen SR with parameters (r1/r5): stationary output stats, and the
        # adapter is stale from step 0 -> one recompute BEFORE fitting.
        ("sr4rs", 0, False, "pre"),
        ("sen2sr", 0, True, "pre"),
        # Trainable SR + EMA (r4/r6, the target arms): clean the EMA's residual
        # lag out of the shipped checkpoint at the end.
        ("sr4rs", 12, True, "post"),
        ("sen2sr", 12, True, "post"),
        # Trainable SR without the EMA: legacy recipe, leave it alone.
        ("sr4rs", 12, False, "off"),
    ],
)
def test_auto_resolution_per_arm(upsampler, n_sr_params, adaptive, expected):
    """The `auto` branches for the real arms. The SR weights are not available
    in a unit test, so the two properties `_resolve_recalibrate` reads —
    upsampler name and trainable-SR-parameter count — are set directly."""
    model = make_model(norm_recalibrate="auto", adaptive_norm=adaptive)
    model.hparams.upsampler = upsampler
    model._n_sr_p0 = n_sr_params
    assert model._resolve_recalibrate() == expected


def test_post_recalibration_runs_exactly_once():
    """`post` must not re-run if on_train_end fires twice (or after a manual
    call) — the second pass would re-measure statistics the first already
    installed."""
    model = make_model(norm_recalibrate="post")
    tensors = [lr_batch(b=1, mean=0.35, seed=s)[0] for s in range(3)]
    model.recalibrate_norm_stats(dataloader=loader_of(tensors), n_batches=3)
    model._recal_done = False                       # pretend it has not run
    assert model._maybe_post_recalibrate.__self__ is model

    calls = []
    model.recalibrate_norm_stats = lambda **kw: calls.append(kw) or {"batches": 0}
    model._val_iou_snapshot = lambda *a, **k: None
    assert model._maybe_post_recalibrate() is True
    assert model._maybe_post_recalibrate() is False
    assert len(calls) == 1


def test_non_post_modes_never_recalibrate_at_the_end():
    for mode in ("off", "pre"):
        model = make_model(norm_recalibrate=mode)
        model.recalibrate_norm_stats = lambda **kw: pytest.fail("must not run")
        assert model._maybe_post_recalibrate() is False


def test_rejects_an_unknown_recalibration_mode():
    with pytest.raises(ValueError, match="norm_recalibrate"):
        make_model(norm_recalibrate="sometimes")


def test_rejects_a_degenerate_momentum():
    with pytest.raises(ValueError, match="adaptive_norm_momentum"):
        make_model(adaptive_norm_momentum=0.0)


# --------------------------------------------------------------------------- #
# §4.1 / §4.5  persistence, resume, warm start
# --------------------------------------------------------------------------- #
def test_adapted_buffers_survive_a_state_dict_round_trip(tmp_path):
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=0)
    y = hr_tensor(mean=0.30, std=0.045, seed=5)
    model._adapt_update(y)
    adapted_mean = model.band_mean.clone()
    adapted_std = model.band_std.clone()
    assert not torch.allclose(adapted_mean.reshape(-1),
                              torch.tensor(NORM_MEAN)), "test is vacuous"

    p = tmp_path / "sd.pt"
    torch.save(model.state_dict(), p)
    restored = make_model(adaptive_norm=True, adaptive_norm_check_every=0)
    # Strict: the adaptive state adds NO new persistent tensors, so a legacy
    # checkpoint still loads and a new one carries the adapted values.
    restored.load_state_dict(torch.load(p, weights_only=True), strict=True)
    assert torch.equal(restored.band_mean, adapted_mean)
    assert torch.equal(restored.band_std, adapted_std)

    # E[x^2] is reconstructed from (mean, std), so the EMA must resume exactly
    # where it stopped. Checked at m<1, where the NEXT update genuinely depends
    # on the prior second moment (at m=1 the update discards it, which would
    # make this assertion pass even if the reconstruction were garbage).
    control = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                         adaptive_norm_check_every=0)
    control._adapt_update(y)                      # same state, never serialised
    y2 = hr_tensor(mean=0.50, std=0.030, seed=6)
    for mdl in (restored, control):
        mdl.hparams.adaptive_norm_momentum = 0.3
        mdl._adapt_update(y2)
    assert torch.allclose(restored.band_mean, control.band_mean, atol=1e-7)
    assert torch.allclose(restored.band_std, control.band_std, atol=1e-7)
    # ...and genuinely different from a model that restarted at dataset stats.
    fresh = make_model(adaptive_norm=True, adaptive_norm_momentum=0.3,
                       adaptive_norm_check_every=0)
    fresh._adapt_update(y2)
    assert not torch.allclose(restored.band_std, fresh.band_std, atol=1e-4)


def test_reference_buffers_are_the_config_stats_not_the_adapted_ones():
    """`_adapt_init_*` are non-persistent by design: the drift reference must
    stay the DATASET stats even when resuming an adapted checkpoint."""
    model = make_model(adaptive_norm=True, adaptive_norm_momentum=1.0,
                       adaptive_norm_check_every=0)
    model._adapt_update(hr_tensor(mean=0.30, std=0.045, seed=5))
    assert torch.allclose(model._adapt_init_mean.reshape(-1),
                          torch.tensor(NORM_MEAN), atol=1e-6)
    sd = model.state_dict()
    assert "_adapt_init_mean" not in sd
    assert "_adapt_init_std" not in sd
    assert "_adapt_fast_mean" not in sd
    assert "_adapt_skips" not in sd


def test_warm_start_adopts_the_stage_one_norm_buffers(tmp_path):
    stage1 = make_model()
    with torch.no_grad():                      # pretend stage 1 recalibrated
        stage1.band_mean.fill_(0.42)
        stage1.band_std.fill_(0.077)
    ckpt = tmp_path / "stage1.ckpt"
    torch.save({"state_dict": stage1.state_dict(),
                "hyper_parameters": dict(stage1.hparams),
                "epoch": 3}, ckpt)

    stage2 = make_model(warm_start_unet=str(ckpt))
    assert torch.allclose(stage2.band_mean, torch.full_like(stage2.band_mean, 0.42))
    assert torch.allclose(stage2.band_std, torch.full_like(stage2.band_std, 0.077))
    for a, b in zip(stage2.model.parameters(), stage1.model.parameters()):
        assert torch.equal(a, b)


def test_warm_start_from_a_checkpoint_without_norm_buffers(tmp_path):
    """The band_mean/band_std keys are stripped, so the copy branch is skipped
    and the config's dataset stats stand."""
    stage1 = make_model()
    sd = {k: v for k, v in stage1.state_dict().items()
          if k not in ("band_mean", "band_std")}
    ckpt = tmp_path / "no_buffers.ckpt"
    torch.save({"state_dict": sd, "hyper_parameters": dict(stage1.hparams),
                "epoch": 1}, ckpt)
    stage2 = make_model(warm_start_unet=str(ckpt))
    assert torch.allclose(stage2.band_mean.reshape(-1), torch.tensor(NORM_MEAN),
                          atol=1e-6)


def test_warm_start_rebases_the_guard_band_on_the_adopted_stats(tmp_path):
    """The hard band must measure movement from where THIS run starts. A
    stage-1 arm that legitimately recalibrated would otherwise hand stage 2 a
    step-0 'drift' it never caused — and for frozen SR4RS, whose output stats
    differ from the dataset stats by construction, that can trip the guard
    before a single gradient step."""
    stage1 = make_model()
    with torch.no_grad():
        stage1.band_mean.fill_(0.42)
        stage1.band_std.fill_(0.15)          # 3.75x band 0's dataset std
    ckpt = tmp_path / "stage1.ckpt"
    torch.save({"state_dict": stage1.state_dict(),
                "hyper_parameters": dict(stage1.hparams), "epoch": 3}, ckpt)

    stage2 = make_model(warm_start_unet=str(ckpt), adaptive_norm=True,
                        adaptive_norm_check_every=1)
    assert torch.allclose(stage2._adapt_init_std, stage2.band_std)
    stage2._check_std_band()                 # would raise against dataset stats


def test_resume_guard_rejects_freezing_an_adaptive_run(monkeypatch):
    """Continuing an adaptive_norm fit under adaptive_norm=false would freeze
    the statistics mid-trajectory while the UNet kept co-adapting."""
    model = make_model(adaptive_norm=False)
    ck = {"hyper_parameters": {"reflectance_scale": 1.0, "adaptive_norm": True}}

    # No trainer -> a test/viz restore: allowed (the buffers carry the values).
    model.on_load_checkpoint(ck)

    class _Trainer:
        class state:
            fn = "fit"

    monkeypatch.setattr(model, "_trainer", _Trainer(), raising=False)
    with pytest.raises(ValueError, match="adaptive_norm"):
        model.on_load_checkpoint(ck)


def test_resume_guard_allows_the_matching_configuration(monkeypatch):
    model = make_model(adaptive_norm=True)
    ck = {"hyper_parameters": {"reflectance_scale": 1.0, "adaptive_norm": True}}

    class _Trainer:
        class state:
            fn = "fit"

    monkeypatch.setattr(model, "_trainer", _Trainer(), raising=False)
    model.on_load_checkpoint(ck)


# --------------------------------------------------------------------------- #
# §4.3  functional drift monitor
# --------------------------------------------------------------------------- #
def test_functional_monitor_is_auto_skipped_without_trainable_sr_params():
    """Nothing evolves in a bicubic/frozen front-end, so there is nothing to
    measure drift against."""
    model = make_model(sr_functional_monitor=True)
    assert model._n_sr_p0 == 0
    assert model._functional_monitor_on() is False


def test_functional_monitor_reports_zero_drift_against_itself():
    model = make_model()
    x = lr_batch(b=1, mean=0.30, std=0.05, seed=17)
    with torch.no_grad():
        ref = model._sr_forward(x)
        cur = model._sr_forward(x)
    mse = float((cur - ref).pow(2).mean())
    assert mse == 0.0
    assert 10.0 * math.log10(1.0 / max(mse, 1e-20)) > 100.0


def test_functional_monitor_is_a_no_op_without_a_cached_reference():
    model = make_model()
    assert model._srmon_ref is None
    model._log_functional_drift()          # must not raise


def test_sr_forward_matches_the_full_forward_front_end():
    """The recalibration pass and the drift monitor must run EXACTLY the
    front-end training runs."""
    model = make_model()
    x = lr_batch(b=2, mean=0.30, std=0.05, seed=19)
    with torch.no_grad():
        y = model._sr_forward(x)
        got = model(x)
        want = model.model((y - model.band_mean) / model.band_std)
    assert torch.equal(got, want)
    assert torch.equal(y, bicubic_up(x))
