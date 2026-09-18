"""Unit tests for the rl-series HARD HOLD on the SR learning rate
(docs/rl_lightning_campaign_plan.md §2).

The campaign's fairness argument is structural, not statistical: a joint arm's
first ``sr_hold_epochs`` epochs must BE a frozen-arm run, so that the joint and
frozen branches diverge only at the hold boundary and the frozen arm's
remaining epochs are the matched-budget control. "Held" therefore has to mean
lr_sr == 0.0 EXACTLY — a 1e-9 leak would still be adaptation, just slow, and
the whole "30 = 30, no inheritance, no stage pairing" claim would be a
rounding error.

What is pinned here:

  * §2  the gate: lr_sr is identically 0 for every step of the hold, non-zero
        on the first step after it, and the head/U-Net group is UNTOUCHED
        throughout (that is what makes the hold phase reproduce the frozen
        twin's curve — registered prediction §6.1);
  * §2  the ramp: after the boundary the SR group runs its OWN cosine over the
        remaining budget, peaking at the full lr_sr and completing at 0, with
        ``sr_warmup_epochs`` re-based to the boundary;
  * back-compat: ``sr_hold_epochs=0`` (the default, i.e. every arm already in
        the append-only store) reproduces the pre-hold schedule bit-for-bit;
  * the auto-off: a frozen/bicubic front-end has no SR group to hold;
  * §2  the ladder-pinning path: ``LR_SR_MIN == LR_SR_MAX`` must make Optuna
        suggest that exact constant, since the rungs are pinned, never searched.

Everything runs on the parameter-free ``bicubic`` upsampler with a stand-in
trainable generator spliced in, so no SR weights, no dataset and no network
access are needed. The schedule is read straight off the LambdaLR rather than
through a real fit: what is under test is ``configure_optimizers``' arithmetic.
"""
from __future__ import annotations

import pytest
import torch

from sr.model import JointSRUNetLightning

NORM_MEAN = [0.10, 0.12, 0.14, 0.20]
NORM_STD = [0.04, 0.05, 0.06, 0.07]
BANDS = (1, 2, 3, 4)
C, P, UP = 4, 16, 4

# The campaign's shape: 30 epochs, hold 10, ramp over the remaining 20.
EPOCHS, HOLD, STEPS_PER_EPOCH = 30, 10, 8
TOTAL = EPOCHS * STEPS_PER_EPOCH
LR, LR_SR = 5.0e-3, 1.0e-4


class _StubTrainer:
    """The two attributes ``configure_optimizers`` reads off the trainer."""

    def __init__(self, max_epochs=EPOCHS, total=TOTAL):
        self.max_epochs = max_epochs
        self.estimated_stepping_batches = total


def make_model(trainable_sr=True, **kw):
    kwargs = dict(
        encoder_name="resnet18",
        encoder_weights=None,           # never hits the network
        classes=1,
        bands=BANDS,
        in_channels=C,
        norm_mean=NORM_MEAN,
        norm_std=NORM_STD,
        upsampler="bicubic",            # no SR weights to download
        upscale=UP,
        reflectance_scale=1.0,
        image_size=P * UP,
        lr=LR,
        lr_sr=LR_SR,
        lr_schedule="cosine",
    )
    kwargs.update(kw)
    model = JointSRUNetLightning(**kwargs)
    if trainable_sr:
        # Stand in for SEN2SR/SR4RS: all configure_optimizers wants from the
        # generator is a non-empty list of trainable parameters. Splicing one
        # in keeps the test free of model weights while exercising the real
        # two-group path.
        model.sr = torch.nn.Conv2d(C, C, 1)
        model._n_sr_p0 = len(list(model.sr.parameters()))
        # __init__ zeroes the hold when the front-end has no gradients, which
        # bicubic does not — restore the requested value now that it does.
        model._sr_hold_epochs = float(kwargs.get("sr_hold_epochs", 0.0))
        model._sr_warmup_epochs = float(kwargs.get("sr_warmup_epochs", 1.0))
    model._trainer = _StubTrainer()
    return model


def lr_trace(model):
    """Per-step (head_lr, sr_lr) over the whole budget, as the fit would see."""
    out = model.configure_optimizers()
    opt, sched = out["optimizer"], out["lr_scheduler"]["scheduler"]
    trace = []
    for _ in range(TOTAL):
        trace.append(tuple(g["lr"] for g in opt.param_groups))
        sched.step()
    return trace


# --------------------------------------------------------------------------- #
# §2  the gate: EXACTLY zero, and only for the hold
# --------------------------------------------------------------------------- #
def test_lr_sr_is_exactly_zero_through_the_hold_and_nonzero_after():
    # sr_warmup_epochs=0 so the boundary is the gate alone: with the default
    # 1-epoch ramp the first joint step is legitimately 0 too (ramp factor 0/8),
    # which is the re-basing tested separately below.
    trace = lr_trace(make_model(sr_hold_epochs=HOLD, sr_warmup_epochs=0.0))
    boundary = HOLD * STEPS_PER_EPOCH

    held = [sr for _, sr in trace[:boundary]]
    assert held == [0.0] * boundary, "lr_sr must be identically 0 during the hold"
    # Not "close to zero" — exactly, including the last held step.
    assert all(sr == 0.0 for sr in held)

    # Epoch 11 (0-based epoch 10) is where the branches diverge.
    assert trace[boundary][1] > 0.0


def test_the_head_group_is_untouched_by_the_hold():
    """Registered prediction §6.1: a joint arm's hold phase must reproduce the
    frozen arm's curve, which it can only do if the head sees the same LR
    schedule in both. The hold is an SR-group gate, not a run-wide pause."""
    held = lr_trace(make_model(sr_hold_epochs=HOLD))
    frozen = lr_trace(make_model(trainable_sr=False))
    assert [h for h, _ in held] == [f[0] for f in frozen]


# --------------------------------------------------------------------------- #
# §2  the ramp: the SR group's own cosine over the remaining budget
# --------------------------------------------------------------------------- #
def test_ramp_peaks_at_full_lr_sr_and_completes_at_zero():
    # sr_warmup_epochs=0 isolates the cosine from the (re-based) linear ramp.
    trace = lr_trace(make_model(sr_hold_epochs=HOLD, sr_warmup_epochs=0.0))
    boundary = HOLD * STEPS_PER_EPOCH

    # Peak = the rung's nominal lr_sr, at the boundary itself.
    assert trace[boundary][1] == pytest.approx(LR_SR, rel=1e-12)
    # Monotone decay from there, ending at (numerically) 0 — the dose is the
    # area under this, which is what the ladder's rungs are compared on.
    joint = [sr for _, sr in trace[boundary:]]
    assert all(a >= b for a, b in zip(joint, joint[1:]))
    assert joint[-1] == pytest.approx(0.0, abs=LR_SR * 1e-3)
    # Half-way through the JOINT phase a cosine is at half its peak. If the
    # schedule were instead the run-wide cosine merely gated to zero, this
    # would read ~0.25 * LR_SR.
    mid = joint[len(joint) // 2]
    assert mid == pytest.approx(LR_SR / 2, rel=0.05)


def test_warmup_ramp_is_rebased_to_the_hold_boundary():
    """A hold of 10 epochs then a 1-epoch ramp must ramp over epoch 11, not
    over epoch 1 (where it would have been consumed inside the hold)."""
    trace = lr_trace(make_model(sr_hold_epochs=HOLD, sr_warmup_epochs=1.0))
    boundary = HOLD * STEPS_PER_EPOCH
    # First joint step: ramp factor 0/warm == 0.
    assert trace[boundary][1] == 0.0
    # ...then climbing through the ramp epoch...
    ramp = [sr for _, sr in trace[boundary:boundary + STEPS_PER_EPOCH]]
    assert all(a <= b for a, b in zip(ramp, ramp[1:]))
    # ...and past the ramp it is a decaying cosine, well clear of zero.
    assert trace[boundary + STEPS_PER_EPOCH][1] > 0.5 * LR_SR


# --------------------------------------------------------------------------- #
# back-compat: hold 0 is the pre-2026-08-29 schedule, bit-for-bit
# --------------------------------------------------------------------------- #
def test_hold_zero_reproduces_the_legacy_schedule_exactly():
    import math

    trace = lr_trace(make_model(sr_hold_epochs=0.0, sr_warmup_epochs=1.0))
    warm = STEPS_PER_EPOCH  # 1.0 epochs, the recipe-v2 default

    for step, (head, sr) in enumerate(trace):
        cos = 0.5 * (1.0 + math.cos(math.pi * min(step, TOTAL) / TOTAL))
        assert head == pytest.approx(LR * cos, rel=1e-12)
        assert sr == pytest.approx(LR_SR * min(1.0, step / warm) * cos, rel=1e-12)


# --------------------------------------------------------------------------- #
# auto-off and the mis-set guard
# --------------------------------------------------------------------------- #
def test_hold_is_auto_disabled_when_there_is_no_sr_group():
    """rl0/rl1/rl3 pass the same hold as their joint twins for uniformity; a
    front-end with no gradients must ignore it rather than error."""
    model = JointSRUNetLightning(
        encoder_name="resnet18", encoder_weights=None, classes=1, bands=BANDS,
        in_channels=C, norm_mean=NORM_MEAN, norm_std=NORM_STD,
        upsampler="bicubic", upscale=UP, reflectance_scale=1.0,
        image_size=P * UP, lr_schedule="cosine", sr_hold_epochs=HOLD)
    assert model._sr_hold_epochs == 0.0
    model._trainer = _StubTrainer()
    # One group only, and it is the head's — no zero-LR group to explain away.
    out = model.configure_optimizers()
    assert len(out["optimizer"].param_groups) == 1


def test_hold_longer_than_the_budget_degrades_to_a_frozen_run_not_a_crash():
    """A mis-set hold is a legible result ('lr_sr was 0 all run'), never a
    division by zero that takes out an unattended overnight job."""
    trace = lr_trace(make_model(sr_hold_epochs=EPOCHS + 5))
    assert [sr for _, sr in trace] == [0.0] * TOTAL


# --------------------------------------------------------------------------- #
# §2  the ladder-pinning path: LR_SR_MIN == LR_SR_MAX is a constant, not a band
# --------------------------------------------------------------------------- #
def test_pinned_lr_sr_band_suggests_the_constant():
    """Every rung of the ladder is pinned by setting the search band's two ends
    equal (the pos_weight / r2grid pattern), so that the pinned value lands in
    best_params.yaml exactly like a searched one and the fit stage needs no
    special case. If Optuna ever stopped honouring a degenerate log-uniform
    band, every rung would silently collapse onto one dose."""
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    seen = []

    def objective(trial):
        seen.append(trial.suggest_float("lr_sr", 1e-4, 1e-4, log=True))
        return 0.0

    optuna.create_study().optimize(objective, n_trials=5)
    assert seen == [1e-4] * 5
