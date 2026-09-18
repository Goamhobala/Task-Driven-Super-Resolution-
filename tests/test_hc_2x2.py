"""Unit tests for the FFT hard constraint as a treatment (docs/hc_2x2_plan.md).

The 2x2 crosses the constraint with the generator — SEN2SR-Lite and SR4RS, each
with the constraint on and off — so the code had to learn to take the operator
OFF the generator that ships it and PUT it ON the one that does not. These
tests pin the parts of §6 that can be checked without cluster weights:

  * §6.1  `sr_hc='native'` reproduces each upsampler's previous forward exactly
          — the guard for the ten arms already in the benchmark store;
  * §6.2  `off` is the RAW generator (no clamp either — the bundle is one
          treatment, §4 "Scheme B") and `on` is the upstream formula, on
          EITHER generator;
  * §5.1  the resolution table, the rejected bicubic cell, the input-size pin
          moving with the mask, and the checkpoint guard that stops an arm
          being evaluated under the other column's operator;
  * §6.5  gradient reach through the constraint, including the documented trap
          that a spatial-mean probe reports a dead network.

§6.3 (negative-pixel fraction of the raw generators), §6.4 (the radiometric
downsample check) and §6.6 (the bs=4 memory check at 576 px) need the real
weights, real ROSA_New batches and a 44 GB GPU; they are cluster pre-flight,
not unit tests.

The generators here are throwaway convs: the code under test is the wiring, and
a real CNNSR/SR4RS would only add weight downloads to the assertion.
"""
from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from sr.model import JointSRUNetLightning
from sr.sen2sr_loader import (
    TrainableSEN2SR, hard_constraint_from_mask, pad_low_pass_mask,
    resolve_sr_hc)

C, P, UP = 4, 16, 4          # 4 bands, 16 px LR patch, x4 -> 64 px HR
HR = P * UP
NORM_MEAN = [0.10, 0.12, 0.14, 0.20]
NORM_STD = [0.04, 0.05, 0.06, 0.07]
BANDS = (1, 2, 3, 4)


class ToyGenerator(nn.Module):
    """A x4 upsampler whose output STRADDLES ZERO, so the clamp is observable.

    Signed weights, not the usual positive-definite toy: with an all-positive
    kernel on reflectance input every pixel is positive and "no clamp" would be
    a vacuous assertion. Some pixels still survive the clamp, so the gradient
    test below is not measuring a fully saturated ReLU either.

    Deterministic given the seed, and cheap: the tests care about which
    operators wrap it, not about what it draws.
    """

    def __init__(self, seed: int = 0, bias: float = 0.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv = nn.Conv2d(C, C, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(
                (torch.rand(self.conv.weight.shape, generator=g) - 0.5) * 0.4)
            self.conv.bias.fill_(bias)

    def forward(self, x):
        up = nn.functional.interpolate(x, scale_factor=UP, mode="nearest")
        return self.conv(up)


def toy_mask(side: int = HR, cutoff: float = 6.0) -> torch.Tensor:
    """A centred Gaussian low-pass mask, shaped like the shipped one (H, W)."""
    c = side // 2
    d2 = ((torch.arange(side) - c) ** 2)[:, None] + ((torch.arange(side) - c) ** 2)[None, :]
    return torch.exp(-d2.float() / (2 * cutoff ** 2))


def toy_hc(side: int = HR):
    return hard_constraint_from_mask(toy_mask(side))


def batch(seed: int = 1, p: int = P) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, C, p, p, generator=g) * 0.4 + 0.05


# ------------------------------------------------------------------- §5.1 flag
@pytest.mark.parametrize(
    "upsampler,mode,expected",
    [
        ("sen2sr", "native", True),
        ("sen2sr_full", "native", True),
        ("sr4rs", "native", False),
        ("bicubic", "native", False),
        ("sen2sr", "off", False),
        ("sr4rs", "on", True),
        ("bicubic", "off", False),
        ("sen2sr", None, True),          # missing hparam == native
    ],
)
def test_resolve_sr_hc_table(upsampler, mode, expected):
    assert resolve_sr_hc(upsampler, mode) is expected


def test_bicubic_with_hard_constraint_is_rejected():
    """HC(bicubic(x), bicubic(x)) is a near-identity — not a treatment."""
    with pytest.raises(ValueError, match="bicubic"):
        resolve_sr_hc("bicubic", "on")


def test_unknown_sr_hc_mode_raises():
    with pytest.raises(ValueError, match="sr_hc"):
        resolve_sr_hc("sen2sr", "yes")


# -------------------------------------------------- §6.1 / §6.2 the operator
def test_hc_on_is_the_upstream_formula():
    """§6.1: with the constraint mounted the forward is, exactly,
    `hard_constraint(x, clamp(sr_model(x), min=0))` — what every SEN2SR arm in
    the store was trained under."""
    x = batch()
    gen, hc = ToyGenerator(), toy_hc()
    wrapped = TrainableSEN2SR(gen, hc, clamp_min=0.0)
    expected = hc(x, torch.clamp(gen(x), min=0.0))
    assert torch.equal(wrapped(x), expected)


def test_hc_off_is_the_raw_generator():
    """§6.2: `off` drops the splice AND the clamp (§4 Scheme B), so the b
    column compares raw generator to raw generator."""
    x = batch()
    gen = ToyGenerator()
    bare = TrainableSEN2SR(gen, None, clamp_min=None)
    out = bare(x)
    assert torch.equal(out, gen(x))
    # The toy generator's negative bias makes the "no clamp" half of that
    # assertion load-bearing rather than vacuous.
    assert (out < 0).any()


def test_wrapper_defaults_do_not_silently_clamp():
    """A bare wrapper must never be built with half the bundle by accident."""
    x = batch()
    gen = ToyGenerator()
    assert torch.equal(TrainableSEN2SR(gen)(x), torch.clamp(gen(x), min=0.0))


def test_pad_low_pass_mask_refuses_without_a_constraint():
    with pytest.raises(ValueError, match="no hard constraint"):
        pad_low_pass_mask(TrainableSEN2SR(ToyGenerator(), None, clamp_min=None), 2)


# ------------------------------------------------------------ model wiring
@pytest.fixture
def fake_sr_loaders(monkeypatch):
    """Swap the weight-loading helpers for toy modules.

    Mirrors how `JointSRUNetLightning.__init__` reaches them: the SEN2SR loader
    is imported into `sr.model`'s namespace, the SR4RS one inside the branch
    from `sr.sr4rs_torch`, and `build_hard_constraint` reads a file we do not
    have here.
    """
    import sr.model as m
    import sr.sr4rs_torch as s4

    def fake_sen2sr(model_dir, hard_constraint=True):
        gen = ToyGenerator(seed=2)
        if not hard_constraint:
            return TrainableSEN2SR(gen, None, clamp_min=None)
        return TrainableSEN2SR(gen, toy_hc(), clamp_min=0.0)

    monkeypatch.setattr(m, "load_trainable_sen2sr", fake_sen2sr)
    monkeypatch.setattr(s4, "load_trainable_sr4rs", lambda d: ToyGenerator(seed=3))
    monkeypatch.setattr(m, "build_hard_constraint", lambda p: toy_hc())


def make_model(**kw):
    kwargs = dict(
        encoder_name="resnet18",
        encoder_weights=None,              # never hits the network
        classes=1,
        bands=BANDS,
        in_channels=C,
        norm_mean=NORM_MEAN,
        norm_std=NORM_STD,
        upscale=UP,
        reflectance_scale=1.0,
        image_size=HR,
        sen2sr_dir="/nonexistent",         # the fake loaders never read it
    )
    kwargs.update(kw)
    return JointSRUNetLightning(**kwargs)


def test_sen2sr_native_keeps_the_constraint_and_the_size_pin(fake_sr_loaders):
    model = make_model(upsampler="sen2sr")
    assert model._sr_hc_on is True
    assert model.sr.hard_constraint is not None
    assert model._required_lr == P              # mask side // upscale
    with pytest.raises(ValueError, match="pins the LR patch"):
        model._sr_forward(batch(p=P + 4))


def test_sen2sr_off_is_the_bare_generator(fake_sr_loaders):
    """r2b: no clamp, no splice — and no mask, so no structural size pin."""
    model = make_model(upsampler="sen2sr", sr_hc="off")
    assert model._sr_hc_on is False
    assert model.sr.hard_constraint is None
    assert model._required_lr is None
    x = batch()
    assert torch.equal(model._sr_forward(x), model.sr.sr_model(x))
    # No `hard_constraint.*` keys: a wrong-config restore fails the strict load.
    assert not any(k.startswith("sr.hard_constraint")
                   for k in model.state_dict())


def test_sr4rs_native_is_unwrapped(fake_sr_loaders):
    model = make_model(upsampler="sr4rs")
    assert model._sr_hc_on is False
    assert not isinstance(model.sr, TrainableSEN2SR)
    assert model._required_lr is None


def test_sr4rs_on_mounts_the_same_operator(fake_sr_loaders):
    """r4a: the constraint is architecture-agnostic, so the SR4RS branch gains
    the wrapper, the input-size pin and the pad-grown mask that the SEN2SR
    branch has always had."""
    model = make_model(upsampler="sr4rs", sr_hc="on",
                       hc_mask_path="/nonexistent/hard_constraint.safetensor")
    assert model._sr_hc_on is True
    assert isinstance(model.sr, TrainableSEN2SR)
    assert model._required_lr == P
    x = batch()
    gen = model.sr.sr_model
    expected = model.sr.hard_constraint(x, torch.clamp(gen(x), min=0.0))
    assert torch.equal(model._sr_forward(x), expected)


def test_sr4rs_on_without_a_mask_path_raises(fake_sr_loaders):
    with pytest.raises(ValueError, match="hc_mask_path"):
        make_model(upsampler="sr4rs", sr_hc="on")


def test_sr4rs_on_grows_the_mask_for_the_pad(fake_sr_loaders):
    """SR_PAD travels with the constraint, so r4a runs the padded grid: the
    mask must grow with it or the FFT sizes disagree."""
    pad = 2
    model = make_model(upsampler="sr4rs", sr_hc="on", sr_pad=pad,
                       hc_mask_path="/nonexistent/hard_constraint.safetensor")
    assert model._required_lr == P              # computed BEFORE the grow
    assert model.sr.hard_constraint.low_pass_mask.shape[-1] == HR + 2 * pad * UP
    out = model._sr_forward(batch())            # would throw on a size mismatch
    assert out.shape[-1] == HR


def test_bicubic_with_hc_on_is_rejected_at_construction(fake_sr_loaders):
    with pytest.raises(ValueError, match="bicubic"):
        make_model(upsampler="bicubic", sr_hc="on")


# ------------------------------------------------------------ ckpt guard
def _ckpt(**hp):
    base = {"reflectance_scale": 1.0, "upsampler": "sen2sr"}
    base.update(hp)
    return {"hyper_parameters": base}


def test_legacy_checkpoint_without_sr_hc_loads_native(fake_sr_loaders):
    """Old checkpoints carry no `sr_hc` key at all — they must resolve to the
    behaviour they were trained under, not to a mismatch."""
    make_model(upsampler="sen2sr").on_load_checkpoint(_ckpt())


def test_checkpoint_guard_catches_a_swapped_column(fake_sr_loaders):
    model = make_model(upsampler="sen2sr", sr_hc="off")
    with pytest.raises(ValueError, match="sr_hc mismatch"):
        model.on_load_checkpoint(_ckpt())                     # r2a ckpt, r2b config
    model.on_load_checkpoint(_ckpt(sr_hc="off"))              # its own arm is fine


def test_checkpoint_guard_resolves_native_per_upsampler(fake_sr_loaders):
    """'native' means ON for sen2sr and OFF for sr4rs, so the comparison has to
    resolve both sides rather than compare the raw strings."""
    model = make_model(upsampler="sr4rs", sr_hc="on",
                       hc_mask_path="/nonexistent/hard_constraint.safetensor")
    with pytest.raises(ValueError, match="sr_hc mismatch"):
        model.on_load_checkpoint(_ckpt(upsampler="sr4rs"))    # r4b ckpt (native=off)
    model.on_load_checkpoint(_ckpt(upsampler="sr4rs", sr_hc="on"))


# ---------------------------------------------------------- §6.5 gradients
def test_gradients_reach_the_generator_through_the_constraint():
    """A spatially varying loss flows into the generator. The trap this guards:
    the constraint takes the DC bin entirely from the input, so an
    `output.mean()` smoke probe has EXACTLY zero gradient and would report a
    dead network."""
    x = batch()
    gen, hc = ToyGenerator(), toy_hc()
    wrapped = TrainableSEN2SR(gen, hc, clamp_min=0.0)

    ramp = torch.linspace(-1, 1, HR)[None, None, None, :]
    (wrapped(x) * ramp).sum().backward()
    grads = [p.grad for p in gen.parameters() if p.grad is not None]
    assert grads and any(g.abs().max() > 0 for g in grads)

    gen.zero_grad(set_to_none=True)
    wrapped(x).mean().backward()
    flat = torch.cat([p.grad.reshape(-1) for p in gen.parameters()])
    assert math.isclose(float(flat.abs().max()), 0.0, abs_tol=1e-9)
