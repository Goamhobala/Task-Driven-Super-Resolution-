"""Instrument D's splice must BE the deployed operator, not a lookalike.

The whole claim of the band swap is that it applies the same frequency cut the
hard constraint applies. If `splice` and `HardConstraint.forward` differ by so
much as a convention, every figure downstream is about a different operator.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sr.probes.bandswap_extract import ideal_mask, mean_match, splice


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(20260831)


def test_splice_degenerate_masks(rng):
    """An all-ones mask takes everything from the low donor, all-zeros the high."""
    a = torch.randn(1, 4, 32, 32, generator=rng)
    b = torch.randn(1, 4, 32, 32, generator=rng)
    ones = torch.ones(32, 32)
    assert torch.allclose(splice(a, b, ones, torch), a, atol=1e-5)
    assert torch.allclose(splice(a, b, 1 - ones, torch), b, atol=1e-5)


def test_splice_is_hard_constraint():
    """`splice(bicubic(lr), sr, mask)` == `HardConstraint(lr, sr)`, exactly.

    This is the test the instrument rests on: the same mask file, the same
    fftshift convention, the same low/high combination. HardConstraint shifts
    ALL dims (it passes no `dim=`), which for a (B,C,H,W) tensor rolls the
    channel axis; the roll cancels because the mask is shared across channels
    and the inverse shift undoes it. `splice` shifts only the spatial dims and
    must land on the same answer.
    """
    HardConstraint = pytest.importorskip("sen2sr.models.tricks").HardConstraint

    g = torch.Generator().manual_seed(7)
    lr = torch.rand(2, 4, 8, 8, generator=g)
    sr = torch.rand(2, 4, 32, 32, generator=g)
    mask = ideal_mask(32, 6, "cpu", torch)

    ref = HardConstraint(low_pass_mask=mask, bands="all")(lr, sr)
    lr_up = torch.nn.functional.interpolate(lr, size=(32, 32), mode="bicubic",
                                            antialias=True)
    got = splice(lr_up, sr, mask, torch)
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max()


def test_ideal_mask_matches_upstream():
    """Same `distance <= cutoff` boundary as `sen2sr...ideal_filter`."""
    up = pytest.importorskip("sen2sr.models.tricks").ideal_filter
    for r in (0, 1, 5, 16):
        assert torch.equal(ideal_mask(32, r, "cpu", torch), up((32, 32), r))


def test_mean_match_moves_only_dc(rng):
    """Mean-matching is a pure DC move: every other Fourier bin is untouched."""
    a = torch.randn(1, 3, 16, 16, generator=rng)
    t = torch.randn(1, 3, 16, 16, generator=rng)
    out = mean_match(a, t, torch)
    assert torch.allclose(out.mean(dim=(-2, -1)), t.mean(dim=(-2, -1)), atol=1e-6)
    Fa = torch.fft.fftshift(torch.fft.fftn(a, dim=(-2, -1)), dim=(-2, -1))
    Fo = torch.fft.fftshift(torch.fft.fftn(out, dim=(-2, -1)), dim=(-2, -1))
    Fa[..., 8, 8] = Fo[..., 8, 8]        # blank the DC bin, then require equality
    assert torch.allclose(Fa, Fo, atol=1e-4)


def test_radius_zero_keeps_only_dc_from_donor(rng):
    """radius 0 donates the DC bin alone — the weakest rung of the curve."""
    a = torch.randn(1, 2, 16, 16, generator=rng)
    b = torch.randn(1, 2, 16, 16, generator=rng)
    out = splice(a, b, ideal_mask(16, 0, "cpu", torch), torch)
    assert torch.allclose(out.mean(dim=(-2, -1)), a.mean(dim=(-2, -1)), atol=1e-5)
    assert torch.allclose(out - out.mean(dim=(-2, -1), keepdim=True),
                          b - b.mean(dim=(-2, -1), keepdim=True), atol=1e-5)
