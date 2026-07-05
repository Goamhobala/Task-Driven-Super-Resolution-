"""Pytest wrappers around sr.smoke (synthetic data; needs only SEN2SR weights).

Point SEN2SR_DIR at an mlstac SEN2SRLite_RGBN download; tests skip when it (or
the sen2sr package) is missing so the suite stays green on machines without
the weights. The overfit test is slow on CPU — opt in with RUN_OVERFIT=1.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("sen2sr")

SEN2SR_DIR = os.environ.get("SEN2SR_DIR", "")
needs_weights = pytest.mark.skipif(
    not (SEN2SR_DIR and (Path(SEN2SR_DIR) / "model.safetensor").exists()),
    reason="set SEN2SR_DIR to an mlstac SEN2SRLite_RGBN download",
)


@needs_weights
def test_forward_backward_shapes_and_grads():
    import torch

    from sr.smoke import build_module, make_synthetic_batch, run_smoke

    torch.manual_seed(0)
    x, y = make_synthetic_batch(batch=1)
    module = build_module(SEN2SR_DIR, encoder_weights=None)
    norms = run_smoke(module, x, y)  # asserts 4x shape + both groups get grads
    assert norms["sr"] > 0 and norms["seg"] > 0


@needs_weights
def test_frozen_sr_gets_no_grad():
    import torch

    from sr.smoke import build_module, make_synthetic_batch, run_smoke

    torch.manual_seed(0)
    x, y = make_synthetic_batch(batch=1)
    module = build_module(SEN2SR_DIR, freeze_sr=True, encoder_weights=None)
    norms = run_smoke(module, x, y)
    assert norms["sr"] == 0 and norms["seg"] > 0


@needs_weights
@pytest.mark.skipif(os.environ.get("RUN_OVERFIT") != "1",
                    reason="slow; set RUN_OVERFIT=1 to run")
def test_single_batch_overfit():
    import torch

    from sr.smoke import build_module, make_synthetic_batch, run_overfit

    torch.manual_seed(0)
    x, y = make_synthetic_batch(batch=1)
    module = build_module(SEN2SR_DIR, encoder_weights=None)
    first, last = run_overfit(module, x, y, steps=60)
    assert last < 0.5 * first
