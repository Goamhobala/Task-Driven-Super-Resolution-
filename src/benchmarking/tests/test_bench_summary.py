"""What the bench pushes into a wandb run's summary.

The training loop's own test metrics are scored at θ=0.5. The reported number
is the bench row, scored at the θ* the post-refit sweep chose on SEEN
(train+val) data — so the summary this builds is the only θ* number wandb ever
sees, and it has to carry everything the store recorded: the pixel means, AP,
and every buffered tolerance.
"""
from __future__ import annotations

import pandas as pd
import pytest

from benchmarking.cli import _bench_summary
from benchmarking.store import append_chips, append_run


def _seed_store(store_dir, run_id="r1", theta=0.35):
    append_run({"run_id": run_id, "model_name": "sr_r3a_new", "seed": 0,
                "split": "test", "threshold": theta, "n_chips": 2}, store_dir)
    append_chips(pd.DataFrame([
        {"run_id": run_id, "chip_id": "c0", "iou": 0.40, "f1": 0.60,
         "precision": 0.55, "recall": 0.65, "ap": 0.50,
         "buffered_f1_r1": 0.70, "buffered_f1_r5": 0.90,
         "buffered_precision_r3": 0.80, "buffered_recall_r3": 0.60},
        {"run_id": run_id, "chip_id": "c1", "iou": 0.60, "f1": 0.80,
         "precision": 0.75, "recall": 0.85, "ap": 0.70,
         "buffered_f1_r1": 0.80, "buffered_f1_r5": 1.00,
         "buffered_precision_r3": 0.90, "buffered_recall_r3": 0.80},
    ]), store_dir)


def test_summary_carries_the_buffered_tolerances_and_ap(tmp_path):
    _seed_store(tmp_path)
    s = _bench_summary(tmp_path, "r1", "test")

    assert s["bench_test/f1"] == pytest.approx(0.70)   # chip mean, not re-pooled
    assert s["bench_test/ap"] == pytest.approx(0.60)
    # Every recorded tolerance travels, so the wandb summary and the store agree.
    assert s["bench_test/buffered_f1_r1"] == pytest.approx(0.75)
    assert s["bench_test/buffered_f1_r5"] == pytest.approx(0.95)
    assert s["bench_test/buffered_precision_r3"] == pytest.approx(0.85)
    assert s["bench_test/buffered_recall_r3"] == pytest.approx(0.70)


def test_summary_records_the_operating_point_it_was_scored_at(tmp_path):
    """θ is the point of the exercise: a summary without it cannot be told
    apart from the training loop's θ=0.5 numbers."""
    _seed_store(tmp_path, theta=0.45)
    s = _bench_summary(tmp_path, "r1", "test")
    assert s["bench_test/threshold"] == 0.45
    assert s["bench_test/n_chips"] == 2
    assert s["bench_test/store_run_id"] == "r1"


def test_absent_columns_are_simply_absent(tmp_path):
    """A store benched before the buffered sweep became the default has no
    buffered_* columns — the push must not invent them."""
    append_run({"run_id": "old", "model_name": "sr_r5_new", "seed": 0,
                "split": "test", "threshold": 0.5, "n_chips": 1}, tmp_path)
    append_chips(pd.DataFrame([{"run_id": "old", "chip_id": "c0",
                                "iou": 0.5, "f1": 0.6}]), tmp_path)
    s = _bench_summary(tmp_path, "old", "test")
    assert not [k for k in s if "buffered" in k]
    assert s["bench_test/f1"] == pytest.approx(0.6)
