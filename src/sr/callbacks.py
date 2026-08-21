"""Extra checkpoints at pre-registered epochs (training-curve probes).

The joint-SR runs keep exactly two checkpoints today: the selected one
(``unet_s2rosa_jointsr_best.ckpt``, val-monitored — or, under the train+val
protocol, the unmonitored ``unet_s2rosa_jointsr_final.ckpt`` at the end of the
fixed budget) and ``last.ckpt``. Both sit at the ends of the run, so nothing on
disk answers "what did this model look like half-way through the budget?".

``EpochSnapshotCheckpoint`` writes a full checkpoint after a fixed LIST of
epochs (default: 50 and 75 of the 100-epoch refit budget), leaving the two
existing checkpoints alone.

Deliberately a plain ``Callback`` rather than a second ``ModelCheckpoint``:

  * ``JointSRUNetLightning.on_train_end`` rewrites every UNMONITORED
    ``ModelCheckpoint``'s best/last file with post-fit recalibrated norm stats.
    A periodic ``ModelCheckpoint`` is unmonitored, so its most recent file
    (epoch 75) would be silently re-saved with END-of-run statistics stapled
    onto mid-run weights — the exact failure that method's docstring refuses to
    commit for a monitored best.ckpt. ``trainer.checkpoint_callbacks`` only
    collects ``ModelCheckpoint`` instances, so a plain Callback is invisible to
    it and each snapshot keeps its own epoch's stats.
  * ``ModelCheckpoint`` has no "these specific epochs" mode; ``every_n_epochs``
    would also emit epoch 25 and 100 duplicates.

EPOCH NUMBERING: ``epochs`` counts COMPLETED epochs, so ``50`` means "after the
50th epoch". Lightning is 0-based internally, so the resulting
``..._epoch050.ckpt`` carries ``ckpt["epoch"] == 49`` — the same convention as
the end of a 100-epoch budget, whose final ckpt carries 99.
"""

from __future__ import annotations

from lightning.pytorch.callbacks import Callback


class EpochSnapshotCheckpoint(Callback):
    """Save a full checkpoint after each epoch listed in ``epochs``.

    Args:
        epochs: completed-epoch counts to snapshot at (1-based; see module doc).
        dirpath: checkpoint dir, relative to the run's cwd — the same
            ``checkpoints`` the run's ``ModelCheckpoint`` writes into.
        prefix: file stem prefix. Keep it sorting AFTER ``last.ckpt``:
            ``benchmarking.cli._find_checkpoint`` falls back to the
            alphabetically first ``checkpoints/*.ckpt``, and a snapshot sorting
            ahead of ``last.ckpt`` would quietly become the benched model.
    """

    def __init__(
        self,
        epochs: list[int] = [50, 75],
        dirpath: str = "checkpoints",
        prefix: str = "unet_s2rosa_jointsr",
    ):
        super().__init__()
        self.epochs = sorted({int(e) for e in epochs})
        self.dirpath = dirpath
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        done = trainer.current_epoch + 1          # epochs COMPLETED
        if done not in self.epochs:
            return
        path = f"{self.dirpath}/{self.prefix}_epoch{done:03d}.ckpt"
        # Collective under DDP: every rank calls it, the checkpoint IO writes
        # from rank zero (and creates dirpath itself).
        trainer.save_checkpoint(path)
        if trainer.is_global_zero:
            print(f"[joint_sr] epoch snapshot -> {path} "
                  f"(ckpt['epoch']={trainer.current_epoch})")
