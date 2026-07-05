"""Train/val/test splitting"""
import numpy as np

def splitset_map(keys, val_frac, test_frac, seed):
    """Map each unique key -> 'train'/'val'/'test' by a seeded permutation."""
    keys = list(keys)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    n_test = int(round(len(keys) * test_frac))
    n_val = int(round(len(keys) * val_frac))

    assignment = {}
    for rank, idx in enumerate(order):
        key = keys[idx]
        if rank < n_test:
            assignment[key] = "test"
        elif rank < n_test + n_val:
            assignment[key] = "val"
        else:
            assignment[key] = "train"
    return assignment