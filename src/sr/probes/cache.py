"""Reading the extraction cache — shared by `lda.py`, `occlusion.py`, `cka.py`.

Three guards live here rather than in each analysis, because each of them is a
way to silently draw a wrong figure:

  * **one fixture per figure.** Caches extracted against different fixtures are
    not on one axis. The hash covers both fixture files, and any disagreement
    is fatal rather than a warning.
  * **labels sliced to what the cache holds.** A `--limit N` extraction covers
    the first N chips only; pairing its pixels against the full label vector
    would mislabel every one of them.
  * **theta provenance.** A run whose theta is a default rather than a
    re-argmax of its own sweep must be dropped from theta-dependent readouts
    (plan §9) and kept in threshold-free ones.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import numpy as np

from sr.probes import style
from sr.probes.make_fixtures import PIXELS_NPZ


def load_metas(cache_dir, arms=None, require=()) -> list[dict]:
    """Every arm-seed cache under `cache_dir`, in figure order.

    `require` names files a cache must hold to be included (e.g.
    `("occlusion.parquet",)`), so an analysis skips a partial extraction
    instead of dying on it. `meta["dir"]` is added for the caller.

    `arms` are fnmatch patterns against the arm key, so a family selects with
    one token: `r2a@*` takes the whole lr_sr grid row and leaves the seeded
    `r2a` runs out.

    The arm key is RE-DERIVED here from the run name via `style.arm_of` rather
    than trusted from the cache. `extract.py` stamps one in at extraction time,
    but a naming rule that changes afterwards (as it did when the lr_sr grid
    arrived) would otherwise mean re-running hours of GPU work to relabel a
    figure.
    """
    out = []
    for d in sorted(Path(cache_dir).iterdir()):
        f = d / "meta.json"
        if not (d.is_dir() and f.exists()):
            continue
        if any(not (d / r).exists() for r in require):
            continue
        m = json.loads(f.read_text())
        m["arm"] = style.arm_of(m["run"])
        if arms and not any(fnmatch.fnmatch(m["arm"], p) for p in arms):
            continue
        m["dir"] = d
        out.append(m)
    if not out:
        raise SystemExit(
            f"no extraction cache under {cache_dir}"
            + (f" holding {', '.join(require)}" if require else "")
            + " — run `python -m sr.probes.extract` first")
    hashes = {m["fixture_hash"] for m in out}
    if len(hashes) > 1:
        raise SystemExit(
            f"caches were extracted against {len(hashes)} different fixtures "
            f"({', '.join(sorted(hashes))}). They are not on one axis and must "
            "not share a figure — re-extract the stale ones.")
    return sorted(out, key=lambda m: (style.sort_key(m["arm"]),
                                      m["seed"] if m["seed"] is not None else -1))


def seed_counts(metas) -> dict[str, int]:
    """arm -> how many seeds are cached (drives the hollow-marker cue)."""
    c = {}
    for m in metas:
        c[m["arm"]] = c.get(m["arm"], 0) + 1
    return c


def theta_usable(meta) -> bool:
    """False when theta is a default rather than this run's own sweep argmax."""
    return meta.get("theta_provenance") != "fallback"


def labels_for(meta, fixture_dir) -> np.ndarray:
    """The fixture's road/background labels, truncated to this cache's chips."""
    px = np.load(Path(fixture_dir) / PIXELS_NPZ)
    y = px["is_road"][px["chip_idx"] < meta["n_chips"]]
    if y.size != meta["n_pixels"]:
        raise SystemExit(
            f"{meta['run']}: cache holds {meta['n_pixels']} pixels but the "
            f"fixture offers {y.size} for its first {meta['n_chips']} chips — "
            "the cache and the fixture have diverged.")
    return y
