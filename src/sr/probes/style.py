"""One visual system for every probe figure (plan §9).

Imported by `lda.py`, `occlusion.py` and `cka.py` so an arm's colour, an
HC state's line style and the "provisional, single seed" cue are decided ONCE.
Re-assigning any of them between figures would make a reader compare two panels
by colour and be wrong, which is the specific failure this module exists to
prevent.

THREE ORTHOGONAL CHANNELS, ONE PER FACTOR OF THE DESIGN
-------------------------------------------------------
The R-series crosses three things, so the encoding gives each its own channel
and never overloads one:

    hue        GENERATOR provenance   grey bicubic / blue SEN2SR / green SR4RS
    lightness  ADAPTATION state       frozen SR = light, jointly tuned = dark
    linestyle  HARD CONSTRAINT        on = solid, off = dashed

Hue carries the row because that is the contrast the eye should read first
(§1's `r4x − r2x`), and blue/orange/grey survives the common colour-vision
deficiencies. A fourth factor would need a fourth channel and there isn't one —
which is a reason to keep the arm set as it is.

FILLED MARKER = REPLICATED, HOLLOW = SINGLE SEED
------------------------------------------------
§9 asks for the provisional cue to be visual rather than a caption footnote.
`marker_kwargs(n_seeds)` returns a hollow marker below `MIN_SEEDS_FOR_ERRORBAR`,
so an arm with no within-arm noise floor cannot be read as if it had one.
Error bars are drawn only at n >= 3 for the same reason.
"""
from __future__ import annotations

from pathlib import Path

# Every figure lands here. Repo-relative and absolute, so a script run from any
# cwd writes to the same place; `figures/` is gitignored, which is why the
# CSVs beside the figures are outputs to be regenerated, not artifacts to keep.
# (The plan's §9 says `figs/probes/`; the repo already groups outputs under
# `figures/<group>/`, so this follows the repo.)
FIGURES_DIR = str(Path(__file__).resolve().parents[3] / "figures" / "probes")

# Points, not pixels: the thesis is LaTeX and these are the column widths.
SINGLE_COL_IN = 3.3
FULL_WIDTH_IN = 7.0
MIN_SEEDS_FOR_ERRORBAR = 3

GREY = "#6b6b6b"
ROAD = "#b3283a"       # road class, in the LDA densities
BG = "#3d3d3d"         # background class
ZERO_LINE = "#9a9a9a"

# The lr_sr grid (`sr_r2grid_{on,off}_ls{LR}_..._rails_seed0`). `ls` in those
# run names is lr_sr, NOT l2sp_lambda — every grid run has l2sp_lambda 0.0.
# Ordered by lr_sr ASCENDING, so a grid reads as adaptation winding up: the
# near-frozen cell first, the strongly adapted one last (the ramp's dark end).
GRID_LR = ("1e-7", "1e-6", "1e-5", "1e-4")

# arm -> (generator, adapted?, hard constraint?). The pending arms are listed
# so a cache that lands later inherits its colour without a code change.
ARMS = {
    "r0":  ("bicubic", False, False),
    "r1a": ("sen2sr", False, True),
    "r1b": ("sen2sr", False, False),
    "r2a": ("sen2sr", True, True),
    "r2b": ("sen2sr", True, False),
    "r3a": ("sr4rs", False, True),
    "r3b": ("sr4rs", False, False),
    "r4a": ("sr4rs", True, True),
    "r4b": ("sr4rs", True, False),
}
# Each grid cell is its OWN arm: they differ in lr_sr, which is the treatment
# the grid exists to vary, so pooling them would average away the finding.
for _lr in GRID_LR:
    ARMS[f"r2a@{_lr}"] = ("sen2sr", True, True)
    ARMS[f"r2b@{_lr}"] = ("sen2sr", True, False)
    ARMS[f"r4a@{_lr}"] = ("sr4rs", True, True)
    ARMS[f"r4b@{_lr}"] = ("sr4rs", True, False)

# Left-to-right / top-to-bottom order in every figure: the anchor, then the
# SEN2SR row frozen-before-joint, then the SR4RS row the same way, so the two
# rows are read as the parallel constructions they are. Inside the grid, lr_sr
# ascends, so a row reads as adaptation strength winding up from near-frozen.
ORDER = (["r0", "r1a", "r1b", "r2a", "r2b"]
         + [f"r2a@{lr}" for lr in GRID_LR] + [f"r2b@{lr}" for lr in GRID_LR]
         + ["r3a", "r3b", "r4a", "r4b"]
         + [f"r4a@{lr}" for lr in GRID_LR] + [f"r4b@{lr}" for lr in GRID_LR])

_HUE = {
    #            frozen (light)   jointly tuned (dark)
    "bicubic": (GREY, GREY),
    "sen2sr":  ("#9ecae1", "#08519c"),
    "sr4rs":   ("#a1d99b", "#238b45"),
}

# lr_sr IS an adaptation-strength knob — 1e-7 is nearly frozen, 1e-4 strongly
# adapted — so the grid keeps the same lightness channel the a/b arms use and
# simply reads it as a ramp. ColorBrewer Blues, perceptually ordered, and every
# step darker than the frozen shade above so the frozen/joint split survives.
_GRID_RAMP = {"1e-4": "#08306b", "1e-5": "#08519c",
              "1e-6": "#3182bd", "1e-7": "#6baed6"}
# The SR4RS grid (r4grid) is the same ramp in the SR4RS hue, so a grid cell
# still reads as "this generator, adapted this hard" across the two families.
_GRID_RAMP_SR4RS = {"1e-4": "#005a32", "1e-5": "#238b45",
                    "1e-6": "#41ab5d", "1e-7": "#74c476"}


def arm_of(run_or_arm: str) -> str:
    """Run-directory name (or an arm key) -> the arm key figures are keyed on.

        sr_r2b_new_nohc_..._seed66                -> r2b
        sr_r2grid_on_ls1e-4_..._rails_seed0       -> r2a@1e-4
        sr_r2grid_off_ls1e-4_nohc_..._seed0       -> r2b@1e-4

    The grid's on/off is the same hard-constraint switch that names r2a vs r2b,
    so the cells are keyed into those two families rather than given a third
    name — `r2a@1e-4` reads as "the r2a recipe at lr_sr 1e-4" without inventing
    vocabulary. This is the single authority on the mapping: `extract.py` stamps
    an arm into each cache too, but that one is advisory and `cache.load_metas`
    overwrites it from here, so a naming fix never needs a re-extraction.
    """
    import re

    g = re.match(r"^(?:sr_)?r(2|4)grid_(on|off)_ls([0-9.e+-]+?)_", run_or_arm)
    if g:
        return (f"r{g.group(1)}{'a' if g.group(2) == 'on' else 'b'}"
                f"@{g.group(3)}")
    m = re.match(r"^(?:sr_)?(r\d+[a-z]?)(?:@[^_]*)?(?:_|$)", run_or_arm)
    return m.group(1) + (run_or_arm[m.end(1):] if run_or_arm[m.end(1):].startswith("@")
                         else "") if m else run_or_arm


def color(arm: str) -> str:
    a = arm_of(arm)
    if "@" in a:
        base, lr = a.split("@", 1)
        gen = ARMS.get(a, ARMS.get(base, ("sen2sr",)))[0]
        ramp = _GRID_RAMP_SR4RS if gen == "sr4rs" else _GRID_RAMP
        return ramp.get(lr, _HUE[gen][1])
    gen, adapted, _ = ARMS.get(a, ("bicubic", False, False))
    return _HUE[gen][1 if adapted else 0]


def linestyle(arm: str) -> str:
    """Solid when the FFT hard constraint is mounted, dashed when it is off.

    r0 has no generator to constrain; it is drawn solid as the anchor rather
    than dashed, which would read as "constraint off" and invite a comparison
    that does not exist.
    """
    a = arm_of(arm)
    if a == "r0":
        return "-"
    return "-" if ARMS.get(a, (None, None, False))[2] else "--"


def label(arm: str) -> str:
    a = arm_of(arm)
    gen, adapted, hc = ARMS.get(a, (None, None, None))
    if a == "r0":
        return "r0 (bicubic)"
    if "@" in a:
        base, lr = a.split("@", 1)
        return f"{base} (HC {'on' if hc else 'off'}, lr_sr {lr})"
    if gen is None:
        return a
    return f"{a} ({'joint' if adapted else 'frozen'} {gen}, HC {'on' if hc else 'off'})"


def hatch(arm: str) -> str | None:
    """Hatching carries the hard-constraint state on BARS.

    The third channel has to be re-expressed once per mark type: lines get
    `linestyle`, markers get `marker`, bars get this. Without it the two joint
    arms share a hue and a fill and become one bar in the loadings and
    first-conv panels.
    """
    a = arm_of(arm)
    if a == "r0":
        return None
    return None if ARMS.get(a, (None, None, False))[2] else "///"


def marker(arm: str) -> str:
    """Marker SHAPE carries the hard-constraint state, as linestyle does on lines.

    A dot plot has no lines, so `linestyle` cannot encode HC there and the two
    joint arms would collapse onto one another — fill is already spoken for by
    the replication cue. Shape restores the third channel, with the same meaning
    as the linestyle it stands in for: round = constraint on, square = off.
    """
    a = arm_of(arm)
    if a == "r0":
        return "D"                      # the anchor, neither on nor off
    return "o" if ARMS.get(a, (None, None, False))[2] else "s"


def marker_kwargs(n_seeds: int, marker: str = "o", size: float = 5.5) -> dict:
    """Filled when the arm is replicated, hollow when it is a single seed."""
    c = {"marker": marker, "markersize": size, "linestyle": "none"}
    return ({**c, "markerfacecolor": "none", "markeredgewidth": 1.4}
            if n_seeds < MIN_SEEDS_FOR_ERRORBAR else {**c, "markeredgewidth": 0.8})


def sort_key(arm: str):
    a = arm_of(arm)
    return (ORDER.index(a) if a in ORDER else len(ORDER), a)


def apply_rc():
    """Thesis-facing matplotlib defaults. Call once, before any figure."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        # Type 42 = TrueType: Overleaf/arXiv reject the Type 3 bitmaps
        # matplotlib emits by default.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out_dir, stem: str):
    """PDF (vector, for Overleaf) + PNG preview, same stem. Returns both paths."""
    from pathlib import Path

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = [d / f"{stem}.pdf", d / f"{stem}.png"]
    for p in paths:
        fig.savefig(p)
    return paths
