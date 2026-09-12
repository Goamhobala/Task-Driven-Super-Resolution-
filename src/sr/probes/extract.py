"""One GPU pass per arm-seed: cache everything the three probes read.

    PYTHONPATH=src python -m sr.probes.extract \
        --runs-dir /Volumes/MAC_KIOXIA/Data/InstaRoad/SRruns \
        --runs-dir /Volumes/MAC_KIOXIA/Data/InstaRoad/SRruns/refits \
        --sr-dir models/SEN2SRLite_RGBN --device mps

Per run this writes `<out-dir>/<run>/`:

    meta.json        ckpt, hparams, theta*, band stats, fixture hash, timings
    sr_pixels.npy    (n_pixels, C) fp32  -- instrument A (LDA)
    cka_own/*.npy    (n_chips*P, C) fp32 -- instrument B, own-pipeline stimulus
    cka_common/*.npy (n_chips*P, C) fp32 -- instrument B, common-input stimulus
    occlusion.pq     per chip x condition: AP + confusion at theta* -- instrument C
    first_conv.npy   the U-Net stem convolution, for C's weight-space cross-check

Nothing is analysed here. The three analysis scripts read only this cache, so
figures regenerate free of the GPU and an arm landing later is a re-run of this
one script, not a redesign.

THE THREE THINGS THIS PASS MUST GET RIGHT
-----------------------------------------
**Working space.** Instrument A lives on `y = _sr_forward(x)` — the exact
tensor the U-Net consumes BEFORE the z-score — divided by the checkpoint's
`reflectance_scale`, i.e. in reflectance. `_sr_forward` returns raw units
(it re-multiplies by the scale on the way out, because scale and the baked
band statistics are coupled), so the division is what puts every arm on one
axis. We assert all arms share a scale rather than silently comparing units:
with the ROSA_New V2 COGs they are all 1.0, and a mismatch is a real finding
about the cache, not something to paper over.

**Two stimulus conventions for CKA.** `cka_own` feeds each U-Net its own SR
output (the deployed representation). `cka_common` feeds every U-Net the SAME
reference tensor — the r0 arm's bicubic output — so what remains is what the
U-Net WEIGHTS learned differently. The reference is re-expressed into the
consuming checkpoint's own units and passed through THAT checkpoint's z-score:
the normaliser is part of an arm's front-end, not part of the stimulus, and
holding it fixed instead would compare a network to an input distribution it
was never trained on.

**Occlusion is a substitution, not a deletion.** Band b of `y` is replaced by
that checkpoint's own post-recalibration `band_mean[b]`, so after the z-score
the channel is exactly zero — clean "this band carries no signal" semantics
with no arm-dependent offset. Reliance, not information content: the bands are
correlated and mean-substitution is mildly off-manifold (plan §5).

For a 4-band model the plan's "NIR group" IS the NIR band, so five distinct
conditions are emitted, not six; `RGB` is its complement.

PRECISION AND EVAL STATE
------------------------
fp32 throughout, autocast never enabled: CKA and Fisher ratios are statistics
of activations and bf16 quantisation is avoidable noise. The models are put in
`.eval()`, which also disables the adaptive-norm EMA (`_adapt_enabled` requires
`self.training`), so `band_mean`/`band_std` cannot move during extraction —
the final checkpoints are already `_recalpost` and must not be recalibrated
again.
"""
from __future__ import annotations

# MPS guard rails, set BEFORE torch initialises its Metal allocator (same values
# as viz_models/viz_sr: the Mac shares its GPU with the UI and an unbounded
# allocation swaps the machine instead of raising something catchable).
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import fnmatch
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sr.probes.make_fixtures import CHIPS_JSON, FIXTURE_DIR, PIXELS_NPZ

# `*_final.ckpt` always, never `last.ckpt`: the R-arm run dirs hold both, the
# generic `_find_checkpoint` helper sorts `last.ckpt` first, and that would
# silently probe the wrong epoch. Finals are named per-arm in some runs
# (`r2a_s2rosa_jointsr_final.ckpt`), hence the leading wildcard.
CKPT_GLOB = "*_s2rosa_jointsr_final.ckpt"
# The R-series era these probes are registered against.
RUN_GLOB = "*gap_ce_anorm_recalpost*"
SR_DIR = "models/SEN2SRLite_RGBN"
# The SR4RS arms (r3x frozen, r4x joint) load their generator from the SR4RS dir
# rather than --sr-dir, and the HC-on ones recorded the training node's mask path.
SR4RS_DIR = "models/SR4RS_RGBN"
HC_MASK = "models/SEN2SRLite_RGBN/hard_constraint.safetensor"

# bands (1, 2, 3, 4) of the V2 COGs are [B4, B3, B2, B8].
BAND_NAMES = ("R", "G", "B", "NIR")
# name -> band indices flattened to their own mean. "NIR" doubles as the plan's
# NIR *group*; "RGB" is its complement.
OCCLUSION = {
    "R": (0,), "G": (1,), "B": (2,), "NIR": (3,), "RGB": (0, 1, 2),
}

ARM_RE = re.compile(r"^sr_(?P<arm>r\d+[a-z]?)_")
SEED_RE = re.compile(r"_seed(?P<seed>\d+)$")


# ------------------------------------------------------------------- fixtures
def load_fixture(fixture_dir: Path):
    """(meta, chips, pixel sample, sha256 of both files)."""
    chips_p, pix_p = fixture_dir / CHIPS_JSON, fixture_dir / PIXELS_NPZ
    if not chips_p.exists() or not pix_p.exists():
        raise SystemExit(
            f"fixture missing under {fixture_dir} — run `python -m "
            "sr.probes.make_fixtures` first (it is meant to be built once).")
    rec = json.loads(chips_p.read_text())
    px = np.load(pix_p)
    h = hashlib.sha256(chips_p.read_bytes())
    h.update(pix_p.read_bytes())
    return rec["meta"], rec["chips"], px, h.hexdigest()[:16]


# ------------------------------------------------------------------ discovery
def run_meta(run: Path) -> dict:
    """arm / seed parsed out of a run-dir name (`sr_r2b_new_nohc_..._seed66`)."""
    a, s = ARM_RE.match(run.name), SEED_RE.search(run.name)
    return {"run": run.name,
            "arm": a.group("arm") if a else run.name,
            "seed": int(s.group("seed")) if s else None}


def discover(runs_dirs, include: str, ckpt_glob: str) -> list[Path]:
    """Run dirs under any `--runs-dir` matching `include` and holding one final.

    Non-recursive on purpose: `SRruns/` and `SRruns/refits/` are two flat
    collections of run dirs, and a recursive walk would also sweep up the
    `sr_snapshots/` and per-run copies that live inside them.
    """
    from sr.viz_models import find_ckpt

    out = []
    for d in runs_dirs:
        d = Path(d)
        if not d.is_dir():
            raise SystemExit(f"--runs-dir not a directory: {d}")
        for run in sorted(d.iterdir()):
            if not run.is_dir() or not fnmatch.fnmatch(run.name, include):
                continue
            if find_ckpt(run, ckpt_glob):
                out.append(run)
    return sorted(out, key=lambda r: (run_meta(r)["arm"], run_meta(r)["seed"] or 0))


# ------------------------------------------------------------------ the model
def load_model(ckpt: Path, sr_dir: Path, device: str,
               sr4rs_dir: Path = Path(SR4RS_DIR), hc_mask: Path = Path(HC_MASK)):
    """The bench-safe load: `JointSRUNetLightning.load_from_checkpoint`.

    NOT viz_single's upsampler-keyed snapshot loader — that one reconstructs
    the SR net from hparams and would trip over r2b's missing hard_constraint
    keys. `warm_start_unet=None` skips the stage-1 initialisation (the restore
    supplies the trained U-Net anyway), so arms warm-started on another machine
    still load here. `sen2sr_dir` overrides the training node's weights path —
    with the SR4RS dir when the checkpoint's own `upsampler` is sr4rs — and a
    recorded `hc_mask_path` that does not exist here is swapped for `hc_mask`.
    """
    import torch
    from sr.model import JointSRUNetLightning

    hp = torch.load(ckpt, map_location="cpu", weights_only=False).get(
        "hyper_parameters", {})
    kw = {"sen2sr_dir": str(sr4rs_dir if hp.get("upsampler") == "sr4rs" else sr_dir)}
    if hp.get("hc_mask_path") and not Path(hp["hc_mask_path"]).exists():
        kw["hc_mask_path"] = str(hc_mask)
    m = JointSRUNetLightning.load_from_checkpoint(
        str(ckpt), map_location="cpu", warm_start_unet=None, **kw)
    if getattr(m.hparams, "reflectance_scale", None) is None:
        m.hparams.reflectance_scale = 1.0     # some s2rosa ckpts saved it None
    return m.eval().float().to(device)


def hook_points(model):
    """[(name, module)] for the encoder stages and decoder blocks of `model.model`.

    resnet34's stem output is taken at `encoder.relu` — the smp encoder calls it
    exactly once per forward (the BasicBlocks own their own ReLU), which the
    extraction asserts rather than assumes.
    """
    net = model.model
    if not hasattr(net, "encoder"):
        return []                              # linear-probe head: no internals
    enc, dec = net.encoder, net.decoder
    pts = [("enc_stem", enc.relu)]
    pts += [(f"enc_layer{i}", getattr(enc, f"layer{i}")) for i in range(1, 5)
            if hasattr(enc, f"layer{i}")]
    pts += [(f"dec_block{i}", b) for i, b in enumerate(getattr(dec, "blocks", []))]
    return pts


# ------------------------------------------------------------------- the pass
class Extractor:
    """Holds one loaded arm-seed and the per-chip caches it accumulates."""

    def __init__(self, model, device, positions, cka_seed, stages):
        import torch

        self.torch = torch
        self.model = model
        self.device = device
        self.positions = positions
        self.cka_seed = cka_seed
        self.scale = float(model.hparams.reflectance_scale)
        self.stages = stages
        self._grab = {}          # stage name -> last forward's output
        self._calls = {}
        self._handles = []
        # The occlusion conditions run six extra forwards per chip and want none
        # of this; a hook that always fired would hold ~0.5 GB of activations
        # per chip for nothing.
        self._capture = False

    # --- hooks ---------------------------------------------------------
    def attach(self):
        def mk(name):
            def fn(_m, _i, out):
                if not self._capture:
                    return
                self._grab[name] = out.detach()
                self._calls[name] = self._calls.get(name, 0) + 1
            return fn
        for name, mod in self.stages:
            self._handles.append(mod.register_forward_hook(mk(name)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # --- forwards ------------------------------------------------------
    def sr(self, x):
        """`_sr_forward` on raw input -> the pre-adapter tensor, in raw units."""
        with self.torch.no_grad():
            return self.model._sr_forward(x)

    def unet(self, y, capture=False):
        """The tail of `forward`: z-score, then the segmentation net.

        Split out so a caller can inject a `y` that did not come from THIS
        model's SR net (the common-input convention) or one with a band
        flattened (occlusion), without re-running the generator.
        """
        t = self.torch
        self._grab, self._calls, self._capture = {}, {}, capture
        with t.no_grad(), t.autocast(device_type=y.device.type, enabled=False):
            x_seg = (y - self.model.band_mean) / self.model.band_std
            logits = self.model.model(x_seg)
            probs = t.sigmoid(logits.float())
        self._capture = False
        if capture:
            for name, _ in self.stages:
                n = self._calls.get(name, 0)
                if n != 1:
                    raise RuntimeError(
                        f"hook on {name!r} fired {n} times in one forward — the "
                        "encoder is re-using that module, so its activations are "
                        "not a single stage. Move the hook point.")
        return probs

    # --- samplers ------------------------------------------------------
    def sample_pixels(self, y, flat_idx):
        """(k, C) reflectance rows of one chip's SR output at `flat_idx`."""
        c = y.shape[0]
        flat = y.reshape(c, -1)[:, self.torch.as_tensor(flat_idx, device=y.device)]
        return (flat.t().float() / self.scale).cpu().numpy()

    def sample_stage(self, name, b, chip_i, stage_i):
        """(P, C) activation rows at positions fixed by (stage, chip), not by arm.

        The index draw is seeded from the stage and chip alone, so every arm and
        every seed samples the SAME spatial positions of the SAME chip and the
        CKA examples are paired. Positions vary per chip so the sample sweeps
        the spatial field rather than re-reading one fixed lattice.
        """
        a = self._grab[name][b]                      # (C, H, W)
        c, h, w = a.shape
        n = min(self.positions, h * w)
        rng = np.random.default_rng(self.cka_seed * 1_000_003
                                    + stage_i * 10_007 + chip_i)
        idx = rng.choice(h * w, size=n, replace=False)
        sel = a.reshape(c, -1)[:, self.torch.as_tensor(np.sort(idx), device=a.device)]
        return sel.t().float().cpu().numpy()


def read_chip(dataset_dir: Path, chip: dict, crop: int, upscale: int,
              bands, mask_dirname: str):
    """(image (C,crop,crop) raw, mask (crop*up, crop*up) bool) — the loader's read.

    Uses the training dataloader's own helpers so the tensor the probe feeds the
    model is byte-for-byte the tensor the bench fed it.
    """
    import rasterio
    from rasterio.windows import Window
    from sentinel2data.dataset.joint_sr_dataset import (_read_native,
                                                        _read_raster_hr_mask)

    img = dataset_dir / chip["image_path"]
    with rasterio.open(img) as src:
        H, W = src.height, src.width
        win = Window(chip["left"], chip["top"],
                     min(crop, W - chip["left"]), min(crop, H - chip["top"]))
        x = _read_native(src, list(bands), win, crop)
    m = _read_raster_hr_mask(dataset_dir, chip, mask_dirname, win,
                             crop * upscale, upscale)
    return x, m > 0


def chip_readout(probs, mask, theta, bins):
    """(AP, tp, fp, fn, tn) for one chip at one condition.

    AP is threshold-free and binned (101 bins resolves 0.01, finer than the
    0.025 theta grid the sweeps use); it is NaN on a road-free chip, where AP is
    undefined — a fabricated 0.0 would drag every mean down in proportion to how
    many empty chips the fixture happens to hold, and the fixture holds them on
    purpose.
    """
    from benchmarking.ap_metrics import chip_average_precision

    p = np.asarray(probs, dtype="float32")
    pred = p > theta
    tp = int((pred & mask).sum())
    fp = int((pred & ~mask).sum())
    fn = int((~pred & mask).sum())
    tn = int(mask.size - tp - fp - fn)
    return chip_average_precision(p, mask.astype("uint8"), bins), tp, fp, fn, tn


def extract_run(run: Path, args, fixture, ref_ckpt: Path | None) -> dict:
    """Full extraction for one arm-seed. Returns its meta dict."""
    import pandas as pd
    import torch
    from sr.viz_models import find_ckpt, resolve_theta

    meta_fx, chips, px, fx_hash = fixture
    ckpt = find_ckpt(run, args.ckpt_glob)
    out_dir = Path(args.out_dir) / run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_epoch = raw_ckpt.get("epoch")
    ckpt_step = raw_ckpt.get("global_step")
    del raw_ckpt

    model = load_model(ckpt, Path(args.sr_dir), args.device,
                       Path(args.sr4rs_dir), Path(args.hc_mask))
    # --sr-only stops at the generator, so there is nothing to hook.
    stages = [] if args.sr_only else hook_points(model)
    if not stages and not args.sr_only:
        print(f"  WARN: {run.name} has no encoder/decoder to hook — CKA skipped")
    ex = Extractor(model, args.device, args.positions, args.cka_seed, stages)
    ex.attach()

    ref = None
    if ref_ckpt is not None:
        ref_model = load_model(ref_ckpt, Path(args.sr_dir), args.device,
                               Path(args.sr4rs_dir), Path(args.hc_mask))
        ref = Extractor(ref_model, args.device, args.positions, args.cka_seed, [])
        if abs(ref.scale - ex.scale) > 1e-9:
            raise SystemExit(
                f"{run.name} reflectance_scale={ex.scale} but the common-input "
                f"reference is {ref.scale}. The scale is coupled to the baked "
                "band statistics, so the two caches would be in different units. "
                "Re-express the reference before comparing (viz_grid's rule).")

    theta, theta_note = resolve_theta(run, args.select_on, args.default_theta)
    # A machine-readable provenance, because the figures FILTER on it: a
    # theta-dependent readout must drop any run whose theta is a default rather
    # than a re-argmax of its own sweep (plan §9, currently r2a/s1). Prose in
    # this field would make that filter a substring guess.
    theta_provenance = ("sweep" if theta_note.startswith(args.select_on)
                        else "recorded" if theta_note == "recorded θ*"
                        else "fallback")
    crop, up = meta_fx["crop"], meta_fx["upscale"]
    bands = tuple(model.hparams.bands)
    if tuple(bands) != (1, 2, 3, 4):
        raise SystemExit(f"{run.name}: bands={bands}; the band names "
                         f"{BAND_NAMES} assume the V2 (1,2,3,4) order.")

    # Fixture pixel sample, grouped by chip so each chip is visited once.
    chip_idx, pix_idx = px["chip_idx"], px["pix_idx"]
    order_start = np.searchsorted(chip_idx, np.arange(len(chips)), "left")
    order_end = np.searchsorted(chip_idx, np.arange(len(chips)), "right")

    n = len(chips) if args.limit is None else min(args.limit, len(chips))
    sr_rows = [None] * n
    cka_own = {s: [] for s, _ in stages}
    cka_com = {s: [] for s, _ in stages}
    occ_rows = []
    band_mean = model.band_mean.reshape(-1)
    t0 = time.perf_counter()

    def progress(i):
        if (i + 1) % args.log_every == 0:
            el = time.perf_counter() - t0
            print(f"    {i + 1}/{n} chips  {el:.0f}s  ({el / (i + 1):.2f} s/chip)")

    for i in range(n):
        chip = chips[i]
        x_np, mask = read_chip(Path(args.dataset_dir or meta_fx["dataset_dir"]), chip, crop, up,
                               bands, meta_fx["mask_dirname"])
        x = torch.from_numpy(np.ascontiguousarray(x_np))[None].to(args.device)

        y = ex.sr(x)                                   # (1, C, H, W) raw units
        sr_rows[i] = ex.sample_pixels(y[0], pix_idx[order_start[i]:order_end[i]])
        if args.sr_only:                               # the LDA reads nothing past here
            del y
            if args.device == "mps":
                torch.mps.empty_cache()                # unified memory: 8 GB shared with the UI
            progress(i)
            continue

        # --- own-pipeline forward: CKA stimulus 1 + the occlusion baseline
        probs = ex.unet(y, capture=bool(stages))
        for si, (s, _) in enumerate(stages):
            cka_own[s].append(ex.sample_stage(s, 0, i, si))
        ap, tp, fp, fn, tn = chip_readout(probs[0, 0].cpu().numpy(), mask,
                                          theta, args.ap_bins)
        occ_rows.append(dict(chip=i, condition="none", ap=ap, tp=tp, fp=fp,
                             fn=fn, tn=tn, road_px=int(mask.sum())))

        # --- occlusion: flatten band(s) to this ckpt's own post-recal mean
        for name, idxs in OCCLUSION.items():
            yo = y.clone()
            for b in idxs:
                yo[:, b] = band_mean[b]
            po = ex.unet(yo, capture=False)
            ap, tp, fp, fn, tn = chip_readout(po[0, 0].cpu().numpy(), mask,
                                              theta, args.ap_bins)
            occ_rows.append(dict(chip=i, condition=name, ap=ap, tp=tp, fp=fp,
                                 fn=fn, tn=tn, road_px=int(mask.sum())))

        # --- common-input forward: CKA stimulus 2 (same tensor for every arm)
        if ref is not None and stages:
            y_ref = ref.sr(x)
            ex.unet(y_ref, capture=True)
            for si, (s, _) in enumerate(stages):
                cka_com[s].append(ex.sample_stage(s, 0, i, si))

        progress(i)

    ex.detach()
    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------- write
    np.save(out_dir / "sr_pixels.npy", np.concatenate(sr_rows).astype("float32"))
    for tag, store in (("cka_own", cka_own), ("cka_common", cka_com)):
        if not any(store.values()):
            continue
        d = out_dir / tag
        d.mkdir(exist_ok=True)
        for s, chunks in store.items():
            np.save(d / f"{s}.npy", np.concatenate(chunks).astype("float32"))
    if occ_rows:
        pd.DataFrame(occ_rows).to_parquet(out_dir / "occlusion.parquet", index=False)

    first_conv = None
    if (not args.sr_only and hasattr(model.model, "encoder")
            and hasattr(model.model.encoder, "conv1")):
        first_conv = model.model.encoder.conv1.weight.detach().cpu().numpy()
        np.save(out_dir / "first_conv.npy", first_conv)

    meta = {
        **run_meta(run),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(ckpt),
        "epoch": ckpt_epoch,
        "global_step": ckpt_step,
        "fixture_hash": fx_hash,
        "fixture_meta": meta_fx,
        "dataset_dir": str(args.dataset_dir or meta_fx["dataset_dir"]),
        "n_chips": n,
        "n_pixels": int(sum(r.shape[0] for r in sr_rows)),
        "theta": theta,
        "theta_provenance": theta_provenance,
        "theta_note": theta_note,
        "theta_select_on": args.select_on,
        "reflectance_scale": ex.scale,
        "band_mean": [float(v) for v in model.band_mean.reshape(-1)],
        "band_std": [float(v) for v in model.band_std.reshape(-1)],
        "band_names": list(BAND_NAMES),
        "occlusion_conditions": {k: list(v) for k, v in OCCLUSION.items()},
        "stages": [s for s, _ in stages],
        "cka_positions": args.positions,
        "cka_seed": args.cka_seed,
        "ap_bins": args.ap_bins,
        "common_input_ref": str(ref_ckpt) if ref_ckpt else None,
        "hparams": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                        else str(v))
                    for k, v in dict(model.hparams).items()},
        "device": args.device,
        "sr_only": bool(args.sr_only),
        "seconds": round(elapsed, 1),
        "first_conv_shape": list(first_conv.shape) if first_conv is not None else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  wrote {out_dir}  ({elapsed:.0f}s, theta={theta:g} "
          f"[{theta_provenance}: {theta_note}])")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", action="append", default=None,
                    help="flat collection of run dirs (repeatable: SRruns and "
                         "SRruns/refits are two such collections)")
    ap.add_argument("--run", action="append", default=None,
                    help="an explicit run dir, bypassing discovery (repeatable)")
    ap.add_argument("--include", default=RUN_GLOB,
                    help="glob a run-dir name must match to be probed")
    ap.add_argument("--ckpt-glob", default=CKPT_GLOB,
                    help="the FINAL checkpoint's name; never last.ckpt")
    ap.add_argument("--sr-dir", default=SR_DIR,
                    help="SR weights dir, overriding the training node's path")
    ap.add_argument("--sr4rs-dir", default=SR4RS_DIR,
                    help="SR weights dir for upsampler=sr4rs checkpoints (r3x/r4x)")
    ap.add_argument("--hc-mask", default=HC_MASK,
                    help="local hard_constraint.safetensor, used when a ckpt's "
                         "recorded hc_mask_path does not exist on this machine")
    ap.add_argument("--sr-only", action="store_true",
                    help="write sr_pixels.npy + meta.json only (the LDA's input); "
                         "no U-Net, CKA or occlusion. Give it its own --out-dir: "
                         "cka.py reads every cache under a dir without a require")
    ap.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ap.add_argument("--dataset-dir", default=None,
                    help="dataset root overriding the fixture's recorded one "
                         "(chip image paths are relative to it)")
    ap.add_argument("--out-dir", default="probe_cache")
    ap.add_argument("--device", default=None, help="default: mps -> cuda -> cpu")
    ap.add_argument("--positions", type=int, default=64,
                    help="spatial positions sampled per chip per stage (CKA)")
    ap.add_argument("--cka-seed", type=int, default=20260828)
    ap.add_argument("--ap-bins", type=int, default=101)
    ap.add_argument("--select-on", default="iou",
                    help="sweep.json criterion theta* is re-argmaxed on")
    ap.add_argument("--default-theta", type=float, default=0.5)
    ap.add_argument("--ref-run", default=None,
                    help="run dir supplying the common-input stimulus "
                         "(default: the r0 arm among the selected runs)")
    ap.add_argument("--no-common", action="store_true",
                    help="skip the common-input CKA convention")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N fixture chips only — for smoke tests")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the discovered runs, theta* and ckpt sizes; "
                         "imports no torch")
    args = ap.parse_args(argv)

    runs = [Path(r) for r in (args.run or [])]
    runs += discover(args.runs_dir or [], args.include, args.ckpt_glob)
    if not runs:
        raise SystemExit("no runs discovered — check --runs-dir / --include")

    from sr.viz_models import find_ckpt, resolve_theta

    print(f"{len(runs)} run(s):")
    for r in runs:
        ck = find_ckpt(r, args.ckpt_glob)
        th, note = resolve_theta(r, args.select_on, args.default_theta)
        m = run_meta(r)
        print(f"  {m['arm']:<5} seed {str(m['seed']):<5} theta*={th:<6g} "
              f"[{note:<12}] {ck.stat().st_size / 1e6:7.1f} MB  {r.name}")

    ref_ckpt = None
    if not (args.no_common or args.sr_only):
        if args.ref_run:
            ref_ckpt = find_ckpt(Path(args.ref_run), args.ckpt_glob)
        else:
            r0s = [r for r in runs if run_meta(r)["arm"] == "r0"]
            if r0s:
                ref_ckpt = find_ckpt(sorted(r0s)[0], args.ckpt_glob)
        if ref_ckpt is None:
            raise SystemExit(
                "no r0 run among the selected runs to supply the common-input "
                "stimulus — pass --ref-run <r0 run dir>, or --no-common to drop "
                "that CKA convention.")
        print(f"common-input reference: {ref_ckpt}")

    if args.dry_run:
        return 0

    if args.device is None:
        import torch
        args.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {args.device}")

    fixture = load_fixture(Path(args.fixture_dir))
    print(f"fixture {fixture[3]}: {fixture[0]['n_chips']} chips, "
          f"{fixture[0]['n_pixels']} pixels, {fixture[0]['split']} split")

    for r in runs:
        print(f"\n--- {r.name}")
        extract_run(r, args, fixture, ref_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
