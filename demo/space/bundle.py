"""Assemble and push the ZeroGPU Space (plan §5 Phase 3).

The Space is a BUNDLE, not the repo: it needs `sr`/`unet` importable flat
alongside app.py, the SEN2SR-Lite base dir (architecture construction reads it
even though the fine-tuned weights come from the checkpoint), and the stripped
checkpoint. Everything else in this repo -- training scripts, benchmarking,
probes, figures -- would only slow the image build.

`benchmarking` is deliberately NOT vendored: infer.py imports it only inside
the MPS branch, which never executes on ZeroGPU's CUDA.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SPACE_ID = "Goamhobala/instaroad-demo"

# The MEASURED import closure of `sr.model` -- not a guess. Recompute with:
#   python -c "import sys; sys.path.insert(0,'src'); import sr.model; \
#     print([m for m in sys.modules if m.startswith(('sr.','unet.'))])"
# viz_*, probes, tune and cli are training/figure tooling (matplotlib, optuna,
# wandb) and are deliberately left out. `unet.losses` is NOT optional: it is
# imported inside UNetLightning.__init__, so a narrower list builds fine and
# then dies at model construction.
SR_KEEP = ("__init__.py", "model.py", "sen2sr_loader.py", "sr4rs_torch.py",
           "callbacks.py")
UNET_KEEP = ("__init__.py", "model.py", "losses.py")

README = """---
title: InstaRoad
emoji: 🛣️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
short_description: Road extraction from 10 m Sentinel-2
---

# InstaRoad — inference back end

Takes a **cell id**, not an image: the Space holds its own copy of the ROSA_New
tiles and fetches one 2.1 MB file per request. The map UI lives separately and
calls this directly from the browser, so ZeroGPU quota is spent per visitor
rather than drained from one server token.

| arm | front end | head | window | θ* |
|---|---|---|---|---|
| `r2a` | SEN2SR-Lite ×4, jointly fine-tuned, FFT hard constraint on | ResNet34 U-Net | 128 px | 0.70 |

Windowing mirrors the benchmarked path bit-for-bit (`demo/space/parity.py`:
max\\|Δ prob\\| = 0.0). Predictions on `train`/`val` sites are on imagery the
model has seen — the UI labels them.
"""


def build(dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data").mkdir(parents=True)

    for f in ("app.py", "infer.py", "requirements.txt"):
        shutil.copy2(HERE / f, dst / f)
    for f in ("cells.json", "stretch.json"):
        shutil.copy2(HERE / "data" / f, dst / "data" / f)
    (dst / "README.md").write_text(README)

    for pkg, keep in (("sr", SR_KEEP), ("unet", UNET_KEEP)):
        (dst / pkg).mkdir()
        for name in keep:
            src = REPO / "src" / pkg / name
            if src.exists():
                shutil.copy2(src, dst / pkg / name)
    shutil.copytree(REPO / "src" / "unet" / "configs", dst / "unet" / "configs",
                    ignore=shutil.ignore_patterns("*.pyc"))

    shutil.copytree(REPO / "models" / "SEN2SRLite_RGBN",
                    dst / "models" / "SEN2SRLite_RGBN")
    shutil.copytree(REPO / "demo" / "weights" / "r2a", dst / "weights" / "r2a")

    n = sum(1 for _ in dst.rglob("*") if _.is_file())
    b = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"bundle: {dst}  ({n} files, {b/1e6:.1f} MB)")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/Volumes/KIOXIA/instaroad_space"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--space", default=SPACE_ID)
    args = ap.parse_args()

    dst = build(args.out)
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        # MUST specify ZeroGPU hardware at CREATE time. A Gradio Space on the
        # default cpu-basic now requires PRO (402), while ZeroGPU hosting is
        # free for accounts in good standing -- so there is no "create cheap,
        # upgrade later" path; the hardware has to be right on the first call.
        api.create_repo(args.space, repo_type="space", space_sdk="gradio",
                        space_hardware="zero-a10g", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(dst), repo_id=args.space,
                          repo_type="space", commit_message="r2a serving bundle")
        print(f"pushed -> https://huggingface.co/spaces/{args.space}")
        print("NOTE: set hardware to ZeroGPU in Space settings, and either make "
              "the tiles dataset public or add an HF_TOKEN secret.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
