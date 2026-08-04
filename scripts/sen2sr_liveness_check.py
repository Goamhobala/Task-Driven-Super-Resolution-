"""Prove which parts of the shipped SEN2SR-Lite actually influence the output.

Three independent demonstrations that SPAB blocks 1..N-2 are structurally dead
(upstream `CNNSR.forward` feeds `out_feature` to EVERY block and only consumes
block 0's and the last block's results — the ModuleList refactor of the
original SPAN lost the block-to-block chaining):

  1. autograd graph membership — after backprop of a spatially varying loss,
     middle-block parameter grads are None (they are not in the graph at all);
  2. function identity — zeroing every weight of a middle block changes the
     output by exactly 0.0; zeroing block 0 (control) changes it;
  3. (offline, no torch needed) weight forensics: in model.safetensor the
     middle blocks' train-branch stats are IDENTICAL to each other
     (std .0450/.0449/.0450/.0450, max/std 4.53-4.54, kurt 7.39-7.41 = frozen
     at kaiming-uniform init) while blocks 0/5, conv_1/2/cat and the upsampler
     show trained, heavy-tailed weights (max/std 8-13). SEN2SR itself was
     trained with the disconnected topology — do NOT "fix" the chaining.

NB use a spatially varying loss for test 1: the FFT hard constraint takes the
DC bin from the bicubic branch, so a plain mean() has exactly zero gradient
w.r.t. ALL CNNSR parameters (a healthy net would falsely look dead).

Run locally (torch venv):
    PYTHONPATH=src python scripts/sen2sr_liveness_check.py [model_dir]
"""
import sys

import torch

from sr.sen2sr_loader import load_trainable_sen2sr

MD = sys.argv[1] if len(sys.argv) > 1 else "models/SEN2SRLite_RGBN"

torch.manual_seed(0)
x = torch.rand(1, 4, 128, 128) * 0.3          # reflectance-domain input

# --- 1. graph membership --------------------------------------------------
m = load_trainable_sen2sr(MD)
for p in m.parameters():                       # undo the loader's DDP freeze so
    p.requires_grad_(True)                     # "dead" cannot be an artefact of it
y = m(x)
(y * torch.randn_like(y)).sum().backward()     # spatially varying scalar

def verdict(mod):
    ps = [p for p in mod.parameters()]
    live = any(p.grad is not None and p.grad.abs().sum() > 0 for p in ps)
    return "LIVE" if live else "DEAD (grad is None -> not in the graph)"

print("conv_1 (stem):", verdict(m.sr_model.conv_1))
for i, blk in enumerate(m.sr_model.blocks):
    print(f"blocks[{i}]  :", verdict(blk))
print("conv_2      :", verdict(m.sr_model.conv_2))
print("conv_cat    :", verdict(m.sr_model.conv_cat))
print("upsampler   :", verdict(m.sr_model.upsampler))

# --- 2. function identity -------------------------------------------------
with torch.no_grad():
    m = load_trainable_sen2sr(MD).eval()
    y0 = m(x)
    mid = len(m.sr_model.blocks) // 2
    for p in m.sr_model.blocks[mid].parameters():
        p.zero_()
    d_mid = (m(x) - y0).abs().max().item()
    m = load_trainable_sen2sr(MD).eval()       # fresh copy for the control
    for p in m.sr_model.blocks[0].parameters():
        p.zero_()
    d_b0 = (m(x) - y0).abs().max().item()
print(f"\nzeroed blocks[{mid}] (middle):  max|dy| = {d_mid}   (expect exactly 0.0)")
print(f"zeroed blocks[0]  (control): max|dy| = {d_b0}   (expect > 0)")
