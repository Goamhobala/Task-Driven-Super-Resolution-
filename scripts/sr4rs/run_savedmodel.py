"""Ground-truth check + re-extraction from the official SR4RS SavedModel.

The distributed *Checkpoint* artifact appears to contain suspect weights
(zeroed optimizer state, step-0 suffix, garbage output). The *SavedModel* is
the inference-grade artifact. This script, run in the throwaway TF venv:

  1. runs the SavedModel on a real crop -> sr4rs_savedmodel.png + band stats
     (settles whether official SR4RS produces sane output on our data);
  2. compares its variables to the checkpoint-extracted ones (same names?
     same values? -> tells us if the Checkpoint was untrained);
  3. if the generator variable names match, RE-EXTRACTS gen_weights.safetensors
     / gen_meta.json / gen_reference.npz from the SavedModel, overwriting the
     ones in --model-dir — after which rerun the torch parity check
     (python -m sr.sr4rs_torch --model-dir ...) and the sanity viz.

    ~/tfenv/bin/python run_savedmodel.py \
        --saved-model <downloaded SavedModel dir> \
        --model-dir ../../models/SR4RS_RGBN \
        --image ../../src/sr/examples/Durban_r4_c3.tif --row 0 --col 0
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow.compat.v1 as tf  # noqa: E402

tf.disable_eager_execution()

RESBLOCKS = 16
CROP = 128


def read_crop(image_path, row, col):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(image_path) as src:
        x = src.read([1, 2, 3, 4], window=Window(col, row, CROP, CROP)).astype("float32")
    x[x == -32768] = 0.0
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x  # raw DN, (4, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-model", required=True)
    ap.add_argument("--model-dir", required=True, help="dir with the ckpt-extracted gen_* files")
    ap.add_argument("--image", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--col", type=int, default=0)
    ap.add_argument("--no-extract", action="store_true", help="only run + compare")
    args = ap.parse_args()

    sess = tf.Session(graph=tf.Graph())
    with sess.graph.as_default():
        meta = tf.saved_model.load(sess, ["serve"], args.saved_model)
    g = sess.graph

    # ---- discover the serving signature -----------------------------------
    sig = meta.signature_def[list(meta.signature_def.keys())[0]]
    in_name = list(sig.inputs.values())[0].name
    out_name = list(sig.outputs.values())[0].name
    print(f"signature: {in_name} -> {out_name}")

    # ---- 1. run on the real crop (feed raw DN, NHWC) -----------------------
    x = read_crop(args.image, args.row, args.col)
    nhwc = x.transpose(1, 2, 0)[None]
    out = sess.run(g.get_tensor_by_name(out_name), {g.get_tensor_by_name(in_name): nhwc})
    sr = out[0].transpose(2, 0, 1)  # (4, H', W')
    print(f"output shape {sr.shape};  input DN mean {x.mean():.1f}")
    for i, n in enumerate(("B4", "B3", "B2", "B8")):
        print(f"  {n}: in mean {x[i].mean():9.2f}   sr mean {sr[i].mean():9.2f}  "
              f"sr min/max {sr[i].min():.1f}/{sr[i].max():.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = np.percentile(x[:3], (2, 98))
    rgb = lambda a, s=1.0: np.clip((np.transpose(a[:3], (1, 2, 0)) / s - lo) / (hi - lo), 0, 1)
    # output may be DN-domain (mul_4 inside SavedModel) — display raw
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(rgb(x)); axes[0].set_title("input 10 m (DN)")
    axes[1].imshow(rgb(sr)); axes[1].set_title("SavedModel output (as-is)")
    for a in axes: a.set_axis_off()
    fig.savefig("sr4rs_savedmodel.png", dpi=180, bbox_inches="tight")
    print("wrote sr4rs_savedmodel.png  <- if THIS looks good, the Checkpoint "
          "artifact was the problem")

    # ---- 2./3. compare + re-extract effective kernels ----------------------
    def tensor(name):
        return sess.run(g.get_tensor_by_name(name))
    try:
        w = tensor("gen/encoder/conv1_9x9/mul:0")
    except Exception:
        print("\nNOTE: 'gen/...' names not found in SavedModel; generator ops are:")
        for op in g.get_operations():
            if "conv1_9x9" in op.name or "res_4x/output" in op.name:
                print("  ", op.name, op.type)
        raise SystemExit("adapt the name prefix above and rerun")

    import safetensors.numpy
    old = safetensors.numpy.load_file(os.path.join(args.model_dir, "gen_weights.safetensors"))
    new_stem = np.ascontiguousarray(w.transpose(3, 2, 0, 1))
    d = float(np.abs(new_stem - old["gen/encoder/conv1_9x9/weight"]).max())
    print(f"\nstem effective-kernel max|SavedModel - Checkpoint| = {d:.3e} "
          f"{'(same weights!)' if d < 1e-5 else '(DIFFERENT -> Checkpoint was not the trained model)'}")

    if args.no_extract:
        return
    tensors, metad = {}, {"resblocks": RESBLOCKS, "source": "savedmodel"}
    def eff(name): return np.ascontiguousarray(tensor(f"{name}/mul:0").transpose(3, 2, 0, 1))
    def bias(name): return tensor(f"{name}/bias/read:0")
    convs = ["gen/encoder/conv1_9x9", "gen/res_1x/conv1", "gen/res_1x/output"]
    for i in range(RESBLOCKS):
        convs += [f"gen/encoder/ResBlock{i}/conv1", f"gen/encoder/ResBlock{i}/conv2"]
    for s in ("res_2x", "res_4x"):
        convs += [f"gen/{s}/conv1", f"gen/{s}/conv2", f"gen/{s}/conv3", f"gen/{s}/output"]
    for c in convs:
        tensors[c + "/weight"] = eff(c)
        tensors[c + "/bias"] = bias(c)
    for s in ("res_2x", "res_4x"):
        tensors[f"gen/{s}/conv_upsample/weight"] = eff(f"gen/{s}/conv_upsample")
        tensors[f"gen/{s}/blur_filter"] = tensor(f"gen/{s}/Blur2D/filter_blur2d:0")
    node = lambda n: g.as_graph_element(n).node_def
    metad["lrelu_alpha"] = float(node("gen/encoder/LeakyRelu").attr["alpha"].f)
    metad["pixelnorm_eps"] = float(tensor("gen/encoder/ResBlock0/PixelNorm/epsilon:0"))

    # reference taps in the reflectance domain (input scaling stays outside)
    rng = np.random.default_rng(0)
    xr = rng.uniform(0.0, 0.35, (2, 32, 32, 4)).astype("float32")
    # the SavedModel input is DN-domain; find its lr scale from the graph
    try:
        lr_scale = float(tensor("mul_3/x:0"))
    except Exception:
        lr_scale = 1e-4
    metad["lr_scale"] = lr_scale
    taps = {"stem": "gen/encoder/LeakyRelu:0", "resblock0": "gen/encoder/ResBlock0/add:0",
            "res_1x_add": "gen/res_1x/add:0", "res_2x_blur": "gen/res_2x/Blur2D/IdentityN:0",
            "res_2x_feat": "gen/res_2x/PixelNorm_3/mul:0", "out_1x": "gen/res_1x/output/add:0",
            "out_2x": "gen/res_2x/output/add:0", "out_4x": "gen/res_4x/output/add:0"}
    outs = sess.run(list(taps.values()),
                    {g.get_tensor_by_name(in_name): xr / lr_scale})
    ref = {"input": xr.transpose(0, 3, 1, 2)}
    for (k, _), v in zip(taps.items(), outs):
        ref[k] = v.transpose(0, 3, 1, 2)

    safetensors.numpy.save_file(tensors, os.path.join(args.model_dir, "gen_weights.safetensors"))
    with open(os.path.join(args.model_dir, "gen_meta.json"), "w") as fh:
        json.dump(metad, fh, indent=2)
    np.savez_compressed(os.path.join(args.model_dir, "gen_reference.npz"), **ref)
    print(f"\nre-extracted from SavedModel -> {args.model_dir}. Now rerun:")
    print("  python -m sr.sr4rs_torch --model-dir", args.model_dir)


if __name__ == "__main__":
    main()
