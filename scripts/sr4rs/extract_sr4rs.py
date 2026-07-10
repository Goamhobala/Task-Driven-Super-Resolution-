"""One-time SR4RS TF1-checkpoint -> PyTorch-ready extraction. Run LOCALLY in a
throwaway TF venv (TF never enters the training environment):

    python -m venv ~/tfenv && ~/tfenv/bin/pip install tensorflow safetensors numpy
    ~/tfenv/bin/python extract_sr4rs.py --model-dir <...>/models/SR4RS_RGBN

Writes into --model-dir:
    gen_weights.safetensors   EFFECTIVE kernels (weight x equalized-LR const,
                              taken from the graph's own `*/mul` nodes) in
                              PyTorch OIHW layout, + biases, + blur filters
    gen_meta.json             LeakyRelu alphas, PixelNorm epsilons, io scales,
                              conv_transpose attrs — everything non-tensor
    gen_reference.npz         random reflectance inputs + intermediate taps +
                              final outputs, for layer-by-layer parity checks

The generator (from the graph): 9x9 stem (4->64) + LReLU; 16 ResBlocks
(conv->LReLU->PixelNorm->conv->PixelNorm->+skip); res_1x (conv->PixelNorm->
+stem-skip) + 1x1 head; res_2x/res_4x: fused conv_transpose x2 (StyleGAN2
shifted-kernel-sum) -> Blur2D -> LReLU -> PixelNorm -> conv1..conv3 -> head.
Input domain: DN * 1e-4 (== reflectance); output likewise (heads are in the
scaled domain — the graph's mul_3/mul_4 scaling stays OUTSIDE the module).
"""
import argparse
import glob
import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow.compat.v1 as tf  # noqa: E402

tf.disable_eager_execution()

RESBLOCKS = 16


def ckpt_prefix(model_dir):
    metas = glob.glob(os.path.join(model_dir, "*.meta"))
    assert len(metas) == 1, f"expected one .meta in {model_dir}, got {metas}"
    return metas[0][:-len(".meta")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr-size", type=int, default=32, help="reference input px")
    args = ap.parse_args()
    prefix = ckpt_prefix(args.model_dir)

    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(prefix + ".meta")
    sess = tf.Session(graph=graph)
    saver.restore(sess, prefix)
    g = graph

    # ---- effective conv kernels: the graph's own weight*const nodes --------
    tensors, meta = {}, {"resblocks": RESBLOCKS}
    def eff(name):                                # HWIO -> OIHW
        arr = sess.run(g.get_tensor_by_name(f"{name}/mul:0"))
        return np.ascontiguousarray(arr.transpose(3, 2, 0, 1))
    def bias(name):
        return sess.run(g.get_tensor_by_name(f"{name}/bias/read:0"))

    convs = ["gen/encoder/conv1_9x9", "gen/res_1x/conv1", "gen/res_1x/output"]
    for i in range(RESBLOCKS):
        convs += [f"gen/encoder/ResBlock{i}/conv1", f"gen/encoder/ResBlock{i}/conv2"]
    for s in ("res_2x", "res_4x"):
        convs += [f"gen/{s}/conv1", f"gen/{s}/conv2", f"gen/{s}/conv3", f"gen/{s}/output"]
    for c in convs:
        tensors[c + "/weight"] = eff(c)
        tensors[c + "/bias"] = bias(c)
    for s in ("res_2x", "res_4x"):   # fused-upsample kernels (no bias)
        tensors[f"gen/{s}/conv_upsample/weight"] = eff(f"gen/{s}/conv_upsample")
        tensors[f"gen/{s}/blur_filter"] = sess.run(
            g.get_tensor_by_name(f"gen/{s}/Blur2D/filter_blur2d:0"))

    # ---- non-tensor constants ----------------------------------------------
    def node(name):
        return g.as_graph_element(name).node_def
    meta["lrelu_alpha"] = float(node("gen/encoder/LeakyRelu").attr["alpha"].f)
    meta["pixelnorm_eps"] = float(sess.run(
        g.get_tensor_by_name("gen/encoder/ResBlock0/PixelNorm/epsilon:0")))
    meta["pixelnorm_axis"] = [int(v) for v in np.atleast_1d(sess.run(
        g.get_tensor_by_name("gen/encoder/ResBlock0/PixelNorm/Mean/reduction_indices:0")))]
    # mul_3 = placeholder * lr_scale (either operand order); mul_4 likewise.
    def const_and_ph(mul_name):
        ins = [i.split(":")[0] for i in node(mul_name).input]
        ops = {i: g.as_graph_element(i).node_def.op for i in ins}
        # the graph's inputs are PlaceholderWithDefault (default = the training
        # data iterator); feeding them works exactly like a plain Placeholder.
        ph = next((i for i in ins
                   if ops[i] in ("Placeholder", "PlaceholderWithDefault")), None)
        const = next(i for i in ins if ops[i] == "Const")
        return float(sess.run(g.get_tensor_by_name(const + ":0"))), ph

    meta["lr_scale"], input_ph = const_and_ph("mul_3")
    meta["hr_scale"], _ = const_and_ph("mul_4")
    ct = node("gen/res_2x/conv_upsample/conv2d_transpose")
    meta["convT_strides"] = [int(v) for v in ct.attr["strides"].list.i]
    meta["convT_padding"] = ct.attr["padding"].s.decode()
    blur = node("gen/res_2x/Blur2D/depthwise")
    meta["blur_padding"] = blur.attr["padding"].s.decode()

    # ---- reference pairs (reflectance domain, mul_3/mul_4 kept OUTSIDE) ----
    rng = np.random.default_rng(args.seed)
    x = rng.uniform(0.0, 0.35, (2, args.lr_size, args.lr_size, 4)).astype("float32")
    assert input_ph is not None, "no Placeholder feeds mul_3 — inspect the graph"
    meta["input_placeholder"] = input_ph
    lr_ph = g.get_tensor_by_name(input_ph + ":0")
    taps = {
        "stem": "gen/encoder/LeakyRelu:0",
        "resblock0": "gen/encoder/ResBlock0/add:0",
        "res_1x_add": "gen/res_1x/add:0",
        "res_2x_blur": "gen/res_2x/Blur2D/IdentityN:0",
        "res_2x_feat": "gen/res_2x/PixelNorm_3/mul:0",
        "out_1x": "gen/res_1x/output/add:0",
        "out_2x": "gen/res_2x/output/add:0",
        "out_4x": "gen/res_4x/output/add:0",
    }
    # placeholder is DN-domain; mul_3 applies lr_scale. Feed DN = x / lr_scale
    # so the network sees exactly `x` (reflectance) — matching the torch port.
    feeds = {lr_ph: x / meta["lr_scale"]}
    outs = sess.run(list(taps.values()), feed_dict=feeds)
    ref = {"input": x.transpose(0, 3, 1, 2)}     # NHWC -> NCHW
    for (k, _), v in zip(taps.items(), outs):
        ref[k] = v.transpose(0, 3, 1, 2)

    import safetensors.numpy
    safetensors.numpy.save_file(tensors, os.path.join(args.model_dir, "gen_weights.safetensors"))
    with open(os.path.join(args.model_dir, "gen_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    np.savez_compressed(os.path.join(args.model_dir, "gen_reference.npz"), **ref)
    print("meta:", json.dumps(meta, indent=2))
    print(f"wrote gen_weights.safetensors ({len(tensors)} tensors), "
          f"gen_meta.json, gen_reference.npz -> {args.model_dir}")


if __name__ == "__main__":
    main()
