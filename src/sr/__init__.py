"""Resolution-enhancement experiments (RQ A2): R0 / R1 / R2.

One LightningModule (`sr.model.JointSRUNetLightning`, a `unet.model.UNetLightning`
subclass with an SR front-end) covers all three configs so the segmentation
network, loss, data and logging are held constant and only the upsampler
treatment varies:

    R0  bicubic ×4 upsampling of the input        (upsampler="bicubic")
    R1  SEN2SR as a frozen preprocessing step     (upsampler="sen2sr", freeze_sr=True)
    R2  SEN2SR fine-tuned jointly with the U-Net  (upsampler="sen2sr", freeze_sr=False)
        via the segmentation loss ALONE (task-driven SR, Haris et al.) with a
        low LR on SEN2SR (the Figure-1 `∇L_seg × α` term = lr_sr / lr).

Train:  python -m sr.cli fit  --config src/sr/configs/joint_sr.yaml \\
                              --config src/unet/configs/norm_stats.yaml
Test:   python -m sr.cli test --config ... --ckpt_path <ckpt>
Tune:   python -m sr.tune --base-config src/sr/configs/joint_sr.yaml \\
                          --base-config src/unet/configs/norm_stats.yaml ...
"""
