from dlinknet.networks.dinknet import DinkNet34


def build_model(in_channels=3):
    # designed to receive 1024x1024 images, outputs preds after sigmoid
    return DinkNet34(num_classes=1, num_channels=in_channels)
