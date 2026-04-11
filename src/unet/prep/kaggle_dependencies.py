import kagglehub
from pathlib import Path

# Prepares Sentinel2 dataset
# dataset creates a link to dataset folder in instaroadprototype directory. Dataset is stored in sentinel2_1024 folder.

# Base dataset path
dataset_path = Path("/kaggle/working/InstaRoadPrototype/dataset/sentinel2")

# sentinel-2 dataset (upscaled to 1024x1024) from Kaggle
# sentinel2_dataset = kagglehub.dataset_download("kelvinwei/sentinel2-dataset")
# linked_sentinel2_dst = dataset_path / "sentinel2_1024"
# linked_sentinel2_dst.symlink_to(Path(sentinel2_dataset))
# print("Sentinel-2 Roads Dataset Path:", sentinel2_dataset)

# sentinel-2 original dataset
sentinel2_dataset = kagglehub.dataset_download("sonisuyash/sentinel-2-roads-dataset")
linked_sentinel2_dst = dataset_path / "sentinel2_256"
linked_sentinel2_dst.symlink_to(Path(sentinel2_dataset))
print("Sentinel-2 Roads Dataset Path:", sentinel2_dataset)