import kagglehub
from pathlib import Path

# Downloads the Sentinel-2 Roads Dataset and symlinks it into the project
# dataset directory so that train.py / inference.py can find it at the
# expected paths.

# Base dataset directory inside the Kaggle working directory
dataset_path = Path("/kaggle/working/InstaRoadPrototype/dataset/sentinel2")
dataset_path.mkdir(parents=True, exist_ok=True)

# sentinel-2 original dataset (256x256 tiles)
sentinel2_dataset = kagglehub.dataset_download("sonisuyash/sentinel-2-roads-dataset")
linked_sentinel2_dst = dataset_path / "sentinel2_256"

if not linked_sentinel2_dst.exists():
    linked_sentinel2_dst.symlink_to(Path(sentinel2_dataset))

print("Sentinel-2 Roads Dataset Path:", sentinel2_dataset)
print("Symlink created at:", linked_sentinel2_dst)
