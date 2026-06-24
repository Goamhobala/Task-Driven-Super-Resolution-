import kagglehub
from pathlib import Path
from huggingface_hub import hf_hub_download

# ── Sentinel-2 Roads Dataset ──────────────────────────────────────────────────
# Downloads the dataset and symlinks it into the project directory so that
# train.py / inference.py can find it at the expected paths.

dataset_path = Path("/kaggle/working/InstaRoadPrototype/dataset/sentinel2")
dataset_path.mkdir(parents=True, exist_ok=True)

sentinel2_dataset    = kagglehub.dataset_download("sonisuyash/sentinel-2-roads-dataset")
linked_sentinel2_dst = dataset_path / "sentinel2_256"

if not linked_sentinel2_dst.exists():
    linked_sentinel2_dst.symlink_to(Path(sentinel2_dataset))

print("Sentinel-2 Roads Dataset Path:", sentinel2_dataset)
print("Symlink created at:", linked_sentinel2_dst)
