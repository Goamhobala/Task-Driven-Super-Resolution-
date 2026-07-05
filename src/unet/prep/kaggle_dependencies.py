"""Download the S2-ROSA dataset on Kaggle and symlink it into the repo.

After this runs, ``dataset/s2rosa`` points at the downloaded dataset root, which
is the default ``--dataset-dir`` used by train.py / inference.py on Kaggle.
"""

from pathlib import Path

import kagglehub

# Repo-local directory that holds dataset symlinks.
dataset_root = Path("/kaggle/working/InstaRoadPrototype/dataset")
dataset_root.mkdir(parents=True, exist_ok=True)

# S2-ROSA dataset (imagery/, masks_raster/, metadata.parquet, splits/) from Kaggle.
s2rosa_path = kagglehub.dataset_download("kelvinwei/s2rosa-v2")

link = dataset_root / "s2rosa"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(Path(s2rosa_path))

print("S2-ROSA dataset path:", s2rosa_path)
print("Linked to:", link)
