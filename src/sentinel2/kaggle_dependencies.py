import kagglehub
# import shutil
from pathlib import Path
import os

# Prepares the samroad decoder and Sentinel2 dataset

# sam_road repository
samroad_repo = Path("/kaggle/working/InstaRoad/sam_road")

# sam_road decoder
samroad_decoder = kagglehub.dataset_download("sacuscreed/sam-vit-b-01ec64-pth")
# Linking (not copying to save disk space)
linked_samroad_decoder_dst = samroad_repo / "sam_ckpts"
linked_samroad_decoder_dst.symlink_to(Path(samroad_decoder))
print("SAM ROAD Decoder Path:", samroad_decoder)

# sentinel-2 dataset
sentinel2_dataset = kagglehub.dataset_download("kelvinwei/sentinel2-dataset")
linked_sentinel2_dst = samroad_repo / "sentinel2" / "sentinel2_test_1024"
linked_sentinel2_dst.symlink_to(Path(sentinel2_dataset))
print("Sentinel-2 Roads Dataset Path:", sentinel2_dataset)