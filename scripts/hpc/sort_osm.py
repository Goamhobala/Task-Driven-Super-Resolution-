"""Sort flat OSM mask folders into a ROSA dataset's per-split layout.
 
Takes masks named {zone}_r{r}_c{c}.tif from a flat folder (e.g.
InstaRoad/masks_10m/) and moves each into the split subfolder of the tile it
belongs to, using the dataset's split CSVs to decide train/val/test:
 
    <masks-dir>/Durban_r2_c0.tif  ->  <dataset-dir>/train/<dest-dirname>/Durban_r2_c0.tif
 
Usage:
    python sort_osm_masks.py --dataset-dir /scratch/$USER/InstaRoad/ROSA_Dense_CDNGI \
        --masks-dir /scratch/$USER/InstaRoad/masks_10m [--dest-dirname masks_raster]
        [--copy] [--overwrite]
 
Default dest-dirname is masks_raster (files land beside imagery/). Existing
files are never clobbered unless --overwrite. --copy keeps the originals.
"""
import argparse
import csv
import shutil
from pathlib import Path
 
 
def split_of_tiles(dataset_dir):
    """{tile_stem: split} from the split CSVs (image_path column)."""
    mapping = {}
    for csv_path in sorted((dataset_dir / "splits").glob("*.csv")):
        split = csv_path.stem
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                mapping[Path(row["image_path"]).stem] = split
    if not mapping:
        raise SystemExit(f"No split CSVs with tiles under {dataset_dir}/splits/")
    return mapping
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, help="ROSA dataset root (has splits/)")
    ap.add_argument("--masks-dir", required=True, help="flat folder of {tile}.tif masks")
    ap.add_argument("--dest-dirname", default="masks_osm_raster",
                    help="folder name inside each <split>/ (default masks_raster)")
    ap.add_argument("--copy", action="store_true", help="copy instead of move")
    ap.add_argument("--overwrite", action="store_true", help="replace existing files")
    args = ap.parse_args()
 
    dataset_dir, masks_dir = Path(args.dataset_dir), Path(args.masks_dir)
    tile_split = split_of_tiles(dataset_dir)
 
    moved, skipped, unmatched = 0, 0, []
    for src in sorted(masks_dir.glob("*.tif")):
        split = tile_split.get(src.stem)
        if split is None:
            unmatched.append(src.name)
            continue
        dest = dataset_dir / split / args.dest_dirname / src.name
        if dest.exists() and not args.overwrite:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copy2 if args.copy else shutil.move)(str(src), str(dest))
        moved += 1
 
    verb = "copied" if args.copy else "moved"
    print(f"{verb} {moved}, skipped {skipped} existing, "
          f"{len(unmatched)} not in any split CSV")
    if unmatched:
        print("  unmatched (left in place):", ", ".join(unmatched[:10]),
              "..." if len(unmatched) > 10 else "")
    # tiles that still have no mask at the destination
    missing = [t for t, s in tile_split.items()
               if not (dataset_dir / s / args.dest_dirname / f"{t}.tif").exists()]
    if missing:
        print(f"WARNING: {len(missing)} tiles have no mask in <split>/{args.dest_dirname}/ "
              f"(first: {missing[0]})")
 
 
if __name__ == "__main__":
    main()