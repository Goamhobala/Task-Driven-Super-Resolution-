# Tiled S2-ROSA dataset (torchgeo)

```bash
PYTHONPATH=src python -m sentinel2data.cli tile \
  --imagery-dir  /data/S2ROSA/imagery \
  --output-dir   /data/S2ROSA_tiled \
  --roads        roads_major_medium.parquet \
  --biome-parquet nvm2024_t_biome.parquet      # optional; from scripts/biome.py convert
# --tile-size 512  --patch-size 256  --buffer-m 5
```

Per source COG (`DatasetTiler`, `processor/tiler.py`):

1. rasterise the road vector to a full-zone mask (`RoadMaskGenerator`,
   per-class buffer widths);
2. walk a grid of `tile_size` windows — **partial edge windows are dropped**
   (only full `tile_size×tile_size` tiles survive, so each divides cleanly into
   `patch_size` patches);
3. **drop any tile whose mask is all-zero** (no roads);
4. write each kept tile as an internally-tiled (`patch_size` blocks) image COG
   (`images/<zone>_r<r>_c<c>.tif`, 20-band float32) + mask COG
   (`masks/...`, uint8 {0,1}).

`tile_size` must be a whole multiple of `patch_size`.

### Output

```
<output-dir>/
  images/<zone>_r<r>_c<c>.tif     # 20-band float32 COG, 512², 256 internal blocks
  masks/<zone>_r<r>_c<c>.tif      # 1-band uint8 COG, nodata=0
  metadata.parquet                # ONE ROW PER TILE
  splits/{train,val,test}.csv     # zone-level split (no scene leakage)
```

`metadata.parquet` is **per tile** (not per internal patch): the torchgeo
sampler carves patches at read time, so the catalogue only describes whole
tiles — `tile_id, zone_name, tile_row, tile_col, image_path, mask_path,
tile_size, patch_size, spatial_resolution, road_pixels, road_density, biome,
split_set, satellite_image_dates, crs, geometry` (geometry = tile footprint in
EPSG:4326). `biome` is the NVM2024 `T_BIOME` the tile centroid falls in
(`processor/biome.py`, vectorised port of `scripts/biome.py`); `"Unknown"` when
no biome parquet is given or the tile is outside the map.

## Consumer — torchgeo loader (`torchgeo_dataset.py`)

```python
from sentinel2data.torchgeo_dataset import get_dataloader, S2_BANDS, RGB_BANDS

train = get_dataloader("/data/S2ROSA_tiled", split="train",
                       bands=RGB_BANDS, patch_size=256, batch_size=8)
val   = get_dataloader("/data/S2ROSA_tiled", split="val")   # GridGeoSampler (dense)

batch = next(iter(train))   # {'image': Bx3x256x256 f32, 'mask': Bx256x256 long, 'bounds', 'crs'}
```

- `S2RosaImage`/`S2RosaMask` are `RasterDataset`s intersected spatially
  (`image & mask`); `bands` selects from the 20 `S2_BANDS` (default RGB =
  `B4,B3,B2`, pass `S2_BANDS` for all 20).
- train → `RandomGeoSampler`; val/test → `GridGeoSampler` (stride =
  `patch_size`, i.e. non-overlapping full coverage).
- Source zones span UTM 33S–36S; all tiles in a split are reprojected on read to
  the first tile's CRS, so cross-UTM patches are not strictly pixel-aligned.

Smoke test: `PYTHONPATH=src python -m sentinel2data.torchgeo_dataset <dir> --split train [--all-bands]`.

## Consumer — UNet baseline (`src/unet`)

`src/unet/geo_dataset.py` `ROSAGeoDataModule` adapts this loader to the UNet
model's `(image, mask, filename)` tuple (per-image standardisation; mask →
`(B,1,H,W)` float; `bands` stay 1-based COG indices). `train.py` uses it by
**default** (Kaggle dataset `kelvinwei/s2rosa-v2`); `--legacy-loader` restores
the old per-patch `ROSADataModule`.

```bash
PYTHONPATH=src python -m unet.train <dataset_dir> --bands 1,2,3 --epochs 20
# splits come from the dataset's splits/*.csv; --length sets train patches/epoch
```
