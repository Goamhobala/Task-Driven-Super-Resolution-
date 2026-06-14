import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.windows import transform as window_transform
from shapely.geometry import box, GeometryCollection
from pathlib import Path


class RoadMaskGenerator:
    """Derives road artifacts for one satellite COG from Overture road vectors.
    """

    def __init__(self, sat_cog_path, overture_parquet_path, out_mask_path, out_graph_path, buffer_m=10):
        self.sat_cog_path = Path(sat_cog_path)
        self.overture_parquet_path = Path(overture_parquet_path)
        self.out_mask_path = Path(out_mask_path)
        self.out_graph_path = Path(out_graph_path)
        self.buffer_m = buffer_m

        # store metadata and road graphs
        self.sat_meta = self._read_sat_meta()
        self.roads = self._load_roads()

    def _read_sat_meta(self):
        with rasterio.open(self.sat_cog_path) as src:
            sat_meta = {
                "meta": src.meta.copy(),
                "crs": src.crs,
                "bounds": src.bounds,
                "blockxsize": src.profile.get("blockxsize", 256),
                "blockysize": src.profile.get("blockysize", 256),
            }
        return sat_meta

    def _load_roads(self):
        sat_crs = self.sat_meta["crs"]
        footprint = box(*self.sat_meta["bounds"])

        footprint_gdf = gpd.GeoDataFrame({"geometry": [footprint]}, crs=sat_crs)
        bbox_4326 = footprint_gdf.to_crs("EPSG:4326").total_bounds

        print("Filtering Overture parquet data...")
        roads = gpd.read_parquet(self.overture_parquet_path, bbox=tuple(bbox_4326))

        if roads.empty:
            print("Warning: no roads found in footprint.")
            return gpd.GeoDataFrame({"geometry": []}, crs=sat_crs)

        print(f"Reprojecting and clipping {len(roads)} road segments...")
        roads = roads.to_crs(sat_crs)
        return gpd.clip(roads, footprint)

    def _clip_roads(self, patch_box, sindex):
        """Road centrelines intersected with one patch; empty geometry if none."""
        if sindex is None:
            return GeometryCollection()
        candidates = self.roads.iloc[list(sindex.query(patch_box))]
        if candidates.empty:
            return GeometryCollection()
        clipped = candidates.intersection(patch_box)
        clipped = clipped[~clipped.is_empty]
        if clipped.empty:
            return GeometryCollection()
        return clipped.union_all()


    def generate_raster_mask(self):
        """Rasterize the buffered roads into a binary, internally-tiled,
        LZW-compressed mask COG aligned to the satellite imagery."""
        sat_meta = self.sat_meta["meta"]
        roads = self.roads

        if roads.empty:
            print("No roads to rasterize. Creating an empty mask.")
            mask = np.zeros((sat_meta["height"], sat_meta["width"]), dtype="uint8")
        else:
            print(f"Rasterizing roads with a {self.buffer_m}m buffer...")
            buffered = roads.geometry.buffer(self.buffer_m)
            shapes = ((geom, 1) for geom in buffered)
            mask = features.rasterize(
                shapes=shapes,
                out_shape=(sat_meta["height"], sat_meta["width"]),
                transform=sat_meta["transform"],
                fill=0,
                all_touched=True,
                dtype="uint8",
            )

        sat_meta.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="lzw",
            tiled=True,
            blockxsize=self.sat_meta["blockxsize"],
            blockysize=self.sat_meta["blockysize"],
        )

        self.out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing tiled road-mask COG to {self.out_mask_path}...")
        with rasterio.open(self.out_mask_path, "w", **sat_meta) as dest:
            dest.write(mask, 1)
        print("Raster mask complete.")
        return self.out_mask_path

    def generate_road_graph(self):
        """Split the road network into patches aligned to the COG's internal
        tiling and write one parquet row per patch.
        Returns the path of the written parquet.
        """
        crs = self.sat_meta["crs"]
        block_w = self.sat_meta["blockxsize"]
        block_h = self.sat_meta["blockysize"]
        sindex = self.roads.sindex if not self.roads.empty else None

        print(f"Extracting patch-aligned road graphs (block {block_w}x{block_h})...")
        records = []
        with rasterio.open(self.sat_cog_path) as src:
            for ji, window in src.block_windows(1):
                ptf = window_transform(window, src.transform)
                minx = ptf.c
                maxy = ptf.f
                maxx = minx + window.width * ptf.a
                miny = maxy + window.height * ptf.e
                patch_box = box(minx, miny, maxx, maxy)

                records.append(
                    {
                        "patch_row_id": ji[0],
                        "patch_col_id": ji[1],
                        "geometry": self._clip_roads(patch_box, sindex),
                    }
                )

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

        self.out_graph_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {len(gdf)} patch graphs to {self.out_graph_path}...")
        gdf.to_parquet(self.out_graph_path)
        print("Road graph complete.")
        return self.out_graph_path