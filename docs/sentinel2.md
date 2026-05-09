# Sentinel 2 Dataset Curation
Documentation detailing the curatation of Sentinel 1 & 2 imagery with road vector data.

## Overview
General process of curating the dataset and getting it ready for deep learning models.

1. **Data Aggregation** of satellite imagery and road vector data for our dataset.
2. **Semi-Automated Processing** of dataset.
3. **Upload** dataset to cloud environment.
4. **Transfer** dataset to computing resource for training
5. **InstaGeo** data pipeline replication for inference


## Data Aggregation
### Satellite Imagery

#### Google Earth Engine
Satellite Patches can be manually obtained through Google Earth Engine with the [GEE Script](https://code.earthengine.google.com/0b3fc6acd36c3d651eea522dc2ba8b25?noload=true) for the [GlobalUrbanMapper](https://github.com/LauraChow77/GlobalUrbanMapper) data engine.

Regions of interest can be specified with Google Earth Engine UI or [GeoJson](https://geojson.io).

### Road Vectors
There are a variety of sources for road vector data:

- [OSM](https://download.geofabrik.de/africa/south-africa.html)
- [Overture](https://docs.overturemaps.org/getting-data/)
- [FRST](https://figshare.com/articles/dataset/FRST_dataset/29424107)
- GRIP

Currently Overture and OSM are the most promising sources.

#### Overture
SA road vector data can be obtained using overture's CLI
``` bash
uvx overturemaps download \
  --bbox=16.2,-34.9,33.1,-22.0 \
  --type=segment \
  -f geoparquet \
  -o sa_overture_roads.parquet
```

## Semi-Automated Processing
We can batch process our dataset using the UCT's HPC or Google Colab. It is recommended to first locally run the code on a small image inspect the results for any errors/bugs.


## Upload
Uploading our dataset to a cloud environment makes it faster for computing resources to download our dataset. UCT HPC cluster might be an exception if we locally transfer the dataset on campus.

The most promising cloud storage provider is through Kaggle datasets. However, the storage limits might be a bottleneck if our datasets become greater than 200GB.

The other options is Google Cloud Bucket. Hopefully, Professor Shock can get us free credits if this is the case.

### Kaggle Datasets
Refer to [Setup Documentation](setup.md) to authenticate Kaggle CLI

First initialise the dataset folder with following command. This creates a metadata file.
``` bash
kaggle datasets init -p /example/dataset_folder
```

The metadata file ("dataset-metadata.json") is a template and needs to be edited. Locally edit the file or use python to edit the file if on Google Colab.

``` python
import json

# Define your dataset's title and unique ID (slug)
kaggle_username = userdata.get('Kaggle_username')
dataset_slug = "sentinel2-dataset" # Must be lowercase, alphanumeric, and use hyphens
dataset_title = "Sentinel2 Dataset"

# Load the generated metadata file
meta_path = '/content/sentinel2_test_1024/dataset-metadata.json'
with open(meta_path, 'r') as f:
    meta = json.load(f)

# Update the fields
meta['id'] = f"{kaggle_username}/{dataset_slug}"
meta['title'] = dataset_title

# Save the changes
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=4)

print("Metadata updated successfully!")
```

Now you can upload the dataset.
``` bash
kaggle datasets create -p /example/dataset_folder --dir-mode tar
```

#### Updating Kaggle Dataset
Updating a dataset is easier with the web UI. The CLI requires you to reupload everything.

Tip: Rather zip and upload seperate folders. So you can remove and reupload specific folders if they are changed.

``` bash
kaggle datasets version -p /example/dataset_folder -m "Updated data with new features" --dir-mode tar
```

## Transfer
Refer to [Setup Documentation](setup.md) to setup dataset for model training.

> UCT HPC seems to download kaggle datasets at about 16 MB/s

## InstaGeo Data Replication
Later during the project, we would need to replicate our GEE data engine pipeline into InstaGeo. This is required so the model can do inference on the same proprocessed data that it was trained on.

### Known Discrepancies
InstaGeo uses WGS84 / EPSG4326, which are projections based on degrees. Hence, there would be distortions depending on location. GEE uses the local EPSG projections, which is different depending on the location in South Africa.

## Misc

### How to visualise lots of S2 image tiles
To visualise all the tiles on QGIS, we build a Virtual Raster (VRT). It is also recommended to generate overviews pyramids to pre-render lower resolution versions of the aggregate tiles.  I've run into problems with generating a VRT with QGIS, hence I use GDAL generate the VRT.

Manual process to generate VRT:
``` bash
# list files in a temp file
ls -1 *.tif > file_list.txt
# generate VRT
gdalbuildvrt -input_file_list file_list.txt mosaic.vrt
```

Generate overviews:
``` bash
gdaladdo -r average mosaic.vrt 2 4 8 16 32 64 128
```

- "-r average": averaging algorithm to downsample the images
- "2 4 8...": Levels of decimataion (downsample factor) of respective overlays.
- "-srcnodata 0 -vrtnodata 0": treat black pixels (0) as transparent NoData
- "-srcnodata "0 0 0" -vrtnodata "0 0 0": same as above but for multiple bands

### Sentinel 2 Bands

### 10m bands
The native 10m Sentinel-2 bands are only 4:

- B02 = "blue"
- B03 = "green"
- B04 = "red"
- B08 = "nir broad"
