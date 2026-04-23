import ee
import math
from datetime import datetime, timezone

def authenticate(project_id):
    try:
        ee.Initialize(project=project_id)
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(project=project_id)

def main():
    authenticate("focus-copilot-488313-i4")

    # process_grid()

    test()

def test():
    # CONFIG
    START_DATE = '2025-01-01'
    END_DATE = '2026-01-01'
    GRID_STEP = 0.5 # Degrees
    MAX_CLOUD_COVER = 1 # Maximum percentage of clouds


    lat = 18.5860137
    lon = -33.8988858


    cell_bounds = ee.Geometry.Rectangle([lat-GRID_STEP, lon-GRID_STEP, lat+GRID_STEP, lon + GRID_STEP])
    # print([lat-GRID_STEP, lon-GRID_STEP, lat+GRID_STEP, lon + GRID_STEP])
    print(cell_bounds.coordinates().getInfo())

    # Calculate Native Local UTM Zone for the cell's center
    center_lon = lon + (GRID_STEP / 2)
    utm_zone = math.floor((center_lon + 180) / 6) + 1
    epsg_code = f"EPSG:327{utm_zone}" # 327 - Southern Hemisphere

    # S2 Collection
    s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(cell_bounds)
            .filterDate(START_DATE, END_DATE)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER))
            .sort('CLOUDY_PIXEL_PERCENTAGE', False) # Get the absolute clearest image
            )


    # Grab the first image from your collection
    s2_img = s2_col.first()

    band_projection = s2_img.select('B4').projection()

    # 2. Extract the standard EPSG code string (e.g., 'EPSG:32734')
    native_crs = band_projection.crs().getInfo()

    # 3. Extract the native physical pixel size (Scale)
    native_scale = band_projection.nominalScale().getInfo()

    # Extract the footprint geometry
    image_footprint = s2_img.geometry()

    # Get the raw GeoJSON coordinate array
    # (This will return a list of Longitude/Latitude pairs tracing the swath)
    boundary_coordinates = image_footprint.coordinates().getInfo()

    print("This image covers the following polygon:")
    print(boundary_coordinates)

    print(f"Native CRS: {native_crs}, Native Scale: {native_scale} meters")

def process_grid():
    # CONFIG
    START_DATE = '2025-01-01'
    END_DATE = '2026-01-01'
    DRIVE_FOLDER = 'SA_S1_S2_Road_Tiles_V2'
    GRID_STEP = 2 # Degrees
    MAX_CLOUD_COVER = 1 # Maximum percentage of clouds

    # MIN_LON, MAX_LON = 16, 34
    # MIN_LAT, MAX_LAT = -36, -22
    # Test over cape town
    MIN_LON, MAX_LON = 18, 20
    MIN_LAT, MAX_LAT = -35, -33

    #Potentially do per seasons
#     SEASONS = [
#     {'name': 'Layer_1_Q1_Jan_Mar', 'start': '2024-01-01', 'end': '2024-03-31'},
#     {'name': 'Layer_2_Q2_Apr_Jun', 'start': '2024-04-01', 'end': '2024-06-30'},
#     {'name': 'Layer_3_Q3_Jul_Sep', 'start': '2024-07-01', 'end': '2024-09-30'},
#     {'name': 'Layer_4_Q4_Oct_Dec', 'start': '2024-10-01', 'end': '2024-12-31'}
# ]


    # Trackers
    skipped_tiles = 0
    processed_tiles = 0
    metadata_log = []

    print("Starting Grid Processing...")

    # Grid loop
    for lon in range(MIN_LON, MAX_LON, GRID_STEP):
        for lat in range(MIN_LAT, MAX_LAT, GRID_STEP):

            cell_bounds = ee.Geometry.Rectangle([lon, lat, lon + GRID_STEP, lat + GRID_STEP])

            # Calculate Native Local UTM Zone for the cell's center
            center_lon = lon + (GRID_STEP / 2)
            utm_zone = math.floor((center_lon + 180) / 6) + 1
            epsg_code = f"EPSG:327{utm_zone}" # 327 - Southern Hemisphere

            # S2 Collection
            s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(cell_bounds)
                    .filterDate(START_DATE, END_DATE)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVER))
                    .sort('CLOUDY_PIXEL_PERCENTAGE', False) # Get the absolute clearest image
                    )

            # Check if an S2 image exists
            if s2_col.size().getInfo() == 0:
                skipped_tiles += 1
                print(f"Skipped Grid [{lon}, {lat}]: No cloud-free Sentinel-2 image found.")
                continue

            s2_timestamps = s2_col.aggregate_array('system:time_start').getInfo()
            s2_dates_list = sorted(list(set([
                datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                for ts in s2_timestamps
            ])))
            s2_dates_str = "[" + ", ".join(s2_dates_list) + "]"

            # Mosaic collection
            s2_img = s2_col.mosaic()

            # Extract S2 date to match S1
            s2_date = ee.Date(s2_img.get('system:time_start'))
            s2_date_str = s2_date.format('YYYY-MM-dd').getInfo()

            # S1 Radar
            # S1 image within 1 month of the S2 image
            s1_col = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterBounds(cell_bounds)
                    .filterDate(s2_date.advance(-30, 'day'), s2_date.advance(30, 'day'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    )

            if s1_col.size().getInfo() == 0:
                skipped_tiles += 1
                print(f"Skipped Grid [{lon}, {lat}]: No matching Sentinel-1 image within 30 days.")
                continue

            s1_first = s1_col.first()
            s1_date_str = ee.Date(s1_first.get('system:time_start')).format('YYYY-MM-dd').getInfo()

            s1_img = s1_col.mosaic()
            s1_date_str = ee.Date(s1_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

            # Stacking
            # Select the 10m bands from both
            s2_selected = s2_img.select(['B4', 'B3', 'B2', 'B8', 'SCL']) # R, G, B, NIR, Classification
            s1_selected = s1_img.select(['VV', 'VH'])

            # Combine them into a 7-band image
            stacked_img = ee.Image.cat([s2_selected, s1_selected]).clip(cell_bounds)

            # Masking just in case
            s2_mask = s2_selected.select('B4').mask()
            s1_mask = s1_selected.select('VV').mask()
            common_mask = s2_mask.And(s1_mask)

            # Convert to float
            stacked_img = stacked_img.updateMask(common_mask).toFloat()

            # Export to google drive
            grid_id = f"SA_Grid_{lon}_{lat}"

            task = ee.batch.Export.image.toDrive(
                image=stacked_img,
                description=grid_id,
                folder=DRIVE_FOLDER,
                region=cell_bounds.getInfo()['coordinates'],
                scale=10,                      # Force 10-meter resolution
                crs=epsg_code,                 # Project into local UTM
                maxPixels=1e13,
                fileDimensions=[256, 256],     # Splits into 256x256 tiles
                skipEmptyTiles=True            # Ignores tiles that fall over the edge of the image mask
            )
            task.start()

            # Log metadata
            processed_tiles += 1
            log_entry = f"Grid [{lon}, {lat}] -> UTM {epsg_code} | S2 Date: {s2_date_str} | S1 Date: {s1_date_str}"
            metadata_log.append(log_entry)
            print(f"Task Started: {log_entry}")

    print("\n" + "="*40)
    print("PROCESSING SUMMARY")
    print("="*40)
    print(f"Tasks submitted to Google Drive: {processed_tiles}")
    print(f"2-Degree Grids skipped: {skipped_tiles}")
    print("Metadata of processed grids:")
    for log in metadata_log:
        print(" - " + log)
    print("Check https://code.earthengine.google.com/tasks to monitor export progress.")

if __name__ == "__main__":
    main()

# potentially can also mask out sections
def mask_s2_clouds(image):
    scl = image.select("SCL")
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(clear)