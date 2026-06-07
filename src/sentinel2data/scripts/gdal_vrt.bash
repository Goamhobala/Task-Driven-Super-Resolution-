#!/bin/bash

# 1. Check if the user provided an input folder
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_images_folder>"
    echo "Example: $0 InstaRoad_Dataset/images"
    exit 1
fi

BASE_DIR="$1"

# 2. Check if the directory actually exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' does not exist."
    exit 1
fi

echo "Starting VRT generation in: $BASE_DIR"
echo "----------------------------------------"

# 3. Create temporary files to hold our file lists
# mktemp safely creates a unique temp file in your OS temp directory
TEMP_TIF_LIST=$(mktemp)
TEMP_VRT_LIST=$(mktemp)

# Safety feature: Ensure temp files are deleted when the script exits/crashes
trap 'rm -f "$TEMP_TIF_LIST" "$TEMP_VRT_LIST"' EXIT

# 4. Iterate through every sub-folder (Urban, Rural, etc.)
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r CLASS_DIR; do
    CLASS_NAME=$(basename "$CLASS_DIR")
    VRT_OUTPUT="$BASE_DIR/${CLASS_NAME}_Merged.vrt"

    echo "Processing class: $CLASS_NAME..."

    # Write all .tif file paths in this folder to the temporary text file
    find "$CLASS_DIR" -name "*.tif" ! -name "._*" > "$TEMP_TIF_LIST"

    # Check if the temp file has any contents (i.e., if any tifs were found)
    if [ -s "$TEMP_TIF_LIST" ]; then
        # Count lines to see how many images we found
        FILE_COUNT=$(wc -l < "$TEMP_TIF_LIST" | tr -d ' ')
        echo " -> Found $FILE_COUNT images. Building VRT..."

        # -q suppresses the massive wall of text output from GDAL
        gdalbuildvrt -input_file_list "$TEMP_TIF_LIST" "$VRT_OUTPUT" -q

        if [ $? -eq 0 ]; then
            echo " -> Successfully created $VRT_OUTPUT"
            # Append this successful VRT to our Master List temp file
            echo "$VRT_OUTPUT" >> "$TEMP_VRT_LIST"
        else
            echo " -> Warning: Failed to create VRT for $CLASS_NAME"
        fi
    else
        echo " -> No .tif files found in $CLASS_DIR. Skipping."
    fi
    echo ""
done

# 5. Build the overarching Master VRT
echo "----------------------------------------"
if [ -s "$TEMP_VRT_LIST" ]; then
    MASTER_VRT="$BASE_DIR/Master_Dataset.vrt"
    echo "Building Level 2 Master VRT: $MASTER_VRT..."

    gdalbuildvrt -input_file_list "$TEMP_VRT_LIST" "$MASTER_VRT" -q

    if [ $? -eq 0 ]; then
        echo "Success! Master VRT created. Drag $MASTER_VRT into QGIS."
    else
        echo "Error creating Master VRT."
    fi
else
    echo "No Class VRTs were generated. Skipping Master VRT."
fi