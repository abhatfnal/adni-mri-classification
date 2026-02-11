#!/bin/bash

# --- Configuration ---
# Source directory where your large .zip files are located
SOURCE_DIR="/project/aereditato/abhat/ADNI/Original"

# The new directory where you want to store the unzipped PET scans
DEST_DIR="/project/aereditato/abhat/adni-mri-classification/data/pet_scans_raw"

# The names of your two PET zip files
ZIP_FILE_1="ADNI_all_PET_Original_1.zip"
ZIP_FILE_2="ADNI_all_PET_Original_2.zip"


# --- Script ---
echo "Creating destination directory: ${DEST_DIR}"
mkdir -p "${DEST_DIR}"

echo "Starting to unzip ${ZIP_FILE_1}..."
unzip "${SOURCE_DIR}/${ZIP_FILE_1}" -d "${DEST_DIR}"
echo "Finished unzipping ${ZIP_FILE_1}."

echo "Starting to unzip ${ZIP_FILE_2}..."
unzip "${SOURCE_DIR}/${ZIP_FILE_2}" -d "${DEST_DIR}"
echo "Finished unzipping ${ZIP_FILE_2}."

echo "All PET scans have been unzipped into ${DEST_DIR}"