#!/bin/bash

#SBATCH --job-name=preprocessing_dicom
#SBATCH --output=./logs/preprocessing_dicom_%A_%a.out
#SBATCH --error=./logs/preprocessing_dicom_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --account=pi-aereditato

# Load environment
source ../../env_setup.sh

# Get paths of mri scans in zip file
scan_paths=( $(python ../../utils/get_scan_paths.py "$1") )

# Figure out how many scans each of the 100 tasks should handle
total=${#scan_paths[@]}
num_jobs=100
job_idx=$(( SLURM_ARRAY_TASK_ID - 1 ))                     # zero‑based
chunk_size=$(( (total + num_jobs - 1) / num_jobs ))       # ceil(total/num_jobs)
start=$(( job_idx * chunk_size ))
end=$(( start + chunk_size < total ? start + chunk_size : total ))

# Loop over the subset of scan_ids for this array task
for (( i = start; i < end; i++ )); do
  path=${scan_paths[i]}

  # Get final filename:
  # strip trailing slash, then replace all / with _
  filename="${path%/}"
  filename="${filename//\//_}"

  # Create temp folder
  mkdir -p ./data/${filename}

  # Extract slices there
  python ../../utils/extract_scan_slices.py ${path} $1 ./data/${filename}

  # Convert to .nii
  dicom2nifti -C ./data/${filename} ./data/${filename}

  # Cleanup .dcm files
  rm ./data/${filename}/*.dcm

  # Get output file path
  nii_path=$(find ./data/${filename}/ -name '*.nii')

  # Get filename of .nii file
  nii_filename=$(basename ${nii_path})

  # Perform SPM segmentation + bias correction (also writes deformation field)
  matlab -batch "normalize('./data/${filename}/${nii_filename}')"

  # Perform z-score normalization and handle NaNs.
  python ../../utils/intensity_normalize.py ./data/${filename}/wmasked_${nii_filename} ./data/${filename}.nii

  # Remove subdir
  rm -rf ./data/${filename}

done

# Only one job (1st): wait for all other jobs to finish and create .csv file
if [ "$job_idx" -eq 0 ]; then
  while [ $(ls -l ./data | grep .nii | wc -l) -ne "$total" ]; do
    sleep 10
    echo "Waiting... $(ls -l ./data | grep .nii | wc -l)/$total files"
  done

  echo "All files ready. Running match_diagnosis.py"

  python ../../utils/match_diagnosis.py --diagnosis-file "$DIAGNOSIS_FILE" --scan-dir ./data
fi