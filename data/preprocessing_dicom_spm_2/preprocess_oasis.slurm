#!/bin/bash

#SBATCH --job-name=preprocess_oasis
#SBATCH --output=./logs/preprocess_oasis_%A_%a.out
#SBATCH --error=./logs/preprocess_oasis_%A_%a.err
#SBATCH --time=2:30:00
#SBATCH --partition=amd
#SBATCH --ntasks=1
#SBATCH --mem=8G # Increased memory slightly for SPM's overhead
#SBATCH --account=pi-aereditato
#SBATCH --array=1-200 # Using the same number of jobs as your ADNI script

# --- Configuration ---
# Define the input directory containing the sorted AD and CN folders
INPUT_DATA_DIR="/project/aereditato/abhat/OASIS/OASIS_1/OASIS_1_Processed"

# Define the final output directory for the fully preprocessed .nii files
FINAL_OUTPUT_DIR="/project/aereditato/abhat/OASIS/OASIS_1/OASIS_1_Final"
# --- End Configuration ---

# Load environment (ensure this script loads FSL, MATLAB, and SPM)
source /project/aereditato/abhat/adni-mri-classification/env_setup.sh

module load fsl

# Create the final output directories
mkdir -p "${FINAL_OUTPUT_DIR}/AD"
mkdir -p "${FINAL_OUTPUT_DIR}/CN"

# Get a list of all the .hdr files to be processed. This is simpler than the Python script.
scan_paths=( $(find "${INPUT_DATA_DIR}" -name "*.hdr") )

# --- SLURM Array Logic (identical to your ADNI script) ---
total=${#scan_paths[@]}
num_jobs=${SLURM_ARRAY_TASK_COUNT:-200}
job_idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
chunk_size=$(( (total + num_jobs - 1) / num_jobs ))
start=$(( job_idx * chunk_size ))
end=$(( start + chunk_size < total ? start + chunk_size : total ))
# --- End SLURM Logic ---

echo "This job will process scans from index ${start} to ${end-1}"

# Loop over the subset of scans for this specific array task
for (( i = start; i < end; i++ )); do
  # Get the full path to the .hdr file
  hdr_path=${scan_paths[i]}
  echo "--- Processing: ${hdr_path} ---"

  # Determine the subject's class (AD or CN) from its parent folder
  if [[ "${hdr_path}" == *"/AD/"* ]]; then
    class="AD"
  else
    class="CN"
  fi

  # Get the base filename without the extension (e.g., "OAS1_0001_MR1_mpr_n4_anon_sbj_111")
  base_filename=$(basename "${hdr_path}" .hdr)

  # Create a unique temporary directory for this subject's processing
  # This prevents file conflicts between parallel jobs
  temp_dir=$(mktemp -d -p "$LOCAL_SCRATCH" "${base_filename}_XXXX")
  echo "Created temporary directory: ${temp_dir}"

  # --- Preprocessing Steps ---

  # 1. Convert the .hdr/.img pair to a single .nii file inside the temp directory
  # fslchfiletype changes the file type of an image.
  # We provide the base path (without extension) and it finds both .hdr and .img
  echo "Step 1: Converting Analyze to NIfTI format..."
  fslchfiletype NIFTI "${hdr_path%.*}" "${temp_dir}/${base_filename}"

  # Define the path to the new .nii file
  nii_path="${temp_dir}/${base_filename}.nii"

  # 2. Perform SPM segmentation, bias correction, and normalization
  # This calls your existing MATLAB script on the .nii file we just created.
  echo "Step 2: Running SPM normalization via MATLAB..."
  matlab -nodisplay -nosplash -r "try; normalize('${nii_path}'); catch e; disp(e.message); exit(1); end; exit(0);"

  # 3. Move the final, cleaned file to the correct output directory
  # Your MATLAB script creates a file named "clean_*.nii"
  clean_nii_path="${temp_dir}/clean_${base_filename}.nii"
  final_destination="${FINAL_OUTPUT_DIR}/${class}/${base_filename}_preprocessed.nii"

  if [ -f "${clean_nii_path}" ]; then
    echo "Step 3: Moving final file to ${final_destination}"
    mv "${clean_nii_path}" "${final_destination}"
  else
    echo "ERROR: Final cleaned file was not found at ${clean_nii_path}"
  fi

  # 4. Clean up the temporary directory
  echo "Step 4: Cleaning up temporary directory..."
  rm -rf "${temp_dir}"

  echo "--- Finished: ${hdr_path} ---"
done

echo "Array job ${SLURM_ARRAY_TASK_ID} has completed."