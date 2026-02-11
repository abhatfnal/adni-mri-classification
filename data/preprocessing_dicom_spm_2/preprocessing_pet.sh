#!/bin/bash

# Two-stage debug pipeline:
#   1) Extract a small sample of PET DICOMs from a big ADNI zip (into a fresh per-run dir).
#   2) Convert → find MRI from master CSV → ensure deformation → coreg+normalize in SPM12.
#
# This version creates RUN_DIR=./pet_scans_raw_test/run_YYYYmmdd_HHMMSS
# and records just-extracted series in MANIFEST so Stage 2 only sees those.

# --------------------------- Config ---------------------------
SMALL_PET_ZIP="/project/aereditato/abhat/ADNI/Original/ADNI_all_PET_Original_2.zip"
MRI_MASTER_CSV="/project/aereditato/abhat/adni-mri-classification/data/preprocessing_dicom_spm_2/FINAL_master_dataset_3T.csv"
PET_OUT_DIR="/project/aereditato/abhat/adni-mri-classification/data/processed_pet"

# Root where per-run extractions live
EXTRACT_ROOT="./pet_scans_raw_test"
NUM_SCANS_TO_TEST=50

# MATLAB helpers (.m files) directory
MATLAB_M_DIR="/project/aereditato/abhat/adni-mri-classification/data/preprocessing_dicom_spm_2"

# Python utils
UTIL_DIR="/project/aereditato/abhat/adni-mri-classification/utils"
GET_PET_SCANS_PY="${UTIL_DIR}/get_pet_scan_paths.py"
FIND_CORR_MRI_PY="${UTIL_DIR}/find_corresponding_mri.py"

# Optional: set CLEAN_BEFORE_RUN=1 to wipe old runs
CLEAN_BEFORE_RUN=${CLEAN_BEFORE_RUN:-0}

# ------------------------ Environment ------------------------
set -o pipefail
mkdir -p "${PET_OUT_DIR}" "${EXTRACT_ROOT}"

# Load project env (SPM_PATH, MATLAB, etc.)
# Safe if this file prints helpful context lines.
source /project/aereditato/abhat/adni-mri-classification/env_setup.sh

# Tool checks
if ! command -v dcm2niix &>/dev/null; then
  echo "ERROR: dcm2niix command not found. Try: conda install -c conda-forge dcm2niix"
  exit 1
fi
if ! command -v matlab &>/dev/null; then
  echo "ERROR: matlab not found in PATH."
  exit 1
fi
echo "SUCCESS: dcm2niix and matlab are available."

# --------------------- Per-run directories -------------------
if [[ "${CLEAN_BEFORE_RUN}" == "1" ]]; then
  rm -rf "${EXTRACT_ROOT:?}/"*
fi

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${EXTRACT_ROOT}/run_${RUN_TAG}"
MANIFEST="${RUN_DIR}/manifest.txt"
mkdir -p "${RUN_DIR}"
: > "${MANIFEST}"

# ---------------------- Stage 1: Extract ---------------------
echo
echo "--- STAGE 1: Extracting up to ${NUM_SCANS_TO_TEST} PET series ---"

# get list of series paths inside the zip (e.g., ADNI/007_S_10075/AX_BRAIN_Florbetapir_NF/)
mapfile -t ALL_SERIES < <(python "${GET_PET_SCANS_PY}" "${SMALL_PET_ZIP}")
if [[ "${#ALL_SERIES[@]}" -eq 0 ]]; then
  echo "ERROR: No PET series found in ${SMALL_PET_ZIP}"
  exit 1
fi

count=0
for series_in_zip in "${ALL_SERIES[@]}"; do
  echo "Extracting: ${series_in_zip}"
  # Ensure a clean target series directory in this run
  series_dir="${RUN_DIR}/${series_in_zip%/}"
  mkdir -p "$(dirname "${series_dir}")"
  rm -rf "${series_dir}"

  # Extract just this series (and its sub-entries) into RUN_DIR
  unzip -q -o "${SMALL_PET_ZIP}" "${series_in_zip%/}*" -d "${RUN_DIR}"

  n_files=$(find "${series_dir}" -type f | wc -l)
  if [[ "${n_files}" -eq 0 ]]; then
    echo "WARNING: No files extracted for ${series_in_zip}. Check zip path/pattern."
  fi


  # Record for Stage 2
  echo "${series_dir}" >> "${MANIFEST}"

  count=$((count+1))
  if [[ "${count}" -ge "${NUM_SCANS_TO_TEST}" ]]; then
    break
  fi
done
echo "SUCCESS: Extracted ${count} series into ${RUN_DIR}"

# --------------------- Stage 2: Process ----------------------
echo
echo "--- STAGE 2: Processing extracted PET series ---"

if [[ ! -s "${MANIFEST}" ]]; then
  echo "Nothing extracted for this run; nothing to process."
  exit 0
fi

mapfile -t SERIES_TO_PROCESS < "${MANIFEST}"

for scan_dir in "${SERIES_TO_PROCESS[@]}"; do
  echo
  echo "--- Processing directory: ${scan_dir} ---"

  # 1) DICOM → NIfTI (+ JSON) with recursion (handles nested date/I#### folders)
  dcm2niix -r y -b y -z n -o "${scan_dir}" "${scan_dir}" >/dev/null 2>&1

  # pick the newest .nii we just created
  pet_nii_path=$(find "${scan_dir}" -maxdepth 1 -type f -name '*.nii' -printf '%T@ %p\n' \
                 | sort -nr | head -n1 | awk '{print $2}')
  if [[ -z "${pet_nii_path}" ]]; then
    echo "WARNING: dcm2niix failed for ${scan_dir}. Skipping."
    continue
  fi

  # 2) Match to MRI via master CSV (RID/PTID + closest date)
  corresponding_mri=$(python -W ignore "${FIND_CORR_MRI_PY}" "${pet_nii_path}" "${MRI_MASTER_CSV}" 2>/dev/null | tail -n 1)
  if [[ -z "${corresponding_mri}" ]]; then
    echo "WARNING: No corresponding MRI found for ${pet_nii_path}. Skipping."
    continue
  fi
  echo "Found corresponding MRI: ${corresponding_mri}"

  # 3) Ensure SPM12 deformation field (y_* or iy_*) exists next to the MRI
  mri_dirname=$(dirname "${corresponding_mri}")
  mri_stem=$(basename "${corresponding_mri}" .nii)
  cand_y="${mri_dirname}/y_${mri_stem}.nii"
  cand_iy="${mri_dirname}/iy_${mri_stem}.nii"

  if [[ ! -f "${cand_y}" && ! -f "${cand_iy}" ]]; then
    echo "No deformation field found. Running SPM12 segmentation to create one..."
    matlab -nodisplay -nosplash -r "try; addpath('${MATLAB_M_DIR}'); ensure_deformation_field('${corresponding_mri}'); catch e; disp(getReport(e)); exit(1); end; exit(0);" >/dev/null
  fi

  if   [[ -f "${cand_y}"  ]]; then deformation_field="${cand_y}"
  elif [[ -f "${cand_iy}" ]]; then deformation_field="${cand_iy}"
  else
    echo "WARNING: Still no SPM12 deformation field after segmentation. Skipping."
    continue
  fi
  echo "Using Deformation field: ${deformation_field}"

  # 4) Coreg PET→MRI, then normalize PET to MNI with the deformation field
  matlab_output=$(matlab -nodisplay -nosplash -r "try; addpath('${MATLAB_M_DIR}'); out=coregister_and_normalize_pet('${corresponding_mri}','${pet_nii_path}','${deformation_field}'); fprintf('FINAL_OUTPUT:%s\n', out); catch e; disp(getReport(e)); exit(1); end; exit(0);")
  if [[ $? -ne 0 ]]; then
    echo "WARNING: MATLAB/SPM processing failed. Skipping."
    echo "${matlab_output}"
    continue
  fi

  processed_pet_path=$(echo "${matlab_output}" | grep "FINAL_OUTPUT:" | sed 's/.*FINAL_OUTPUT://')
  if [[ -z "${processed_pet_path}" || ! -f "${processed_pet_path}" ]]; then
    echo "WARNING: Could not find final processed file from MATLAB output. Skipping."
    echo "${matlab_output}"
    continue
  fi

  # 5) Move to canonical output location (MNI-space PET)
  final_filename="$(basename "${pet_nii_path}" .nii)_processed.nii"
  mv -f "${processed_pet_path}" "${PET_OUT_DIR}/${final_filename}"
  echo "Successfully processed. Final file: ${PET_OUT_DIR}/${final_filename}"

  # Optional QC hook:
  # rpet=$(dirname "${processed_pet_path}")/$(basename "${processed_pet_path}" | sed 's|^w_||')
  # matlab -nodisplay -nosplash -r "try; addpath('${MATLAB_M_DIR}'); qc_coreg_metrics('${corresponding_mri}','${rpet}'); catch; end; exit(0);" >/dev/null
done

echo
echo "--- Finished ---"
echo "Run dir: ${RUN_DIR}"
echo "Processed PETs: ${PET_OUT_DIR} (look for *_processed.nii)"
