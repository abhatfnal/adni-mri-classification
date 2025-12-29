#!/bin/bash
#SBATCH --array=1-100%100
#SBATCH --job-name=preprocessing_pet
#SBATCH --output=./logs/preprocessing_mri_pet_%A_%a.out
#SBATCH --error=./logs/preprocessing_mri_pet_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --partition=amd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --account=pi-aereditato

set -euo pipefail

mkdir -p ./logs

ZIP_PATH1="/project/aereditato/cestari/downloads/ADNI/Multimodal_MRI_PET_1.zip"
ZIP_PATH2="/project/aereditato/cestari/downloads/ADNI/Multimodal_MRI_PET_2.zip"
CSV_PATH="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_mri_pet/dataset.csv"
DATA_PATH="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_mri_pet/data"

source ../../env_setup.sh
cd scripts/ || exit 1

# Total work items you want to cover
TOTAL=4000
NJOBS=100
CHUNK=$(( (TOTAL + NJOBS - 1) / NJOBS ))   # ceil(TOTAL/NJOBS)

TASK0=$((SLURM_ARRAY_TASK_ID - 1))
START=$((TASK0 * CHUNK))
END=$((START + CHUNK - 1))
if [ $END -ge $((TOTAL - 1)) ]; then END=$((TOTAL - 1)); fi

echo "Task ${SLURM_ARRAY_TASK_ID}: processing indices ${START}..${END}"

for IDX in $(seq $START $END); do
  python run_pair.py "$IDX" "$CSV_PATH" "$DATA_PATH" "$ZIP_PATH1" "$ZIP_PATH2"
done
