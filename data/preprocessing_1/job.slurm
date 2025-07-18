#!/bin/bash

#SBATCH --job-name=preprocessing_1
#SBATCH --output=./logs/preprocessing_1_%A_%a.out
#SBATCH --error=./logs/preprocessing_1_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --account=pi-aereditato

# Load your conda environment
source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate adni_rcc  

# Add lines here to run your computation on each job
python preproc_img.py ../labels/adni_matched_nii_with_labels.csv $SLURM_ARRAY_TASK_ID