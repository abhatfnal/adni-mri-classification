#!/bin/bash

# Path of the .zip archive containing ADNI .dcm
zip_path='../../../downloads/ADNI/ADNI_1_3T.zip'

# Create logs dir
mkdir -p ./logs

# Run 100 parallel jobs
sbatch --array=1-100 job.slurm ${zip_path}