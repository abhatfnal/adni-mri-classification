#!/bin/bash

# Create logs dir
mkdir -p ./logs

# Run 100 parallel jobs
sbatch --array=1-100 job.slurm ../../../downloads/ADNI/ADNI_1_3T.zip
