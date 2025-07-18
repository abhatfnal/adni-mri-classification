#!/bin/bash

# Get scan count
DATASET_SIZE=$(find ../raw/ -name "*.nii" | wc -l)

# Run parallel jobs
sbatch --array=0-$((DATASET_SIZE-1)) job.slurm

# Generate metadata
python generate_metadata.py