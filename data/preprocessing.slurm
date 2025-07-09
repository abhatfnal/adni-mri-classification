#!/bin/bash
#SBATCH --job-name=data_preprocessing
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./logs/data_preprocessing_%j.out
#SBATCH --account=pi-aereditato

# Load your conda environment
source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate adni_rcc   # ✅ use the new environment

# Optional: Print environment info
echo "Using Python from: $(which python)"
echo "Starting job at $(date)"

# Go to project directory
cd /project/aereditato/cestari/adni-mri-classification/data

# Match the no. of threads with the no. of CPU cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 

# Run parallel preprocessing
python preprocessing.py --job

echo "Finished at $(date)"
