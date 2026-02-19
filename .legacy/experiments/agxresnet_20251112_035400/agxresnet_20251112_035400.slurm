#!/bin/bash
#SBATCH --job-name=agxresnet_20251112_035400
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=./experiments/agxresnet_20251112_035400/agxresnet_20251112_035400_%j.out
#SBATCH --error=./experiments/agxresnet_20251112_035400/agxresnet_20251112_035400_%j.err
#SBATCH --account=pi-aereditato

# Load your conda environment
source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate adni_rcc   

# Optional: Print environment info
echo "Using Python from: $(which python)"
nvidia-smi
echo "Starting job at $(date)"

# Go to project directory
cd /project/aereditato/cestari/adni-mri-classification

# Run training using the same experiment directory
python /project/aereditato/cestari/adni-mri-classification/train.py --config /project/aereditato/cestari/adni-mri-classification/experiments/agxresnet_20251112_035400/config.yaml --dir ./experiments/agxresnet_20251112_035400

echo "Finished at $(date)