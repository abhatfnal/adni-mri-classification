#!/bin/bash

#SBATCH --job-name=exp_analysis
#SBATCH --output=./exp_analysis.out
#SBATCH --error=./exp_analysis.err
#SBATCH --time=00:05:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --account=pi-aereditato

EXP_DIR="./experiments/mynet_20250815_151620"
OUT_DIR="./analysis"
FOLD=1

# Load environment
source ./env_setup.sh

# Run analysis
python analysis.py --exp-dir ${EXP_DIR} --fold ${FOLD} --output-dir ${OUT_DIR}
