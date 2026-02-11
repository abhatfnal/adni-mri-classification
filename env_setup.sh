#!/bin/bash

# ==================================================
# User-configurable script to set up the environment
# with Python 3.10 and MATLAB (or Octave).
# ==================================================

# 1)  Paste below the commands to activate your environment.
# -----------------------------------
# Example: Conda + MATLAB
#   
#   source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
#   conda activate adni_rcc
#   module load matlab
#

source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate adni_rcc
module load matlab

# 2) Set required environment variables
# -------------------------------------

# Absolute path to SPM12 installation folder (WITHOUT final /)
export SPM_PATH="/project/aereditato/abhat/adni-mri-classification/spm/spm"

# Absolute path to diagnosis file DXSUM_05Apr2025.csv
export DIAGNOSIS_FILE="/project/aereditato/abhat/ADNI/Original/DXSUM_13Aug2025.csv"

# 3) Print setup
# --------------------------------------

echo "Environment configured:"
echo " - Python: $(which python)"
echo " - SPM_PATH: $SPM_PATH"
echo " - DIAGNOSIS_FILE: $DIAGNOSIS_FILE"