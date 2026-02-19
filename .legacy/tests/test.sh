#!/bin/bash

# Simulated variables
job_idx=0
total=5  # Simulate expecting 5 files

# Create a test folder
mkdir -p test_data

# Only one job (1st): wait for all other jobs to finish and create .csv file
if [ "$job_idx" -eq 0 ]; then
  while [ $(ls -l ./test_data | grep .nii | wc -l) -ne "$total" ]; do
    sleep 1
    echo "Waiting... $(ls -l ./test_data | grep .nii | wc -l)/$total files"
  done

  echo "All files ready. Running make_csv.py"
  # Simulate your script
  echo "python ../../utils/make_csv.py dummy.csv ./test_data"
fi
