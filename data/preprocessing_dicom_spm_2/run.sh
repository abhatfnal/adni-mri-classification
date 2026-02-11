#!/bin/bash

# Empty logs dir 
rm -rf ./logs 
mkdir -p ./logs

# Empty data dir
rm -rf ./data
mkdir ./data

FILE=$1

# Check existence and extension
if [ ! -f "$FILE" ]; then
  echo "Error: File '$FILE' does not exist."
  exit 1
fi

if [[ "${FILE##*.}" != "zip" ]]; then
  echo "Error: File '$FILE' is not a .zip archive."
  exit 1
fi

# Run 100 parallel jobs
sbatch --array=1-200 job.slurm "$FILE"