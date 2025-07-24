To perform the preprocessing follow the steps below:

- Configure the path of the .zip file containing the .dcm ADNI files in `run.sh`

- Configure the path to the SPM folder and to the template .nii file (tpm path) in `normalize.m`

- run `$ sh run.sh` and wait for the jobs to complete. The ./data directory with the preprocessed data will be created.

- run `python ./make_csv.py` with the right arguments to create the file ./data/dataset.csv

- From the utils directory, run `python train_test_split.py` to split the dataset


Note:

- 1 = CN, 2 = MCI, 3 = AD in the DXSUM file. https://adni.loni.usc.edu/help-faqs/ask-the-experts/#qa-panel 