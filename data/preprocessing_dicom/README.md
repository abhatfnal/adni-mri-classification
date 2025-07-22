To perform the preprocessing follow the steps below:

- Adjust the path of the .zip file containing the .dcm ADNI files in `run.sh`

- run `>$ sh run.sh` and wait for the jobs to complete. The ./data directory with the preprocessed data will be created.


- run `python make_csv` with the right arguments to create the file ./data/dataset.csv

- from the utils directory, run `train_test_split.py` to split the dataset


Note:

- 1 = CN, 2 = MCI, 3 = AD in the DXSUM file. https://adni.loni.usc.edu/help-faqs/ask-the-experts/#qa-panel 