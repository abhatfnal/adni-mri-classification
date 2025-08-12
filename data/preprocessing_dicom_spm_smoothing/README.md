# Preprocessing Instructions


## 1. Run Preprocessing Script

Execute the script:
```bash
sh run.sh /path/to/adni/dicom/zip/archive.zip
```
specifying the correct path to the ADNI zip archive containing unprocessed .dcm images. This will:
- Unzip and preprocess the DICOM files into `.nii` format
- Save the preprocessed files to the `./data` directory
- Generate `./data/dataset.csv` containing labels (diagnosis) and file paths

## 2. Split the Dataset

Use the `train_test_split.py` script located in the `utils/` directory:
```bash
python ../../utils/train_test_split.py \
  --input-csv ./data/dataset.csv \
  --test-size 0.1 \
  --output-dir ./data
```

This creates:
- `./data/trainval.csv`
- `./data/test.csv`

## 3. Configure Training Inputs

For training, in your configuration file update the following fields:
```yaml
data:
  trainval_csv: /absolute/path/to/data/trainval.csv
  test_csv: /absolute/path/to/data/test.csv
```
