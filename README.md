# ADNI MRI Classification

Explainable AI for Alzheimer’s Disease classification using 3D MRI volumes from the ADNI dataset.

## Pipeline

- Preprocessing of MRI volumes → uniform resampling & normalization
- Metadata matching (diagnosis labels)
- Baseline 3D CNN training (classification: CN, MCI, AD)
- Explainability via Grad-CAM

## Files

- `preprocessing.py` — preprocess MRI volumes
- `generate_preprocessed_metadata_csv.py` — match metadata with preprocessed files
- `adni_dataset.py` — dataset loader for PyTorch
- `model_3dcnn.py` — baseline CNN model
- `model_3dcnn_gradcam.py` — model with hooks for Grad-CAM
- `train_3dcnn.py` — training loop
- `gradcam_visualize.py` — Grad-CAM generation
- `inspect_metadata.py`, `test_dataset_loading.py` — utilities

## Dataset

- Requires ADNI MRI volumes (T1-weighted) — not included
- Download metadata CSVs (DXSUM, etc.) from ADNI portal

## Dependencies

- Python 3.10+
- PyTorch
- nibabel
- scikit-learn
- matplotlib

## Author

Avinay Bhat, University of Chicago
