# ADNI MRI Classification

Framework for training and evaluating deep learning models on the ADNI MRI dataset. This repository provides:

- **Flexible configuration** via YAML files (mergeable, override-able)
- **Easy job submission** to RCC or local runs
- **Automatic experiment logging** under `experiments/`
- **Modular model registry** for custom architectures
- **Preprocessing pipelines** with reusable scripts
- **Utility tools** for dataset management and visualization

---

## 📁 Repository Structure

```
├── configs
│   ├── schema.py                # Config schema & validation
│   └── training                 # YAML configs for various models/training
│       ├── custom_3dcnn.yaml
│       ├── default.yaml
│       ├── example.yaml
│       ├── resnet18.yaml
│       └── simple_3dcnn.yaml
├── data
│   ├── augmentation.py
│   ├── datasets.py
│   ├── DXSUM_05Apr2025.csv      # ADNI subject metadata
│   ├── preprocessing_dicom
│   │   ├── data/                # Preprocessed .nii files + CSV labels
│   │   └── run.sh               # Runs preprocessing
│   ├── preprocessing_dicom_spm  # Alternative SPM-based pipeline
│   └── ...
├── experiments/                 # Auto-created subfolders on each run: logs, metrics, configs
├── guide.md                     # (This document)
├── models
│   ├── base.py                  # `BaseModel` abstract class
│   ├── model_custom_3dcnn.py
│   ├── model_resnet18.py
│   ├── model_simple_3dcnn.py
│   ├── model_simple_3dcnn_gradcam.py
│   └── registry.py              # Maps model names → classes
├── train.py                     # CLI entrypoint
└── utils
    ├── ...
    └── train_test_split.py      # Split CSV into train/val & test sets
```

---

## 📦 Requirements

- Python 3.10
- Matlab
- [SPM 12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
---

## 🚀 Quick Start

1. **Install requirements** (e.g. in a Conda env):

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare data**  
   Follow the instructions in `data/preprocessing_dicom/` (or the SPM variant) to generate your `.nii` files and labels CSV under `data/.../data/`.

3. **Split your dataset**:  
   ```bash
   python utils/train_test_split.py      --input-csv /path/to/full_dataset.csv      --output-dir /path/to/preprocessed/data
   ```

4. **Run training**:

   ```bash
   python train.py      --config configs/training/simple_3dcnn.yaml      --config configs/training/custom_3dcnn.yaml      --config configs/training/override.yaml      --job          # omit for local run
   ```

   - You can pass **multiple** `--config` files; they are merged in order.
   - To override a single field on the CLI:
     ```
     python train.py --config default.yaml training.epochs=500
     ```

---

## ⚙️ Configuration

All default settings live in `configs/training/default.yaml`. A full example:

```yaml
training:
  batch_size: 32
  epochs: 200
  optimizer:
    name: adam
    weight_decay: 0
    lr: 1e-3
  scheduler:
    name: CosineAnnealingLR
    t_max: 200
    lr_max: 1e-3
    lr_min: 2e-6

data:
  trainval_csv: /absolute/path/to/trainval.csv
  test_csv:     /absolute/path/to/test.csv

cross_validation:
  folds: 5
```

- **Merging & Overrides**  
  Later files override earlier ones. Command-line overrides (e.g. `training.epochs=500`) take highest precedence.

- **Schema validation** is enforced by `configs/schema.py`. Bad or missing fields will raise errors before any training begins.

---

## 📊 Experiment Logging

Each run creates a new folder under `experiments/`, named by timestamp or job ID. Inside you’ll find:

- A copy of the **merged YAML** config
- `losses.vs` with per-epoch  train and validation losses
- **Metrics** CSV: accuracy, AUC, per-class recall/precision
- **Confusion matrices** for each fold + overall average
- Any model checkpoints (`.pth`)

This structure makes it easy to reproduce and compare runs.

---

## 🧠 Model Registry

- To add a new model:
  1. Implement a subclass of `models/base.py` (must accept a `dict` of params).
  2. Add it to `models/registry.py` under a unique key.
- At runtime, `train.py` looks up the YAML’s `model.name` in the registry and instantiates it.

---

## 🗂️ Data & Preprocessing

- Each preprocessing folder (e.g. `preprocessing_dicom_spm/`) contains:
  - A `data/` subfolder with `.nii` files + `<split>.csv` (paths + labels)
  - Scripts or instructions to reproduce the preprocessing

---

## 🛠️ Utilities

- **Dataset management**:  
  - `get_scan_list.py` → list files in archives  
  - `train_test_split.py` → generate train/val/test CSVs
- **Visualization**:  
  - `plot_training_log.py` → loss & metric curves  
  - `gradcam_visualize.py` → saliency maps  
  - `plot_histogram.py` → data distribution

---

## 🔍 Examples

**Local train with ResNet18 backbone:**

```bash
python train.py   --config configs/training/resnet18.yaml data.trainval_csv=./data/preprocessing_spm/data/trainval.csv   data.test_csv=./data/preprocessing_spm/data/test.csv 
```

**Submit to RCC, override epochs & LR:**

```bash
python train.py   --config configs/training/custom_3dcnn.yaml   --job   training.epochs=300   training.optimizer.lr=5e-4
```

---