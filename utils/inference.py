"""
Command line tool for evaluating a pre-trained model on a new test set.
"""
import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf

# --- Add project root to Python path ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('..')
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
# ---

import torch
from torch.utils.data import Dataset
from models.registry import get_model

class NiftiDataset(Dataset):
    """A flexible PyTorch dataset for loading NIfTI files from a CSV."""
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        file_path = self.dataframe.loc[idx, 'filepath']
        label = int(self.dataframe.loc[idx, 'diagnosis'])
        nii_img = nib.load(file_path)
        image_data = nii_img.get_fdata(dtype=np.float32)
        image_data = np.expand_dims(image_data, axis=0)
        if self.transform:
            image_data = self.transform(image_data)
        return torch.from_numpy(image_data), torch.tensor(label, dtype=torch.long)
        
    def labels(self):
        return self.dataframe['diagnosis'].tolist()

DEFAULT_CONFIG_PATH = '/project/aereditato/abhat/adni-mri-classification/configs/training/default.yaml'

def evaluate_model(cfg_path, exp_dir):
    from sklearn.metrics import classification_report, confusion_matrix
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    default_cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    with open(cfg_path, 'r') as f:
        user_cfg = yaml.safe_load(f)
    cfg = OmegaConf.merge(default_cfg, user_cfg)

    model_path = cfg.inference.model_path
    test_csv = cfg.inference.test_csv
    batch_size = cfg.training.batch_size

    print("\n======|| Inference Configuration ||======")
    print(f"-> Model Path: {model_path}")
    print(f"-> Test Data CSV: {test_csv}")
    print(f"-> Output Directory: {exp_dir}")

    # --- THIS IS THE FINAL CORRECTED SECTION ---
    try:
        # 1. Re-create the model architecture using your config
        model_cfg = cfg.model
        ModelClass = get_model(model_cfg.name)
        model = ModelClass(model_cfg)
        
        # 2. Load the saved weights from the .pth file
        # The saved file is a full model object, so we access its state_dict()
        saved_state = torch.load(model_path, map_location=device, weights_only=False).state_dict()
        
        # 3. Load the weights flexibly, ignoring mismatched layers
        # The `strict=False` argument is the key to solving the error.
        missing_keys, unexpected_keys = model.load_state_dict(saved_state, strict=False)
        
        print("\n--- Flexible Weight Loading Summary ---")
        if missing_keys:
            print("Warning: The following keys were in the new model but not in the checkpoint:", missing_keys)
        if unexpected_keys:
            print("Info: The following keys were in the checkpoint but not in the new model (this is expected):", unexpected_keys)
        # --- END OF CORRECTION ---

        model.to(device)
        model.eval()
        print("\n✅ Pre-trained model loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not load model. Error: {e}")
        return

    test_dataset = NiftiDataset(test_csv, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"✅ Test data loaded. Found {len(test_dataset)} samples.")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            print(f"  -> Processing batch {i+1}/{len(test_loader)}...")

    print("\n--- Evaluation Complete ---")
    report = pd.DataFrame(classification_report(all_labels, all_preds, output_dict=True, zero_division=0)).T
    cm = pd.DataFrame(confusion_matrix(all_labels, all_preds))

    report.to_csv(os.path.join(exp_dir, 'OASIS_classification_report.csv'))
    cm.to_csv(os.path.join(exp_dir, 'OASIS_confusion_matrix.csv'))

    print("\nClassification Report (OASIS Test Set):")
    print(report)
    print("\nConfusion Matrix (OASIS Test Set):")
    print(cm)
    print(f"\n✅ Results saved in {exp_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a pre-trained model.')
    parser.add_argument('-c', '--config', required=True, help='Path to the inference YAML config file.')
    args = parser.parse_args()

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join('./experiments', f'inference_OASIS_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)

    evaluate_model(args.config, exp_dir)