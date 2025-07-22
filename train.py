"""
Command line tool for training models.
"""

import os
import yaml
import datetime
import argparse

from omegaconf import OmegaConf

# Path to default config file
DEFAULT_CONFIG_PATH = '/project/aereditato/cestari/adni-mri-classification/configs/training/default.yaml'

def train_and_evaluate(cfg_path, exp_dir=None):
    import torch
    import torch.nn as nn
    from torch.utils.data import Subset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    from models.registry import get_model
    from data.datasets import ADNIDataset
    from data.augmentation import random_crop

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Load and merge configs
    default_cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    with open(cfg_path, 'r') as f:
        user_cfg = yaml.safe_load(f)
    cfg = OmegaConf.merge(default_cfg, user_cfg)

    # Extract core settings
    folds       = int(cfg.cross_validation.folds)
    seed        = int(cfg.cross_validation.get('seed', 42))
    epochs      = int(cfg.training.epochs)
    batch_size  = int(cfg.training.batch_size)

    optim_cfg   = cfg.training.get('optimizer', {})
    optim_name  = optim_cfg.get('name', 'adam')
    lr          = float(optim_cfg.get('lr', 1e-5))
    weight_decay= float(optim_cfg.get('weight_decay', 0))

    scheduler_cfg     = cfg.training.get('scheduler', {})
    
    use_scheduler = scheduler_cfg != {}
    scheduler_name    = scheduler_cfg.get('name', 'CosineAnnealingLR')
    scheduler_t_max   = int(scheduler_cfg.get('t_max', epochs))
    scheduler_lr_min  = float(scheduler_cfg.get('lr_min', 1e-6))
    scheduler_lr_max  = float(scheduler_cfg.get('lr_max', 1e-3))

    model_cfg  = cfg.model
    model_name = str(model_cfg.name)

    # Data CSVs
    trainval_csv = cfg.data.trainval_csv
    test_csv     = cfg.data.test_csv

    # Build datasets: trainval augmented and no-augmentation, plus fixed test
    dataset_tv_aug = ADNIDataset(trainval_csv, transform=random_crop)
    dataset_tv     = ADNIDataset(trainval_csv, transform=None)
    dataset_test   = ADNIDataset(test_csv,     transform=None)

    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Prepare indices and labels for Stratified K-Fold
    labels = dataset_tv.labels()
    indices = list(range(len(dataset_tv)))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    # Containers for per-fold metrics
    reports = []
    cms = []
    
    # Loop over folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels), start=1):
        print(f"\n===== Fold {fold}/{folds} =====")

        # Subsets
        train_set = Subset(dataset_tv_aug, train_idx)
        val_set   = Subset(dataset_tv,     val_idx)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

        # Initialize model, loss, optimizer
        ModelClass = get_model(model_name)
        model = ModelClass(model_cfg).to(device)
        criterion = nn.CrossEntropyLoss()

        if optim_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim_name == 'nadam':
            optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Scheduler if requested
        scheduler = None
        if use_scheduler:
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(optimizer, min_lr=scheduler_lr_min)
            elif scheduler_name == 'CosineAnnealingLR':
                scheduler = CosineAnnealingLR(optimizer, scheduler_t_max, eta_min=scheduler_lr_min)
            elif scheduler_name == 'OneCycleLR':
                total_steps = epochs * len(train_loader)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=scheduler_lr_max,
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy='linear'
                )

        # Make directory for this fold
        fold_dir = os.path.join(exp_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # CSV logging
        csv_path = os.path.join(fold_dir, 'losses.csv')
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,val_loss\n')

        best_val_loss = float('inf')
        best_epoch = 0
        train_losses, val_losses = [], []

        # Training loop
        for epoch in range(1, epochs+1):
            model.train()
            running_train = 0.0
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
                if use_scheduler and scheduler_name == 'OneCycleLR':
                    scheduler.step()
                running_train += loss.item()
            avg_train = running_train / len(train_loader)
            train_losses.append(avg_train)

            # Validation
            model.eval()
            running_val = 0.0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    out = model(imgs)
                    running_val += criterion(out, lbls).item()
            avg_val = running_val / len(val_loader)
            val_losses.append(avg_val)

            # Step scheduler
            if use_scheduler:
                if scheduler_name == 'ReduceLROnPlateau':
                    scheduler.step(avg_val)
                elif scheduler_name == 'CosineAnnealingLR':
                    scheduler.step()

            # Log losses
            with open(csv_path, 'a') as f:
                f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f}\n")

            print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train:.4f} Val Loss: {avg_val:.4f}")

            # Save best
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pth'))

        # Plot and save loss curve
        plt.figure()
        plt.plot(range(1, epochs+1), train_losses, label='Train')
        plt.plot(range(1, epochs+1), val_losses,   label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold} Loss Curve')
        plt.legend()
        plt.savefig(os.path.join(fold_dir, 'loss_curve.png'))
        plt.close()

        # Test evaluation
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pth')))
        model.eval()
        all_preds, all_lbls = [], []
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_lbls.extend(lbls.cpu().numpy())

        report = pd.DataFrame(classification_report(all_lbls, all_preds, output_dict=True)).T
        cm = pd.DataFrame(confusion_matrix(all_lbls, all_preds))

        report.to_csv(os.path.join(fold_dir, 'classification_report.csv'))
        cm.to_csv(os.path.join(fold_dir, 'confusion_matrix.csv'))

        print(f"\nFold {fold} results (best epoch {best_epoch}, val_loss {best_val_loss:.4f}):")
        print(report)
        print(cm)
        
        # Accumulate metrics
        reports.append(report)
        cms.append(cm)
        
    # Compute and save average metrics across folds
    avg_report = pd.concat(reports).groupby(level=0).mean()
    avg_cm = sum(cms) / folds
    avg_report.to_csv(os.path.join(exp_dir, 'average_classification_report.csv'))
    avg_cm.to_csv(os.path.join(exp_dir, 'average_confusion_matrix.csv'))

    print("\nAverage classification report across folds:")
    print(avg_report)
    print("\nAverage confusion matrix across folds:")
    print(avg_cm)

    print(f"\nCross-validation complete. Artifacts under {exp_dir}")


def submit_slurm(config_path, dir_path):

    job_name = os.path.basename(os.path.normpath(dir_path))
    slurm_path = os.path.join(dir_path, f"{job_name}.slurm")
    
    log_out = os.path.join(dir_path, f"{job_name}_%j.out")
    log_err = os.path.join(dir_path, f"{job_name}_%j.err")
    
    # Absolute paths to this script & the merged config
    abs_train = os.path.abspath(__file__)
    abs_cfg   = os.path.abspath(config_path)

    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output={log_out}
#SBATCH --error={log_err}
#SBATCH --account=pi-aereditato

# Load your conda environment
source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate adni_rcc   

# Optional: Print environment info
echo "Using Python from: $(which python)"
nvidia-smi
echo "Starting job at $(date)"

# Go to project directory
cd /project/aereditato/cestari/adni-mri-classification

# Run training using the same experiment directory
python {abs_train} --config {abs_cfg} --dir {exp_dir}

echo "Finished at $(date)"""  

    with open(slurm_path, 'w') as f:
        f.write(content)
    print(f"Slurm script written to {slurm_path}, submitting job...")
    os.system(f"sbatch {slurm_path}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train or submit as Slurm job')
    
    parser.add_argument(
        "-c", "--config",
        action="append",
        required=True,
        help="One or more YAML config files (you can repeat -c for each)"
    )

    parser.add_argument(
        '--job',
        action='store_true',
        help='Submit via Slurm'
    )

    parser.add_argument(
        '--dir',
        default=None,
        help='Experiment directory'
    )

    # Each override is explicit, not positional, and can be repeated
    parser.add_argument(
        "-o", "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override in dot-list form, e.g. -o training.batch_size=16"
    )

    args = parser.parse_args()

    # Load default config
    cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    
    # Load and merge all YAML configs
    try:
        for path in args.config:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
            
    except Exception as e:
        print(f"Error: {e}")
        exit(-1)

    # Apply any CLI overrides (e.g. training.batch_size=16)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    
    # Check if model name is specified
    assert 'name' in cfg.model, "Error: model name is not specified"
    
    # Create directory
    exp_dir = args.dir
    
    if exp_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{cfg.model.name}_{timestamp}"
        exp_dir = os.path.join('./experiments', exp_name)
        
    os.makedirs(exp_dir, exist_ok=True)
    
    # Dump all configs to file
    config_path = os.path.join(exp_dir,"config.yaml")
    OmegaConf.save(cfg, config_path)
    
    # Dispatch
    if args.job: 
        submit_slurm(config_path=config_path, dir_path=exp_dir)
    else:
        train_and_evaluate(cfg_path=config_path, exp_dir=exp_dir)
