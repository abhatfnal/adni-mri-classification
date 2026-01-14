"""
Command line tool for training models.
"""

import os
import yaml
import datetime
import argparse

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from data.datasets import ADNIDataset
from data.augmentation import build_augmentation, cutmix_3d, mixup_3d, cutout_3d

from inspect import getfile
from models.registry import get_model
from omegaconf import OmegaConf
    

# Path to default config file
DEFAULT_CONFIG_PATH = './configs/training/default.yaml'

def extract_settings(cfg):
    """
    Extracts core settings from config dictionary, converts to appropriate data
    type and replaces missing values with defaults.
    """
    
    settings = {}
    
    # Training config
    settings['epochs'] = int(cfg.training.epochs)
    settings['batch_size'] = int(cfg.training.batch_size)
    settings['loss_function'] = cfg.training.get('loss', 'cross_entropy')
  
    # Data config
    settings['trainval_csv'] = cfg.data.trainval_csv
    settings['test_csv'] = cfg.data.test_csv
    settings['oversample_enabled'] = bool(cfg.data.get('oversample', False))
    settings['classes'] = cfg.data.get('classes', [1, 2, 3])
    
    # Cross validation config
    settings['cv_seed'] = int(cfg.cross_validation.get('seed', 42))
    settings['cv_method'] = str(cfg.cross_validation.get('method', 'kfold'))
    settings['cv_folds'] = int(cfg.cross_validation.get('folds', 5))
    settings['cv_val_frac'] = float(cfg.cross_validation.get('val_fraction', 0.2))
    
    # Optimizer config
    optim_config  = cfg.training.get('optimizer', {})
    
    settings['optim_name'] = optim_config.get('name', 'adam')
    settings['lr'] = float(optim_config.get('lr', 1e-5))
    settings['weight_decay'] = float(optim_config.get('weight_decay', 0))
    settings['patience'] = int(optim_config.get('patience',-1))

    # Scheduler config
    scheduler_config     = cfg.training.get('scheduler', {})
    settings['scheduler_enabled'] = (scheduler_config != {})
    
    if settings['scheduler_enabled']:
        settings['scheduler_name'] = scheduler_config.get('name', 'CosineAnnealingLR')
        settings['scheduler_t_max'] = int(scheduler_config.get('t_max', settings['epochs']))
        settings['scheduler_lr_min'] = float(scheduler_config.get('lr_min', 1e-6))
        settings['scheduler_lr_max']  = float(scheduler_config.get('lr_max', 1e-3))

    # Model config
    settings['model_config'] = cfg.model
    settings['model_name'] = str(settings['model_config'].name)
    settings['num_classes'] = int(cfg.model.get('num_classes', 3))
    
    # Augmentation config
    settings['aug_config'] = cfg.data.get('augmentation', {}) 
    settings['aug_enabled'] = (settings['aug_config'] != {})

    settings['aug_mixup_config'] = cfg.data.get('mixup', {})
    settings['aug_mixup_enabled'] = (settings['aug_mixup_config'] != {})
    
    settings['aug_cutout_config'] = cfg.data.get('cutout', {})
    settings['aug_cutout_enabled'] = (settings['aug_cutout_config'] != {})
    
    settings['aug_cutmix_config'] = cfg.data.get('cutmix', {})
    settings['aug_cutmix_enabled'] = (settings['aug_cutmix_config'] != {})
    
    return settings
    
    
def log_settings(sett):
    """
    Logs configurations and hyperparameters.
    """
    
    print("------------ Configuration ------------")
    
    for key, value in sett.items():
        print(f"{key}: {value}")
    
    print("---------------------------------------")
    
    
def get_splits(labels, indices, groups, seed, method='kfold', folds=5, val_frac=0.2):
    """
    Returns splits for cross validation using StratifiedGroupKFold. 
    """
    
    if method == 'holdout':
        n_splits = int(1/val_frac)
    else:
        n_splits = folds
     
    # Create splitter   
    splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=seed)
        
    # Get splits
    splits = [(fold, (train_idx, val_idx)) for fold, (train_idx, val_idx) in enumerate(
        splitter.split(indices, labels, groups), start=1)
    ]
        
    # If holdout,take only the first split
    if method == 'holdout':  
        splits = [splits[0]]
    
    return splits

    
def get_loss_criterion(name, device, train_labels=None):
    """
    Returns loss criterion. If train labels are passed, weights for the cross entropy loss 
    are computed.
    """
    
    if name == 'cross_entropy':
        
        # If train labels are passed, compute weights for cross entroy loss
        if train_labels is not None:

            class_counts = torch.bincount(torch.tensor(train_labels))
            class_weights = 1.0 / class_counts.float()    
            class_weights = class_weights / class_weights.sum()  # normalize to sum to 1
            class_weights = class_weights.to(device)

            return nn.CrossEntropyLoss(weight=class_weights)
            
    elif name == 'bce':
        return nn.BCEWithLogitsLoss()
    
    else:
        raise ValueError(f"Unknown loss function: {name}")
    
    
def get_optimizer(name, params, lr, weight_decay):
    """
    Returns specified optimizer with given model params, lr and weight decay.
    """
    
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
    elif name == 'nadam':
        return torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay)
        
    else:
        raise ValueError(f"Unknown optimizer name {name}")
    
def get_scheduler(name, optimizer, min_lr, t_max=None):
    """
    Returns specified scheduler.
    """
    
    if name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, min_lr=min_lr)

    elif name == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, t_max, eta_min=min_lr)
        
    else:
        raise ValueError(f"Unknown scheduler name: {name}")
    
def plot_losses(fold, fold_dir, train_losses, val_losses):
    """
    Creates train and validation losses plots and saves them to specified dir.
    """
    
    plt.figure()
    plt.plot([i for i in range(1, len(train_losses)+1)], train_losses, label='Train')
    plt.plot([i for i in range(1, len(val_losses)+1)], val_losses,   label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(fold_dir, 'loss_curve.png'))
    plt.close()

    
def train_and_evaluate(cfg_path, exp_dir=None):
        
    # Set device and empty cache
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Load and merge configs
    default_cfg = OmegaConf.load(DEFAULT_CONFIG_PATH)
    
    with open(cfg_path, 'r') as f:
        user_cfg = yaml.safe_load(f)
        
    cfg = OmegaConf.merge(default_cfg, user_cfg)

    # Extract settings
    sett = extract_settings(cfg)
    
    # Log settings
    log_settings(sett)
    
    # Build augmentation transform 
    aug_transform = None
    if sett['aug_enabled']:
        aug_transform = build_augmentation(sett['aug_config'])
        
    # Load datasets: trainval augmented and no-augmentation, plus fixed test
    dataset_tv_aug = ADNIDataset(sett['trainval_csv'], transform=aug_transform, classes=sett['classes']) 
    dataset_tv     = ADNIDataset(sett['trainval_csv'], transform=None, classes=sett['classes'])
    dataset_test   = ADNIDataset(sett['test_csv'],     transform=None, classes=sett['classes'])

    # Create data loader for test set
    test_loader = DataLoader(dataset_test, 
                             batch_size=sett['batch_size'], 
                             shuffle=False, 
                             num_workers=1, 
                             pin_memory=True)

    # Trainval labels, groups and indices for splits
    labels_tv = dataset_tv.labels()
    indices_tv = list(range(len(dataset_tv)))
    groups_tv = dataset_tv.groups()
    
    # Create splits based on cross validation method
    splits = get_splits(labels_tv, 
                        indices_tv, 
                        groups_tv, 
                        sett['cv_seed'], 
                        method=sett['cv_method'], 
                        folds=sett['cv_folds'], 
                        val_frac=sett['cv_val_frac'])
   

    # Containers for metrics
    reports = []    # Classification reports
    cms = []        # Confusion matrices
    
    # Loop over folds
    for fold, (train_idx, val_idx) in splits:
        
        print(f"\n===== Fold {fold}/{len(splits)} =====")

        # Train and validation subsets according to the splits
        train_set = Subset(dataset_tv_aug, train_idx)
        val_set   = Subset(dataset_tv,     val_idx)
        
        # Train labels
        train_labels = np.array(labels_tv)[train_idx]

        # If oversample is enabled, created weighted random sampler 
        # to sample uniformly from all classes
        if sett['oversample_enabled']:
            
            class_weights = 1/np.bincount(train_labels)
            sample_weights = class_weights[train_labels]
            
            sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), 
                                            len(sample_weights), 
                                            replacement=True)
        else:
            sampler = None
            
        # Train loader
        train_loader = DataLoader(
                            train_set,
                            batch_size=sett['batch_size'], 
                            shuffle= (sampler == None),     # Shuffle if no sampler is used
                            sampler=sampler, 
                            num_workers=4, 
                            pin_memory=True)
        
        # Validation loader
        val_loader = DataLoader(
                            val_set,  
                            batch_size=sett['batch_size'], 
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=True)
        
        # Initialize model, loss, optimizer
        ModelClass = get_model(sett['model_name'])
        model = ModelClass(sett['model_config']).to(device)
        
        # Loss criterion
        criterion = get_loss_criterion(sett['loss_function'], device, train_labels=train_labels)

        # Optimizer
        optimizer = get_optimizer(sett['optim_name'], model.parameters(), sett['lr'], sett['weight_decay'])

        # Scheduler 
        scheduler = None
        if sett['scheduler_enabled']:
            scheduler = get_scheduler(sett['scheduler_name'], optimizer, sett['scheduler_lr_min'], sett['scheduler_t_max'])
            
        # Create directory for this fold
        fold_dir = os.path.join(exp_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # CSV logging
        csv_path = os.path.join(fold_dir, 'losses.csv')
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,val_loss\n')

        best_val_loss = float('inf')
        best_epoch = 0
        train_losses, val_losses = [], []

        # Initialize patience counter
        patience_counter = 0
        
        # Training loop
        for epoch in range(1, sett['epochs']+1):
            
            # Set model to training mode
            model.train()
            
            # Keep track of training loss
            running_train = 0.0
            
            # Loop over batches
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                    
                # Use mixup if enabled
                if sett['aug_mixup_enabled']:
                    if torch.rand(()) < sett['aug_mixup_config'].get('p',0.5):
                        imgs, lbls, perm, lam = mixup_3d(imgs, lbls, 
                                                         num_classes=sett['num_classes'],
                                                         alpha=sett['aug_mixup_config'].get('alpha',0.3))
                        
                # Use cutout if enabled
                if sett['aug_cutout_enabled']:
                    if torch.rand(()) < sett['aug_cutout_config'].get('p',0.5):
                        imgs = cutout_3d(imgs, 2, 0.05, fill="zero", return_mask=False)
                        
                # Use cutmix if enabled:
                if sett['aug_cutmix_enabled']:
                    if torch.rand(()) < sett['aug_cutmix_config'].get('p',0.5):
                        imgs, lbls, perm, lam = cutmix_3d(imgs, lbls, 
                                                          num_classes=sett['num_classes'], 
                                                          alpha=sett['aug_mixup_config'].get('alpha',1.0), 
                                                          same_class=True)

                # Reset gradient
                optimizer.zero_grad()
                
                # Forward pass
                out = model(imgs).squeeze(1)

                # Loss 
                loss = criterion(out, lbls)
                
                # Backward pass
                loss.backward()
                
                # Step
                optimizer.step()
                
                # Update training loss
                running_train += loss.item()
                
            avg_train = running_train / len(train_loader)
            train_losses.append(avg_train)

            # Validation
            model.eval()
            running_val = 0.0
            with torch.no_grad():
                for imgs, lbls in val_loader:
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    out = model(imgs).squeeze(1)
                    
                    running_val += criterion(out, lbls).item()
            avg_val = running_val / len(val_loader)
            val_losses.append(avg_val)

            # Scheduler step
            if scheduler is not None:
                
                if sett['scheduler_name']== 'ReduceLROnPlateau':
                    scheduler.step(avg_val)
                elif sett['scheduler_name'] == 'CosineAnnealingLR':
                    scheduler.step()

            # Log losses
            with open(csv_path, 'a') as f:
                f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f}\n")

            print(f"Epoch [{epoch}/{sett['epochs']}] Train Loss: {avg_train:.4f} Val Loss: {avg_val:.4f}")

            # Save best
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch = epoch
                torch.save(model, os.path.join(fold_dir, 'best_model.pth'))
                
                # Set patience counter to 0
                patience_counter = 0
            else:
                patience_counter += 1
                
            # If patience is enabled and patience counter exceeds patience, stop training
            if sett['patience'] > 0 and patience_counter > sett['patience']:
                break

        # Plot and save loss curve
        plot_losses(fold, fold_dir, train_losses, val_losses)
        
        # Evaluation on the test set:
        
        # Load the full model back in one step.
        model = torch.load(
            os.path.join(fold_dir, 'best_model.pth'),
            map_location=device,
            weights_only=False
        )
        
        # Toggle evaluation mode
        model.eval()

        all_preds, all_lbls = [], []
        
        # Disable gradient calculation
        with torch.no_grad():
            
            # Loop over batches
            for imgs, lbls in test_loader:
                
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                # Forward 
                out = model(imgs)
                
                # Compute predictions
                preds = out.argmax(dim=1)
                
                # Append to predictions list
                all_preds.extend(preds.cpu().numpy())
                all_lbls.extend(lbls.cpu().numpy())

        # Compute classification report
        report = pd.DataFrame(classification_report(all_lbls, all_preds, output_dict=True)).T
        
        # and confusion matrix.
        cm = pd.DataFrame(confusion_matrix(all_lbls, all_preds))

        # Save them to .csv
        report.to_csv(os.path.join(fold_dir, 'classification_report.csv'))
        cm.to_csv(os.path.join(fold_dir, 'confusion_matrix.csv'))

        print(f"\nFold {fold} results (best epoch {best_epoch}, val_loss {best_val_loss:.4f}):")
        print(report)
        print(cm)
        
        # Accumulate metrics
        reports.append(report)
        cms.append(cm)
        
    if sett['cv_method'] == 'kfold':
        
        # Compute average report and confusion matrices
        avg_report = pd.concat(reports).groupby(level=0).mean()
        avg_cm     = sum(cms) / int(sett['cv_folds'])
        
        # Save them to file
        avg_report.to_csv(os.path.join(exp_dir, 'average_classification_report.csv'))
        avg_cm.to_csv(os.path.join(exp_dir, 'average_confusion_matrix.csv'))
        
        # Log
        print("\nAverage classification report across folds:")
        print(avg_report)
        print("\nAverage confusion matrix across folds:")
        print(avg_cm)
    else:
        print("Hold-out validation complete. See fold_1 results in", exp_dir)
        
    print(f"\nEvaluation complete. Artifacts under {exp_dir}")


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
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3:00:00
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
    
    # Dump original model file
    model_path = getfile(get_model(cfg.model.name))
    
    with open(model_path, "rb") as src, open(exp_dir + '/model.py', "wb") as dst:
        while chunk := src.read(8192):  # 8 KB buffer
            dst.write(chunk)
            
    # Dump training file (this file)
    with open(__file__, "rb") as src, open(exp_dir + '/train.py', "wb") as dst:
        while chunk := src.read(8192):  # 8 KB buffer
            dst.write(chunk)
    
    # Dispatch
    if args.job: 
        submit_slurm(config_path=config_path, dir_path=exp_dir)
    else:
        train_and_evaluate(cfg_path=config_path, exp_dir=exp_dir)
