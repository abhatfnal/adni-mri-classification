import os
import yaml
import datetime
import argparse


def train_and_evaluate(cfg_path, exp_dir=None):
    import pandas as pd
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from models.registry import get_model
    from data.datasets import ADNIDataset
    from torch.utils.data import Subset, DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    from data.augmentation import augmentation_transform

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Empty cache
    torch.cuda.empty_cache()
    
    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Model config
    model_cfg = dict(cfg.get('model', {}))
    model_name = str(model_cfg.get('name', 'simple_3dcnn'))
    
    # Train config
    train_cfg = dict(cfg.get('training', {}))
    
    epochs = int(train_cfg.get('epochs', 50))
    batch_size = int(train_cfg.get('batch_size', 8))
    
    # Optimizer config
    optim_cfg = dict(train_cfg.get('optimizer', {}))
    
    optim_name = str(optim_cfg.get('name', 'adam'))
    lr = float(optim_cfg.get('lr', 1e-5))
    weight_decay = float(optim_cfg.get('weight_decay', 0))
    
    # LR scheduler config 
    use_scheduler = 'scheduler' in train_cfg
    
    scheduler_cfg = dict(train_cfg.get('scheduler',{}))
    scheduler_name = str(scheduler_cfg.get('name','CosineAnnealingLR'))
    scheduler_t_max = int(scheduler_cfg.get('t_max',200))
    scheduler_lr_min = float(scheduler_cfg.get('lr_min',1e-6))
    scheduler_lr_max = float(scheduler_cfg.get('lr_max',1e-3))
    
    # Data config
    data_cfg = dict(cfg.get('data',{}))
    
    augmentation = bool(data_cfg.get('augmentation', False))

    # Setup experiment directory once
    if exp_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{cfg['model']['name']}_{timestamp}"
        exp_dir = os.path.join('./experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save a copy of the config for reproducibility
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    print(f"Batch size: {batch_size}, Learning rate: {lr}, Epochs: {epochs}")
    
    # Prepare datasets
    full_dataset = ADNIDataset()                                        # for validation
    augmented_dataset = ADNIDataset(transform=augmentation_transform)   # for training
    
    print(f"Total samples in dataset: {len(full_dataset)}")
    
    labels = full_dataset.labels()
    
    # First: train+val vs test split
    trainval_idx, test_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Second: train vs val split (from trainval)
    train_labels = [labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.2,
        stratify=train_labels,
        random_state=42
    )
    
    if augmentation:
        train_dataset = Subset(augmented_dataset, train_idx)
    else:
        train_dataset = Subset(full_dataset, train_idx)
        
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    ModelClass = get_model(model_name)
    model = ModelClass(model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    if(use_scheduler):
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, min_lr=scheduler_lr_min)
        elif scheduler_name == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer,scheduler_t_max, eta_min=scheduler_lr_min)
        elif scheduler_name == 'OneCycleLR':
            steps = epochs * len(train_loader)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=scheduler_lr_max, total_steps=steps,
                pct_start=0.1, anneal_strategy='linear'
            )

    # Log model, optimizer and dataset option to ensure it's what we wanted to train
    print(model)
    print(optimizer)
    
    if(use_scheduler):
        print(scheduler)
        
    print(f"Dataset augmentation: {augmentation}")
    
    # Real-time CSV logging setup
    csv_path = os.path.join(exp_dir, 'losses.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    # Training loop with validation tracking
    best_val_loss = float('inf')
    best_epoch = -1
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_train = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            
            #scheduler step
            if(use_scheduler):
                if scheduler_name == 'OneCycleLR':
                    scheduler.step()
                
                    
            running_train += loss.item()
            
        avg_train = running_train / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                running_val += criterion(out, lbls).item()
        avg_val = running_val / len(val_loader)
        val_losses.append(avg_val)
        
        #scheduler step
        if(use_scheduler):
            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(avg_val)
            elif scheduler_name == 'CosineAnnealingLR':
                scheduler.step()

        # Append to CSV
        with open(csv_path, 'a') as f:
            f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f}\n")

        print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train:.4f} Val Loss: {avg_val:.4f}")

        # Save best model
        if avg_val < best_val_loss:
            best_val_loss, best_epoch = avg_val, epoch
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))

    # Plot losses
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train')
    plt.plot(range(1, epochs+1), val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'loss_curve.png'))
    plt.close()

    print(f"\nFinished training; best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
    print(f"Artifacts in {exp_dir}")

    # Testing
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pth')))
    model.eval()
    
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.cpu().numpy())

    print("\nClassification Report:")
    print(pd.DataFrame(classification_report(all_lbls, all_preds, output_dict=True)).T)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(confusion_matrix(all_lbls, all_preds)))


def submit_slurm(config_path):
    # Load config for naming
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Experiment folder naming
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{cfg['model']['name']}_{ts}"
    exp_dir = os.path.join('experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    job_name = exp_name
    slurm_path = os.path.join(exp_dir, f"{job_name}.slurm")
    abs_train = os.path.abspath(__file__)
    abs_cfg = os.path.abspath(config_path)

    log_out = os.path.join(exp_dir, f"{job_name}_%j.out")
    log_err = os.path.join(exp_dir, f"{job_name}_%j.err")

    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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
python {abs_train} {abs_cfg} --exp_dir {exp_dir}

echo "Finished at $(date)"""  

    with open(slurm_path, 'w') as f:
        f.write(content)
    print(f"Slurm script written to {slurm_path}, submitting job...")
    os.system(f"sbatch {slurm_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or submit as Slurm job')
    parser.add_argument('config_file', help='Path to config YAML')
    parser.add_argument('--job', action='store_true', help='Submit via Slurm')
    parser.add_argument('--exp_dir', default=None,
                        help='Experiment directory')
    args = parser.parse_args()

    if args.job:
        submit_slurm(args.config_file)
    else:
        train_and_evaluate(args.config_file, args.exp_dir)
