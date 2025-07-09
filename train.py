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
    from data.datasets import TrainValDataset, TestDataset
    from torch.utils.data import Subset, DataLoader

    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Check required params
    assert 'name' in cfg['model'], "Model name not specified in config"
    assert 'epochs' in cfg['training'], "Number of epochs not specified in config"
    assert 'batch_size' in cfg['training'], "Batch size not specified in config"

    # Setup experiment directory once
    if exp_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{cfg['model']['name']}_{timestamp}"
        exp_dir = os.path.join('./experiments', exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save a copy of the config for reproducibility
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    # Device, hyperparams
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = int(cfg['training']['batch_size'])
    lr = float(cfg['training']['lr'])
    epochs = int(cfg['training']['epochs'])

    # Prepare data
    dataset = TrainValDataset()
    labels = dataset.labels()
    train_idx, val_idx = train_test_split(
        range(len(dataset)), test_size=1/9,
        random_state=42, stratify=labels
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    ModelClass = get_model(cfg['model']['name'])
    model = ModelClass(cfg['model']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Log model to ensure it's what we wanted to train
    print(model)
    
    # Real-time CSV logging setup
    csv_path = os.path.join(exp_dir, 'losses.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

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
    test_ds = TestDataset()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
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
