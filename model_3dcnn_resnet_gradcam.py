#!/usr/bin/env python3
import os
import csv
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torchvision

from adni_dataset import ADNIDataset

# --------------------------------------------
# 1) Define your GradCAM‐compatible 3D ResNet
# --------------------------------------------
def _hook_fn(module, input, output):
    model.feature_maps = output

class GradCAMResNet3D(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # load a small 3D ResNet backbone
        self.backbone = torchvision.models.video.r3d_18(pretrained=False)
        # adapt first conv to 1 channel instead of 3
        self.backbone.stem[0] = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3,7,7),
            stride=(1,2,2),
            padding=(1,3,3),
            bias=False
        )
        # remove its final fc
        self.backbone.fc = nn.Identity()

        # register a hook on the last conv-block for GradCAM
        self.feature_maps = None
        self.backbone.layer4.register_forward_hook(_hook_fn)

        # a little dropout + your final classifier
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B,1,D,H,W]
        features = self.backbone(x)            # [B,512]
        out = self.dropout(features)          # [B,512]
        out = self.classifier(out)            # [B,num_classes]
        return out

# --------------------------------------------
# 2) Training script with smoothing & reg
# --------------------------------------------
def train_model():
    # — config —
    csv_file   = "adni_preprocessed_npy_metadata.csv"
    batch_size = 8
    epochs     = 500
    lr         = 1e-5
    patience   = 30
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # — build dataset & split —
    df = pd.read_csv(csv_file)
    labels = df["diagnosis"].astype(int).tolist()

    full_ds = ADNIDataset(csv_file, augment=True)
    train_idx, val_idx = train_test_split(
        list(range(len(full_ds))),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds   = torch.utils.data.Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=4)

    # — model, loss, optimizer, scheduler —
    global model
    model = GradCAMResNet3D(num_classes=3).to(device)

    # class‐balanced weights
    counts = df["diagnosis"].value_counts().sort_index().values
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = (weights / weights.sum() * len(weights)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    # — setup logging & smoothing buffer —
    log_file = "training_log_resnet_gradcam_2.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_loss","smooth_val_loss"])

    best_val = float("inf")
    val_queue = deque(maxlen=5)
    epochs_no_improve = 0

    # — training loop —
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out,y)
            loss.backward()
            # clip to avoid spikes
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # — validation —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X,y in val_loader:
                X,y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()
        val_loss /= len(val_loader)

        # — smooth & log —
        val_queue.append(val_loss)
        smooth_val = sum(val_queue) / len(val_queue)

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{smooth_val:.4f}"])

        print(f"[{epoch:03d}/{epochs}]  Train: {train_loss:.4f}  Val: {val_loss:.4f}  Smooth: {smooth_val:.4f}")

        # — checkpoint on smoothed loss —
        if smooth_val < best_val:
            best_val = smooth_val
            torch.save(model.state_dict(), "best_resnet_gradcam_2.pth")
            epochs_no_improve = 0
            print("    ✅ saved best model")
        else:
            epochs_no_improve += 1

        # — step scheduler —
        scheduler.step(smooth_val)
        
        # print the current LR(s):
        lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"    LR now: {[f'{lr:.2e}' for lr in lrs]}")

        if epochs_no_improve >= patience:
            print(f"    ⏹️ early stopping @ epoch {epoch}")
            break

    # — final evaluation report —
    model.load_state_dict(torch.load("best_resnet_gradcam_2.pth", map_location=device))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X,y in val_loader:
            X,y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(dim=1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    print("\n" + classification_report(
        y_true, y_pred,
        target_names=["MCI","AD","CN"]
    ))

if __name__ == "__main__":
    train_model()
