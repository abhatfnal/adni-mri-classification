#!/usr/bin/env python
import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from model_3dcnn import Simple3DCNN
from adni_dataset import ADNIDataset

# === CONFIGURATION ===
CSV_FILE     = "adni_preprocessed_npy_metadata.csv"
N_SPLITS     = 5
BATCH_SIZE   = 32
MAX_EPOCHS   = 300
LR           = 5e-4
PATIENCE     = 50
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 2.0
NUM_WORKERS  = 8
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES  = {0: "MCI", 1: "AD", 2: "CN"}

def train_and_evaluate():
    # load full dataset and stratify by label
    full_ds = ADNIDataset(CSV_FILE)
    labels  = full_ds.data["label"].values  # already 0,1,2

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        print(f"\n=== Fold {fold}/{N_SPLITS} ===")
        train_counts = np.bincount(labels[train_idx], minlength=3)
        val_counts   = np.bincount(labels[val_idx],   minlength=3)
        print(f"  train counts: {train_counts},  val counts: {val_counts}")

        # make loaders
        train_ds = Subset(full_ds, train_idx)
        val_ds   = Subset(full_ds, val_idx)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )

        # build model / loss / optimizer
        model = Simple3DCNN(num_classes=3).to(DEVICE)

        # class‐balanced weights, explicitly float32
        weights = 1.0 / torch.tensor(train_counts, dtype=torch.float32, device=DEVICE)
        weights = weights / weights.sum()  # normalize if you like
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )

        best_val_loss     = float("inf")
        epochs_no_improve = 0

        # prepare per‐fold log
        log_path = f"training_log_fold{fold}.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, MAX_EPOCHS + 1):
            # --- TRAIN ---
            model.train()
            train_loss = 0.0
            for X, y in train_loader:
                # ensure correct dtypes
                X = X.to(DEVICE, dtype=torch.float32)
                y = y.to(DEVICE, dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(X)
                loss    = criterion(outputs, y)
                loss.backward()

                # debug: print gradient norm
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                print(f"[Fold {fold}][Epoch {epoch}] grad-norm: {total_norm:.3f}")

                # clip + step
                # nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_loader)

            # --- VALIDATION ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.to(DEVICE, dtype=torch.float32)
                    y = y.to(DEVICE, dtype=torch.long)
                    out = model(X)
                    val_loss += criterion(out, y).item()
            val_loss /= len(val_loader)

            # log to CSV
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss])

            print(
                f"[Fold {fold}][Epoch {epoch}/{MAX_EPOCHS}]  "
                f"Train: {train_loss:.4f}  Val: {val_loss:.4f}"
            )

            # checkpoint + early‐stop
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(),
                    f"best_model_fold{fold}.pth"
                )
                print(f"  ✅ saved best model (val={val_loss:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # if epochs_no_improve >= PATIENCE:
            #     print(f"  ⏹️ early stopping at epoch {epoch}")
            #     break

        # --- FINAL EVAL FOR FOLD ---
        model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE, dtype=torch.float32)
                preds = model(X).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        print(f"\n--- Fold {fold} classification report ---")
        print(
            classification_report(
                all_labels, all_preds,
                target_names=[CLASS_NAMES[i] for i in range(3)],
                zero_division=0
            )
        )

if __name__ == "__main__":
    print(f"Starting 5-fold run on device: {DEVICE}")
    train_and_evaluate()
