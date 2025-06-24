import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from adni_dataset import ADNIDataset
from model_3dcnn import Simple3DCNN
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Configuration
csv_file = '/home/abhat/ADNI/adni_preprocessed_metadata.csv'
data_dir = '/scratch/7DayLifetime/abhat/ADNI/ADNI1_Complete_3Yr_3T/ADNI/'
batch_size = 64
epochs = 200
lr = 0.00001
patience = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
full_dataset = ADNIDataset(csv_file, transform=None)
print(f"Total samples in dataset: {len(full_dataset)}")

# Stratified split
label_to_name = {0: "MCI", 1: "AD", 2: "CN"}
name_to_label = {v: int(k) for k, v in label_to_name.items()}

df = pd.read_csv(csv_file)
labels = df['diagnosis']
labels_idx = labels.astype(int).tolist()

from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=labels_idx,
    random_state=42
)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Modified Model for Grad-CAM with Dropout
class GradCAM3DCNN(Simple3DCNN):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.feature_maps = None

        def hook_fn(module, input, output):
            self.feature_maps = output

        self.conv3.register_forward_hook(hook_fn)

# Update Simple3DCNN separately to include Dropout after conv blocks if not done
model = GradCAM3DCNN(num_classes=len(label_to_name)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop with early stopping and best-loss checkpoint
best_val_loss = float('inf')
epochs_no_improve = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    epoch_train_loss = train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    epoch_val_loss = val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "baseline_3dcnn.pth")
        print(f"✅ Best model saved at epoch {epoch+1} with val loss {best_val_loss:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
        break

# Final Evaluation
model.load_state_dict(torch.load("baseline_3dcnn.pth"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:")
target_names = [label_to_name[i] for i in sorted(set(y_true))]
print(classification_report(y_true, y_pred, target_names=target_names))