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
batch_size = 32
epochs = 20
lr = 0.00001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
full_dataset = ADNIDataset(csv_file, transform=None)

print(f"Total samples in dataset: {len(full_dataset)}")



# Stratified split
# Numerical diagnosis mapping from CSV: 1.0 = MCI, 2.0 = AD, 3.0 = CN

label_to_name = {0: "MCI", 1: "AD", 2: "CN"}


name_to_label = {v: int(k) for k, v in label_to_name.items()}

df = pd.read_csv(csv_file)
labels = df['diagnosis']
labels_idx = labels.astype(int).tolist()  # for stratification


from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    stratify=labels_idx,
    random_state=42
)

train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = Simple3DCNN(num_classes=len(label_to_name)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
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

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Print classification report
print("\nClassification Report:")

target_names = [label_to_name[i] for i in sorted(set(y_true))]

print(classification_report(y_true, y_pred, target_names=target_names))


# Save model
torch.save(model.state_dict(), "baseline_3dcnn.pth")
print("\n✅ Model training complete. Saved as baseline_3dcnn.pth")
