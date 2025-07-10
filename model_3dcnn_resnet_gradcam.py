# model_3dcnn_resnet_gradcam.py

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from adni_dataset import ADNIDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === 3D Residual Block ===
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1,1,1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_channels)

        # if we change channels or spatial size, use a 1×1×1 projection
        if stride != (1,1,1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# === ResNet-style 3D CNN ===
class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=7, stride=(1,2,2), padding=(3,3,3), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        # Residual layers
        # layer1: keep depth, downsample H/W by 2
        self.layer1 = ResidualBlock3D(16, 32, stride=(1,2,2))
        # layer2: keep depth, downsample H/W by 2
        self.layer2 = ResidualBlock3D(32, 64, stride=(1,2,2))
        # layer3: now downsample all dims
        self.layer3 = ResidualBlock3D(64, 128, stride=(2,2,2))

        # global pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc      = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)        # shape → [B,16, D, H/2, W/2]
        x = self.layer1(x)      # → [B,32, D, H/4, W/4]
        x = self.layer2(x)      # → [B,64, D, H/8, W/8]
        x = self.layer3(x)      # → [B,128, D/2, H/16, W/16]
        x = self.avgpool(x)     # → [B,128,1,1,1]
        x = x.view(x.size(0), -1)
        return self.fc(x)       # → [B,num_classes]


# === Grad-CAM wrapper ===
class GradCAMResNet3D(ResNet3D):
    def __init__(self, num_classes=3):
        super().__init__(in_channels=1, num_classes=num_classes)
        self.feature_maps = None

        def hook_fn(module, _in, out):
            self.feature_maps = out
            if out.requires_grad:
                out.retain_grad()

        # hook on the end of layer2 (64 channels at 32×32×32)
        self.layer2.register_forward_hook(hook_fn)


# === Training script ===
def train_model():
    csv_file = 'adni_preprocessed_npy_metadata.csv'
    batch_size = 32
    epochs = 500
    lr = 1e-5
    patience = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data + stratified split
    full_dataset = ADNIDataset(csv_file, augment=True)
    df = pd.read_csv(csv_file)
    labels = df['diagnosis'].astype(int).tolist()
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx),
                              batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(torch.utils.data.Subset(ADNIDataset(csv_file, augment=False), val_idx),
                            batch_size=batch_size, shuffle=False, num_workers=4)

    model = GradCAMResNet3D(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # CSV logging
    log_file = "training_log_resnet_gradcam.csv"
    with open(log_file,'w',newline='') as f:
        csv.writer(f).writerow(["epoch","train_loss","val_loss"])

    best_val = float('inf')
    patience_ctr = 0

    for epoch in range(1, epochs+1):
        # — train —
        model.train()
        tloss = 0
        for X,y in train_loader:
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out,y)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        tloss /= len(train_loader)

        # — val —
        model.eval()
        vloss = 0
        with torch.no_grad():
            for X,y in val_loader:
                X,y = X.to(device), y.to(device)
                out = model(X)
                vloss += criterion(out,y).item()
        vloss /= len(val_loader)

        # log
        with open(log_file,'a',newline='') as f:
            csv.writer(f).writerow([epoch, tloss, vloss])
        print(f"Epoch {epoch}/{epochs} | Train: {tloss:.4f}  Val: {vloss:.4f}")

        # checkpoint
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), "best_resnet3d_gradcam.pth")
            patience_ctr = 0
            print("  ✅ saved best model")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("  ⏹️ early stopping")
                break

    # final metrics
    model.load_state_dict(torch.load("best_resnet3d_gradcam.pth"))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X,y in val_loader:
            X,y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1).cpu().tolist()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(preds)
    names = {0:"MCI",1:"AD",2:"CN"}
    print(classification_report(y_true,y_pred,
                                target_names=[names[i] for i in sorted(set(y_true))]))


if __name__ == "__main__":
    train_model()
