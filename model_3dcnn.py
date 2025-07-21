import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,  8, 3, padding=1)
        self.bn1   = nn.BatchNorm3d(8)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(16)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(16, 32, 3, padding=1)
        self.bn3   = nn.BatchNorm3d(32)
        self.pool3 = nn.MaxPool3d((1,2,2))     # or nn.MaxPool3d(2)

        # dummy forward pass to infer flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 128, 128, 128)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            self._to_linear = x.numel()  # 1 * C * D * H * W

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), self._to_linear)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


if __name__ == "__main__":
    model = Simple3DCNN()
    print(model)
    sample = torch.randn(1, 1, 128, 128, 128)
    out = model(sample)
    print(out.shape)  # Should be [1, 3] for 3 classes (CN, MCI, AD)
