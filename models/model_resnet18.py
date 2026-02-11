import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel


class SAM3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        channel_max = torch.max(x, 1, keepdim=True)[0]
        channel_avg = torch.sum(x, 1, keepdim=True) / x.shape[1]
        m = torch.cat((channel_avg, channel_max), dim=1)
        m = torch.sigmoid(self.conv(m))
        return m * x


class BasicBlock1(nn.Module):
    def __init__(self, channel_in, dropout=0.3):
        super().__init__()
        self.residual1 = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, 3, padding=1),
            nn.BatchNorm3d(channel_in),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channel_in, channel_in, 3, padding=1),
            nn.BatchNorm3d(channel_in),
            SAM3d()
        )
        self.residual2 = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, 3, padding=1),
            nn.BatchNorm3d(channel_in),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channel_in, channel_in, 3, padding=1),
            nn.BatchNorm3d(channel_in),
            SAM3d()
        )

    def forward(self, x):
        x = F.relu(self.residual1(x) + x)
        x = F.relu(self.residual2(x) + x)
        return x


class BasicBlock2(nn.Module):
    def __init__(self, channel_in, dropout=0.3):
        super().__init__()
        channel_out = 2 * channel_in

        self.residual1 = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, 3, stride=2, padding=1),
            nn.BatchNorm3d(channel_out),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channel_out, channel_out, 3, padding=1),
            nn.BatchNorm3d(channel_out),
            SAM3d()
        )

        self.skip1 = nn.Conv3d(channel_in, channel_out, 1, stride=2)

        self.residual2 = nn.Sequential(
            nn.Conv3d(channel_out, channel_out, 3, padding=1),
            nn.BatchNorm3d(channel_out),
            nn.ReLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(channel_out, channel_out, 3, padding=1),
            nn.BatchNorm3d(channel_out),
            SAM3d()
        )

    def forward(self, x):
        x = F.relu(self.residual1(x) + self.skip1(x))
        x = F.relu(self.residual2(x) + x)
        return x


class ResNet18(BaseModel):
    def _build(self):
        self.num_classes = int(self.cfg.get('num_classes', 3))
        self.initial_filters = int(self.cfg.get('init_filters', 8))
        self.in_channels = int(self.cfg.get('in_channels', 1))
        self.dropout_rate = float(self.cfg.get('dropout', 0.3))
        self.avgpool_kernel_size = 2

        self.initial_block = nn.Sequential(
            nn.Conv3d(self.in_channels, self.initial_filters, kernel_size=7, padding=3),
            nn.BatchNorm3d(self.initial_filters),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool3d(2)

        self.bb_1 = BasicBlock1(self.initial_filters, dropout=self.dropout_rate)
        self.bb_2 = BasicBlock2(self.initial_filters, dropout=self.dropout_rate)
        self.bb_3 = BasicBlock2(self.initial_filters * 2, dropout=self.dropout_rate)
        self.bb_4 = BasicBlock2(self.initial_filters * 4, dropout=self.dropout_rate)

        self.norm = nn.BatchNorm3d(self.initial_filters * 8)
        self.avgpool = nn.AvgPool3d(self.avgpool_kernel_size)

        self.fc1 = None  # will be initialized in forward()

    def forward_features(self, x):
        x = self.pool1(self.initial_block(x))
        x = self.bb_1(x)
        x = self.bb_2(x)
        x = self.bb_3(x)
        x = self.bb_4(x)
        x = self.avgpool(self.norm(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), self.num_classes).to(x.device)
        return self.fc1(x)


if __name__ == "__main__":
    model = ResNet18({
        'num_classes': 3,
        'in_channels': 1,
        'init_filters': 8,
        'dropout': 0.3
    })
    print(model)
    sample = torch.randn(1, 1, 128, 128, 128)
    out = model(sample)
    print(out.shape)
