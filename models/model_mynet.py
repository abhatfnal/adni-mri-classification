import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CBAM 3D ---
class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False)
        )
        self.gap_out = None

    def forward(self, x):
        b, c, d, h, w = x.size()
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool3d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1, 1)
        return x * attn

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        # compress channel dim with avg+max, then conv
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_map, max_map], dim=1)
        attn = torch.sigmoid(self.conv(x_cat))
        return x * attn

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention3D(channels, reduction)
        self.sa = SpatialAttention3D(spatial_kernel)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels, affine=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels, affine=False)

        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class MyNet(BaseModel):
    def _build(self):
        num_classes = 3
        bottleneck_channels = 8

        # Conv -> MaxPool
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1,bias=False)
        self.bn1   = nn.BatchNorm3d(16,affine=False)
        self.pool1 = nn.MaxPool3d(2, 2)

        # Residual block  -> MaxPool
        self.resblock = ResidualBlock3D(16, 32)
        self.pool2 = nn.MaxPool3d(2, 2)

        # Bottleneck conv -> (CBAM final) -> GMP -> FC
        self.conv4 = nn.Conv3d(32, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.bn4   = nn.BatchNorm3d(bottleneck_channels, affine=False)
        self.cbam_final = CBAM3D(bottleneck_channels, reduction=4)  # <--- key spot
        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.fc  = nn.Linear(bottleneck_channels, num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Block 2 (residual)
        x = self.resblock(x)
        x = self.pool2(x)

        # Bottleneck + final attention
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.cbam_final(x)        # <-- most important for Grad-CAM

        # Save for Grad-CAM (after attention so maps reflect reweighted features)
        self.feature_maps = x

        self.gap_out = self.gmp(x).view(x.size(0), -1)
        
        logits = self.fc(self.gap_out)
        return logits

