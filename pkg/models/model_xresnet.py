import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .base import BaseModel


# class StdNorm3d(nn.Module):
#     def __init__(self, eps=1e-6, floor=1e-2):
#         super().__init__()
#         self.eps, self.floor = eps, floor
#     def forward(self, x):
#         std = x.pow(2).mean(dim=(2,3,4), keepdim=True).add(self.eps).sqrt()
#         std = std.clamp_min(self.floor)   # robust floor
#         return x / std

class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )
        self.gap_out = None

    def forward(self, x):
        b, c, d, h, w = x.size()
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool3d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1, 1)
        return x * attn

class SpatialAttention3d(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        # compress channel dim with avg+max, then conv
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad)

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
        self.sa = SpatialAttention3d(spatial_kernel)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=3//2),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d( out_channels, out_channels, 3, padding=3//2),
            nn.InstanceNorm3d(out_channels),
            SpatialAttention3d()
        )
        
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        
    def forward(self, x):
        
        # residual + skip connection, then activation
        x = F.relu(self.residual(x) + self.shortcut(x))
        
        return x
    
class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.seq = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )
        
    def forward(self, x):
        return self.seq(x)
    
class XResNet(BaseModel):
    
    def _build(self):
        
        # Configuration
        num_classes  = int(self.cfg.get('num_classes', 3))
        initial_filters = int(self.cfg.get('init_filters', 4))
        
        # initial block
        self.initial_block = nn.Sequential(
            nn.Conv3d(1, initial_filters, 7, padding=7//2),
            nn.InstanceNorm3d(initial_filters),
            nn.ReLU()
        ) 
        self.pool1 = nn.MaxPool3d(2)            

        self.b1 = Block(initial_filters, initial_filters*2)
        self.b2 = Block(initial_filters*2, initial_filters*2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.b3 = Block(initial_filters*2, initial_filters*4)
        self.b4 = Block(initial_filters*4, initial_filters*8)
        
        # CBAM 
        #self.cbam = CBAM3D(initial_filters*8)
        
        # Global max/average pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # final fc layer (hardcoded dims)
        self.fc1 = nn.Linear(initial_filters*8, num_classes)
        
    def forward(self, x):
        
        x = self.pool1(self.initial_block(x))
        
        x = self.pool2(self.b2(self.b1(x)))
        
        self.feature_maps = self.b4(self.b3(x))
        
        x = self.gap(self.feature_maps)
        
        x = x.view(x.size(0), -1)
        
        return self.fc1(x)
    
if __name__ == "__main__":

    model = XResNet({})
    print(model)
    sample = torch.randn(1, 1, 79, 95, 79)
    out = model(sample)
    print(out.shape)
