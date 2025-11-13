import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .base import BaseModel

class AttentionGate3D(nn.Module):
    def __init__(self, F_x, F_g, F_int=None, upsample_mode="trilinear", eps=1e-8):
        super().__init__()
        if F_int is None: F_int = max(1, F_x // 2)

        self.W_x = nn.Conv3d(F_x, F_int, 1)
        self.W_g = nn.Conv3d(F_g, F_int, 1)
        
        self.n_x  = nn.InstanceNorm3d(F_int, affine=True, eps=1e-5)
        self.n_g  = nn.InstanceNorm3d(F_int, affine=True, eps=1e-5)

        self.psi = nn.Conv3d(F_int, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.upsample_mode = upsample_mode
        self.eps = eps

    def _resize_to_x(self, g, size_xyz):
        if g.shape[-3:] == size_xyz: return g
        return F.interpolate(g, size=size_xyz, mode=self.upsample_mode,
                             align_corners=False if "linear" in self.upsample_mode else None)

    def _minshift_norm(self, logits):
        # logits: [B,1,D,H,W] -> normalize across N=D*H*W for each sample
        B, C, D, H, W = logits.shape
        flat = logits.view(B, C, -1)                 # [B,1,N]
        a_min = flat.min(dim=2, keepdim=True)[0]     # [B,1,1]
        shifted = flat - a_min
        denom = shifted.sum(dim=2, keepdim=True)     # [B,1,1]
        att = shifted / (denom + self.eps)
        return att.view(B, C, D, H, W)               # [B,1,D,H,W]

    def forward(self, x, g):
        # project to F_int
        theta_x = self.n_x(self.W_x(x))                             # [B,F_int,D,H,W]
        phi_g = self.n_g(self.W_g(self._resize_to_x(g, x.shape[-3:]))) # [B,F_int,D,H,W]
        
        f = self.act(theta_x + phi_g)                                  # additive attention
        logits = self.psi(f)                                           # [B,1,D,H,W]
        att_map = torch.sigmoid(logits)                          # normalized across space

        x_att = x * att_map
        return x_att, att_map



# CBAM attention
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
            CBAM3D(out_channels)
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


class MyNet(BaseModel):
    def _build(self):
        
        # --- config ---
        num_classes = 2   # AD vs CN
        F0 = int(self.cfg.get('init_filters', 4))

        # --- shared backbone  ---
        self.initial_block = nn.Sequential(
            nn.Conv3d(1, F0, 7, padding=3),
            nn.InstanceNorm3d(F0),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)

        self.b1    = Block(F0,   2 * F0)
        self.b2    = Block(2*F0, 4 * F0)
        self.pool2 = nn.MaxPool3d(2)

        self.b3    = Block(4*F0, 8 * F0)

        # --- last conv block: shared feature extractor, not class-specific ---
        C_feat = 8 * F0
        self.b4_feat = Block(8*F0, C_feat)

        # global average pooling to vector
        self.gap = nn.AdaptiveAvgPool3d(1)

        # --- custom classifier with POSITIVE weights ---
        # raw weights (can be negative), we'll map them through softplus
        self.fc_weight_raw = nn.Parameter(torch.randn(num_classes, C_feat))
        self.fc_bias       = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        
        # shared backbone
        x  = self.pool1(self.initial_block(x))
        x1 = self.b1(x)
        x2 = self.pool2(self.b2(x1))
        x3 = self.b3(x2)

        # shared deep feature maps
        fm = self.b4_feat(x3)              # (B, C_feat, D, H, W)
        self.feature_maps = fm             # for inspection if you want

        # pooled feature vector
        pooled = self.gap(fm).view(x.size(0), -1)  # (B, C_feat)

        # enforce positive classifier weights via softplus
        w_pos = F.softplus(self.fc_weight_raw)      # (2, C_feat), all > 0

        # logits for 2 independent heads (AD, CN)
        logits = pooled @ w_pos.t() + self.fc_bias  # (B, 2)

        # ----- class-specific maps for interpretability & orthogonality -----
        # class_maps[b, k, d, h, w] = sum_c w_pos[k, c] * fm[b, c, d, h, w]
        # shape: (B, 2, D, H, W)
        self.class_maps = torch.einsum('bcxyz,kc->bkxyz', fm, w_pos)

        return logits  # logits[:, 0] = CN head, logits[:, 1] = AD head

if __name__ == "__main__":

    model = MyNet({})
    print(model)
    sample = torch.randn(1, 1, 79, 95, 79)
    out = model(sample)
    print(out.shape)
