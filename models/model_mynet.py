import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .base import BaseModel

def _gauss1d(sigma, device, dtype):
    r = max(1, int(3.0*sigma + 0.5))
    x = torch.arange(-r, r+1, device=device, dtype=dtype)
    k = torch.exp(-(x**2)/(2*sigma**2))
    return (k / (k.sum() + 1e-8))

def gaussian_blur3d(x, sigma_vox=1.0):
    # x: [N,1,D,H,W]
    kD = _gauss1d(sigma_vox, x.device, x.dtype).view(1,1,-1,1,1)
    kH = _gauss1d(sigma_vox, x.device, x.dtype).view(1,1,1,-1,1)
    kW = _gauss1d(sigma_vox, x.device, x.dtype).view(1,1,1,1,-1)
    for k, pad in [(kD, (0,0,0,0,kD.size(2)//2,kD.size(2)//2)),
                   (kH, (0,0,kH.size(3)//2,kH.size(3)//2,0,0)),
                   (kW, (kW.size(4)//2,kW.size(4)//2,0,0,0,0))]:
        x = F.pad(x, pad, mode='replicate')
        x = F.conv3d(x, k, groups=1)
    return x

@torch.no_grad()
def compute_mask(x, sigma=0.7, q=0.60, dilate_iters=1):
    # x: [N,1,D,H,W] (CPU or CUDA), returns float mask on same device/dtype
    blurred = gaussian_blur3d(x.float(), sigma_vox=sigma)      # [N,1,D,H,W]
    N, _, D, H, W = blurred.shape
    flat = blurred.view(N, -1)
    thr  = torch.quantile(flat, q, dim=1, keepdim=True)        # per-item threshold
    mask = (flat > thr).view(N, 1, D, H, W).to(x.dtype)

    # optional: small dilation so cortex at the rim isn’t clipped
    for _ in range(dilate_iters):
        mask = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)

    return mask  # [N,1,D,H,W], same device/dtype as x

def downsample_mask(mask, k=2, s=2):
    return F.max_pool3d(mask, kernel_size=k, stride=s)

    
class MaskedBatchNorm3d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.affine = affine
        self.track = track_running_stats
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias',   None)
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var',  torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x, mask):
        # x: [N,C,D,H,W], mask: [N,1,D,H,W] with {0,1}
        N, C, D, H, W = x.shape
        
        # sums over N,D,H,W per-channel
        msum  = mask.sum(dim=(0,2,3,4)) + 1e-8
        mean  = (x*mask).sum(dim=(0,2,3,4)) / msum
        var   = ((x - mean.view(1,C,1,1,1))**2 * mask).sum(dim=(0,2,3,4)) / msum

        if self.training and self.track:
            with torch.no_grad():
                self.num_batches_tracked += 1
                mom = self.momentum
                self.running_mean = (1-mom)*self.running_mean + mom*mean
                self.running_var  = (1-mom)*self.running_var  + mom*var
        if (not self.training) and self.track:
            mean, var = self.running_mean, self.running_var

        xhat = (x - mean.view(1,C,1,1,1)) / torch.sqrt(var.view(1,C,1,1,1) + self.eps)
        #xhat = x / torch.sqrt(var.view(1,C,1,1,1) + self.eps)
        if self.affine:
            xhat = xhat * self.weight.view(1,C,1,1,1) + self.bias.view(1,C,1,1,1)
            
        #Apply mask again
        xhat = xhat*mask
        
        return xhat
 
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
        self.mbn1 = MaskedBatchNorm3d(out_channels, affine=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.mbn2 = MaskedBatchNorm3d(out_channels, affine=False)

        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, mask):
        identity = self.shortcut(x)
        out = F.relu(self.mbn1(self.conv1(x), mask))
        out = self.mbn2(self.conv2(out), mask)
        out += identity
        return F.relu(out)

class MyNet(BaseModel):
    
    def _build(self):
        num_classes = 3
        bottleneck_channels = 8

        # Conv -> MaxPool
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1,bias=False)
        self.mbn1   = MaskedBatchNorm3d(16,affine=False)
        self.pool1 = nn.MaxPool3d(2, 2)

        # Residual block  -> MaxPool
        self.resblock = ResidualBlock3D(16, 32)
        self.pool2 = nn.MaxPool3d(2, 2)

        # Bottleneck conv -> (CBAM final) -> GMP -> FC
        self.conv4 = nn.Conv3d(32, bottleneck_channels, kernel_size=3, padding=1, bias=False)
        self.mbn4   = MaskedBatchNorm3d(bottleneck_channels, affine=False)
        self.cbam_final = CBAM3D(bottleneck_channels, reduction=4)  # <--- key spot
        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.fc  = nn.Linear(bottleneck_channels, num_classes)
            
    def forward(self, x):
        
        # Compute mask
        mask = compute_mask(x)
        
        # Block 1
        x = self.pool1(F.relu(self.mbn1(self.conv1(x), mask)))

        # Downsample mask
        mask = downsample_mask(mask)
        
        # Block 2 (residual)
        x = self.resblock(x, mask)
        x = self.pool2(x)

        # Downsample mask
        mask = downsample_mask(mask)
        
        # Bottleneck + final attention
        x = F.relu(self.mbn4(self.conv4(x), mask))
        self.cbam_out = self.cbam_final(x)        # <-- most important for Grad-CAM


        # Save for Grad-CAM (after attention so maps reflect reweighted features)

        self.gmp_out = self.gmp(self.cbam_out).view(self.cbam_out.size(0), -1)
        
        logits = self.fc(self.gmp_out)
        return logits
