import torch
import torch.nn as nn
import torch.nn.functional as F

class SAM3d(nn.Module):
    """
    3d Spatial attention module
    """
    
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv3d(2,1,kernel_size=7,padding=7//2)
        
    def forward(self, x):
        
        channel_max = torch.max(x, 1, keepdim=True)[0]
        channel_avg = torch.sum(x, 1, keepdim=True)/x.shape[1]
        
        m = torch.concat((channel_avg, channel_max), dim=1)
        m = torch.sigmoid(self.conv(m))
        
        return m*x


class ResidualBlock(nn.Module):
    
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, sam=False):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, kernel_size, stride=stride, padding=kernel_size//2),
            nn.GroupNorm(8, channel_out),
            nn.ReLU(),
            nn.Conv3d(channel_out, channel_out, kernel_size, padding=kernel_size//2),
            nn.GroupNorm(8, channel_out),
            SAM3d() if sam else nn.Identity(),
        )
        
        self.skip1 = nn.Conv3d(channel_in, channel_out, 1, stride=stride, padding=0)
        
    def forward(self, x):
        
        x = F.relu(self.residual(x) + self.skip1(x))
        
        return x
