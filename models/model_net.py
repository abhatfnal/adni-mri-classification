import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .base import BaseModel

class StdNorm3d(nn.Module):
    """
    Normalizes by dividing by standard deviation. Done channel and sample wise.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x has dim (N,C,D,W,H)
        # Compute std over (D,W,H) for each sample and channel
        std = x.std(dim=(2, 3, 4), keepdim=True)
        # Avoid division by zero
        std = std + 1e-6
        return x / std 
    
class SAM3d(nn.Module):
    """
    3d Spatial attention module
    """
    
    def __init__(self):
        super().__init__()
        
        # (7x7x7) convolutional layer
        self.conv = nn.Conv3d(2,1,kernel_size=7,padding=7//2)  # mantain original size
        
    def forward(self, x):
        
        # Assume x has size (N,C,D,H,W)
        
        # Take max in the channel dimension. Size: (N,1,D,H,W)
        channel_max = torch.max(x, 1, keepdim=True)[0]
        
        # Take average in the channel dimension: Size (N,1,D,H,W)
        channel_avg = torch.sum(x, 1, keepdim=True)/x.shape[1]
        
        # Concatenate vectors: Size (N,2,D,H,W)
        m = torch.concat((channel_avg, channel_max), dim=1)
        
        # Apply convolution and sigmoid. Size (N,1,D,H,W)
        m = F.sigmoid(self.conv(m))
        
        # Element wise multiplication 
        return m*x  
    

class Block(nn.Module):
    
    def __init__(self, channel_in, channel_out):
        super().__init__()
        
        self.residual1 = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, kernel_size=3, padding=3//2, bias=False),
            StdNorm3d(),
            nn.ReLU(),
            nn.Conv3d(channel_out, channel_out, kernel_size=3, padding=3//2, bias=False),
            StdNorm3d(),
            SAM3d()
        )
        
        # First skip connection (linear map)
        self.skip1 = nn.Conv3d(channel_in, channel_out, kernel_size=3, padding=3//2, bias=False)
        
        self.residual2 = nn.Sequential(
            nn.Conv3d(channel_out, channel_out, kernel_size=3, padding=3//2, bias=False),
            StdNorm3d(),
            nn.ReLU(),
            nn.Conv3d(channel_out, channel_out, kernel_size=3, padding=3//2, bias=False),
            StdNorm3d(),
            SAM3d()
        )
        
    def forward(self, x):
        
        # First residual + skip connection, then activation
        x = F.relu(self.residual1(x) + self.skip1(x))
        
        # Second residual + skip connection, then activation
        x = F.relu(self.residual2(x) + x)
        
        return x
    
class Net(BaseModel):
    
    def _build(self):
        
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=3//2, bias=False)
        self.pool1 = nn.MaxPool3d(2)
        
        self.block1 = Block(8,16)
        self.pool2 = nn.MaxPool3d(2)
        
        self.block2 = Block(16,8)
        self.dropout = nn.Dropout3d(0.2)
        self.gmp = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Linear(8,3)
        
    def forward(self, x):
        
        x = self.pool1(self.conv1(x))
        
        x = self.pool2(self.block1(x))
        
        self.feature_maps = self.block2(x)
        
        x = self.dropout(self.feature_maps)
        
        logits = self.fc(torch.flatten(self.gmp(x), 1))
        
        return logits
    
if __name__ == "__main__":
    model = Net({})
    x = torch.rand((1,1,79,95,79))
    
    out = model(x)
    print(out.shape)