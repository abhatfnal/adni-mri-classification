import torch
import torch.nn as nn

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
