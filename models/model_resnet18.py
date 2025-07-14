import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

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


class BasicBlock1(nn.Module):
    
    def __init__(self, channel_in):
        super().__init__()
        
        self.residual1 = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, 3, padding=3//2),
            nn.BatchNorm3d(channel_in),
            nn.ReLU(),
            nn.Conv3d(channel_in, channel_in, 3, padding=3//2),
            nn.BatchNorm3d(channel_in),
            SAM3d()
        )
        
        self.residual2 = nn.Sequential(
            nn.Conv3d(channel_in, channel_in, 3, padding=3//2),
            nn.BatchNorm3d(channel_in),
            nn.ReLU(),
            nn.Conv3d(channel_in, channel_in, 3, padding=3//2),
            nn.BatchNorm3d(channel_in),
            SAM3d()
        )
        
    def forward(self, x):
        
        # First residual + skip connection, then activation
        x = F.relu(self.residual1(x) + x)
        
        # Second residual + skip connection, then activation
        x = F.relu(self.residual2(x) + x)
        
        return x

class BasicBlock2(nn.Module):
    """
    Same as BasicBlock1 but doubles output channels and halves spatial dimensions
    """
    
    def __init__(self, channel_in):
        super().__init__()
        
        channel_out = 2*channel_in
        
        self.residual1 = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, 3, stride=2, padding=3//2),
            nn.BatchNorm3d(channel_out),
            nn.ReLU(),
            nn.Conv3d(channel_out, channel_out, 3, padding=3//2),
            nn.BatchNorm3d(channel_out),
            SAM3d()
        )
        
        # First skip connection (linear map)
        self.skip1 = nn.Conv3d(channel_in, channel_out, 1, stride=2, padding=0)
        
        self.residual2 = nn.Sequential(
            nn.Conv3d(channel_out, channel_out, 3, padding=3//2),
            nn.BatchNorm3d(channel_out),
            nn.ReLU(),
            nn.Conv3d(channel_out, channel_out, 3, padding=3//2),
            nn.BatchNorm3d(channel_out),
            SAM3d()
        )
        
    def forward(self, x):
        
        # First residual + skip connection, then activation
        x = F.relu(self.residual1(x) + self.skip1(x))
        
        # Second residual + skip connection, then activation
        x = F.relu(self.residual2(x) + x)
        
        return x
    
    
class ResNet18(BaseModel):
    
    def _build(self):
        
        # Configuration
        num_classes  = int(self.cfg.get('num_classes', 3))
        initial_filters = int(self.cfg.get('init_filters', 8))
        input_shape  = tuple(self.cfg.get('input_shape', (128, 128, 128)))
        
        avgpool_kernel_size = 2
        
        # initial block
        self.initial_block = nn.Sequential(
            nn.Conv3d(1, initial_filters, 7, padding=7//2),
            nn.BatchNorm3d(initial_filters),
            nn.ReLU()
        )   # (4, 128, 128, 128)
        
        # pooling
        self.pool1 = nn.MaxPool3d(2)                # (4, 64, 64, 64)
        
        # basic block 1
        self.bb_1 = BasicBlock1(initial_filters)    # (4, 64, 64, 64)
        
        # basic blocks 2 - 4
        self.bb_2 = BasicBlock2(initial_filters)    # (8, 32, 32, 32)
        self.bb_3 = BasicBlock2(initial_filters*2)  # (16, 16, 16, 16)
        self.bb_4 = BasicBlock2(initial_filters*4)  # (32, 8, 8, 8)
        
        # batch norm
        self.norm = nn.BatchNorm3d(initial_filters*8)
        
        # avg pool
        self.avgpool = nn.AvgPool3d(avgpool_kernel_size)
        
        # out: (32, 4, 4, 4)
        
        # final fc layer
        self.fc = nn.Linear(initial_filters*8*4*4*4, num_classes)
        
    def forward(self, x):
        
        x = self.pool1(self.initial_block(x))
        
        x = self.bb_1(x)
        x = self.bb_2(x)
        x = self.bb_3(x)
        x = self.bb_4(x)
        
        x = self.avgpool(self.norm(x))
        
        x = x.view(x.size(0), -1)
        
        return self.fc(x)
    
if __name__ == "__main__":
    
    sam = SAM3d()
    x = torch.randn(1,5,128,128,128)
    out = sam(x)
    print(out.shape)
    
    # model = ResNet18({})
    # print(model)
    # sample = torch.randn(1, 1, 128, 128, 128)
    # out = model(sample)
    # print(out.shape)  # Should be [1, 3] for 3 classes (CN, MCI, AD)