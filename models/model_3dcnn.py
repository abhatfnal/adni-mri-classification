import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

class Simple3DCNN(BaseModel):
    
    def _build(self):
        # Configuration
        model_cfg    = self.cfg.get('model', {})
        in_channels  = int(model_cfg.get('in_channels', 1))
        num_classes  = int(model_cfg.get('num_classes', 3))
        depth        = int(model_cfg.get('depth', 3))
        init_filters = int(model_cfg.get('init_filters', 8))

        # Dynamically build convolutional blocks
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = in_channels
        for i in range(depth):
            out_channels = init_filters * (2 ** i)
            self.convs.append(nn.Conv3d(channels, out_channels, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm3d(out_channels))
            self.pools.append(nn.MaxPool3d(kernel_size=2))
            channels = out_channels

        # Adaptive pooling to 1x1x1 to remove spatial dependency
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Fully connected layers
        self.fc1     = nn.Linear(channels, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional feature extraction
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = pool(F.relu(bn(conv(x))))

        # Fixed-size pooling and classification
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Example usage
    cfg = {
        'model': {
            'in_channels': 1,
            'num_classes': 3,
            'depth': 3,
            'init_filters': 8
        }
    }
    model = Simple3DCNN(cfg)
    print(model)
    sample = torch.randn(1, 1, 128, 128, 128)
    out = model(sample)
    print(out.shape)  # [1, 3]
