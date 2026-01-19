import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

class Custom3DCNN(BaseModel):
    def _build(self):
        # Configuration
        in_channels  = int(self.cfg.get('in_channels', 1))
        num_classes  = int(self.cfg.get('num_classes', 3))
        depth        = int(self.cfg.get('depth', 3))
        init_filters = int(self.cfg.get('init_filters', 8))
        kernel_size = int(self.cfg.get('kernel_size', 3))
        input_shape  = tuple(self.cfg.get('input_shape', (128, 128, 128)))
        classifier_width = int(self.cfg.get('classifier_width', 128))

        # Build convolutional blocks dynamically
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = in_channels
        d, h, w  = input_shape
        for i in range(depth):
            out_channels = init_filters * (2 ** i)
            self.convs.append(nn.Conv3d(channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False))
            self.bns.append(nn.BatchNorm3d(out_channels))
            self.pools.append(nn.MaxPool3d(kernel_size=2))

            # Update for next block spatial dims
            channels = out_channels
            d //= 2; h //= 2; w //= 2

        # Compute flattened feature size for the first FC layer
        flat_features = channels * d * h * w

        # Fully connected layers
        self.fc1     = nn.Linear(flat_features, classifier_width)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(classifier_width, num_classes)

    def forward(self, x):
        # Apply convolutional blocks
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = pool(F.relu(bn(conv(x))))

        # Flatten and FC
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
            'init_filters': 8,
            'input_shape': (128, 128, 128)
        }
    }
    model = Custom3DCNN(cfg)
    print(model)
    sample = torch.randn(1, 1, 128, 128, 128)
    out = model(sample)
    print(out.shape)  # [1, 3]
