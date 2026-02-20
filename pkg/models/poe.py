import torch
import torch.nn as nn

from .base import BaseModel
from .components import ResidualBlock

class Expert(nn.Module):

    def __init__(self, latent_dim):
        self.__init__()

        self.net = nn.Sequential(

            nn.Conv3d(1,32,kernel_size=3,stride=2,padding=3//2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),

            ResidualBlock(32, 32),
            ResidualBlock(32, 32),

            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),

            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

        self.mlp_mu = nn.Linear(128, latent_dim)
        self.mlp_log_sigma = -nn.functional.relu(nn.Linear(128, 1))    # Impose sigma < 1

    def forward(self, x):

        x = self.net(x)
        mu = self.mlp_mu(x)
        log_sigma = self.mlp_log_sigma(x)

        return mu, log_sigma

    
class PoE(BaseModel):

    def __init__(self, n_modalities, latent_dim=128):
        self.n_modalities = n_modalities
        self.latent_dim = latent_dim

        self.experts = nn.ModuleList([  Expert(latent_dim) for i in range(n_modalities) ])

    def forward(self, X):
        pass 

    def train_batch(self, batch, batch_index):

        # 
        return super().train_batch(batch, batch_index)
    
    def validate_batch(self, batch, batch_index):
        return super().validate_batch(batch, batch_index)

    def test_batch(self, batch, batch_index):
        return super().test_batch(batch, batch_index)

