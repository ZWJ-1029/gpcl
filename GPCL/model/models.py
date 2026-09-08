"""Paper 1D-CNN encoder and frozen-backbone f-CaO regressor."""

import torch
from torch import nn
from torch.nn import functional as F

from config import Config


class GPCL1DEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        c1, c2 = cfg.conv_channels
        k = cfg.conv_kernel_size
        self.features = nn.Sequential(
            nn.Conv1d(cfg.n_variables, c1, kernel_size=k, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=k, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(cfg.max_pool_size),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, cfg.n_variables, cfg.window_size)
            flattened = self.features(dummy).numel()
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, cfg.projection_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.projection_hidden_dim, cfg.embedding_dim),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # CSV/data tensors arrive as (batch, time, variable).
        return self.features(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.forward_features(x)), dim=1)


class FrozenGPCLRegressor(nn.Module):
    def __init__(self, pretrained: GPCL1DEncoder, cfg: Config):
        super().__init__()
        self.features = pretrained.features
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        channels = cfg.conv_channels[-1]
        self.head = nn.Sequential(
            nn.Linear(channels, cfg.regression_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.regression_hidden_dim, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            temporal = self.features(x.transpose(1, 2))
            pooled = temporal.mean(dim=2)  # paper's global average pooling
        return self.head(pooled).squeeze(1)

