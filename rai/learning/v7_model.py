"""
═══════════════════════════════════════════════════════════════════════════════
  RAI v7: SPATIO-TEMPORAL TRANSFORMER NEURAL NETWORK ARCHITECTURE
  ═══════════════════════════════════════════════════════════════════════════════
  Upgrades RAI v6 Conv1D+Transformer architecture with:
    1. Multi-kernel Temporal Conv Block (captures 3-day micro and 7-day trend features)
    2. Multi-Head Spatio-Temporal Transformer Encoder
    3. Sortino-focused Actor/Critic heads
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class MultiScaleConvBlock(nn.Module):
    """Extracts features across multiple temporal kernel scales (3-day micro & 7-day macro)."""
    def __init__(self, in_channels=22, out_channels=64):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=7, padding=3)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        # x shape: (B, C, T)
        c3 = self.act(self.conv3(x))
        c7 = self.act(self.conv7(x))
        return torch.cat([c3, c7], dim=1)  # (B, out_channels, T)


class SpatioTemporalTradingNet(nn.Module):
    """RAI v7 Deep Neural Network Architecture."""
    
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64, nhead=4):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step

        # 1. Multi-scale Convolutional Feature Extractor
        self.multiscale_conv = MultiScaleConvBlock(in_channels=features_per_step, out_channels=embed_dim)

        # 2. Spatio-Temporal Transformer Encoder (2 Transformer Layers, 4 Heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=128, 
            dropout=0.05, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Dense Representation Head
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
        )

        # 4. Actor & Critic Heads
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step)
        
        # Conv over time: (B, T, C) -> (B, C, T)
        x_conv = self.multiscale_conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T, embed_dim)
        
        # Spatio-Temporal Transformer
        x_trans = self.transformer(x_conv)
        
        # Pooling across time steps
        latent = x_trans.mean(dim=1)
        feat = self.fc(latent)
        
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0) if deterministic else Normal(mean, torch.exp(self.log_std)).sample().cpu().numpy().squeeze(0)
