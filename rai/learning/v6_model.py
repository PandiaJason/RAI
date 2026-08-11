"""
RAI v6 Neural Network Model Architecture
=========================================
Multi-Scale Conv1D + Transformer Encoder + Deep Actor-Critic Head
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class DeepEndToEndTradingNet(nn.Module):
    """
    RAI v6 End-to-End Trading Network:
      Raw Obs (30 x features) -> Multi-Scale Conv1D -> Transformer Encoder -> Actor/Critic
    """
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64, nhead=2):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step

        self.conv1d = nn.Sequential(
            nn.Conv1d(features_per_step, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=128,
            dropout=0.05, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * history_len, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
        )
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step)
        x_trans = x.transpose(1, 2)
        conv_out = self.conv1d(x_trans).transpose(1, 2)
        trans_out = self.transformer(conv_out)
        flat = trans_out.reshape(b, -1)
        feat = self.fc(flat)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            mean, _ = self.forward(flat_obs)
            if deterministic:
                return mean.cpu().numpy().squeeze(0)
            dist = Normal(mean, torch.exp(self.log_std))
            return dist.sample().cpu().numpy().squeeze(0)
