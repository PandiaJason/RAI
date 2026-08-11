"""
====================================================================================================
🧠 RAI v8 MULTI-SCALE UNCERTAINTY-AWARE TRADING ARCHITECTURE
====================================================================================================
Includes:
  1. Multi-Scale Temporal Encoders (Kernel Sizes 3, 7, 15 for 3d, 10d, 30d signals).
  2. Spatial Cross-Asset Attention Transformer.
  3. Risk & Epistemic Uncertainty Estimation Head (sigma_risk^2).
  4. Actor Logits & Critic Value Function.
====================================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MultiScaleUncertaintyNet(nn.Module):
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step
        self.action_dim = action_dim

        # 1. Multi-Scale Temporal Convolutions (Short, Medium, Long horizons)
        self.conv_short = nn.Conv1d(features_per_step, 24, kernel_size=3, padding=1)
        self.conv_med   = nn.Conv1d(features_per_step, 24, kernel_size=7, padding=3)
        self.conv_long  = nn.Conv1d(features_per_step, 24, kernel_size=15, padding=7)

        self.scale_fusion = nn.Sequential(
            nn.Conv1d(72, embed_dim, kernel_size=1),
            nn.GELU()
        )

        # 2. Spatial Cross-Asset Transformer Encoder
        trans_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256, dropout=0.05, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=2)

        # 3. Dense Representation Backbone
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * history_len, 128),
            nn.GELU(),
            nn.LayerNorm(128)
        )

        # 4. Heads: Actor, Critic, and Epistemic Uncertainty Head
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Returns positive variance estimation
        )

        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        # Reshape to (Batch, Features, Sequence_Length)
        x = flat_obs.reshape(b, self.history_len, self.features_per_step).transpose(1, 2)

        # Extract multi-scale feature maps
        s_feat = F.gelu(self.conv_short(x))
        m_feat = F.gelu(self.conv_med(x))
        l_feat = F.gelu(self.conv_long(x))

        # Concatenate along channel dimension & fuse
        multi_scale_cat = torch.cat([s_feat, m_feat, l_feat], dim=1)
        fused = self.scale_fusion(multi_scale_cat).transpose(1, 2)

        # Pass through Transformer
        trans_out = self.transformer(fused)
        flat_repr = trans_out.reshape(b, -1)
        latent = self.fc(flat_repr)

        # Compute Actor, Critic, and Uncertainty Variance
        actor_logits = self.actor_head(latent)
        value = self.critic_head(latent)
        epistemic_uncertainty = self.uncertainty_head(latent)

        return actor_logits, value, epistemic_uncertainty

    def get_action(self, flat_obs, deterministic=True, device='cpu'):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).to(device)
                if flat_obs.ndim == 1:
                    flat_obs = flat_obs.unsqueeze(0)
            logits, val, uncertainty = self.forward(flat_obs)
            
            if deterministic:
                action = logits.squeeze(0).cpu().numpy()
            else:
                dist = Normal(logits, torch.exp(self.log_std))
                action = dist.sample().squeeze(0).cpu().numpy()
                
            return action, uncertainty.squeeze(0).cpu().numpy().item()
