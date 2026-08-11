"""
====================================================================================================
🧠 RAI v8.1 MULTI-SCALE RISK-AWARE NEURAL NETWORK (PREDICTION-ERROR RISK)
====================================================================================================
Features:
  1. Multi-Scale Temporal Convolutions (3d, 10d, 30d signals).
  2. Spatial Cross-Asset Transformer Encoder.
  3. Active Prediction-Error Risk Head (sigma_risk^2) trained on MSE(risk, |ret - val|).
  4. Risk-Modulated Actor Policy Head: Action logits = Actor([latent || prediction_error_risk]).
====================================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class MultiScaleRiskAwareNet(nn.Module):
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step
        self.action_dim = action_dim

        self.conv_short = nn.Conv1d(features_per_step, 24, kernel_size=3, padding=1)
        self.conv_med   = nn.Conv1d(features_per_step, 24, kernel_size=7, padding=3)
        self.conv_long  = nn.Conv1d(features_per_step, 24, kernel_size=15, padding=7)

        self.scale_fusion = nn.Sequential(nn.Conv1d(72, embed_dim, kernel_size=1), nn.GELU())

        trans_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256, dropout=0.05, batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=2)

        self.fc = nn.Sequential(nn.Linear(embed_dim * history_len, 128), nn.GELU(), nn.LayerNorm(128))

        # Prediction-Error Risk Estimator Head
        self.risk_head = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus())
        
        # Risk-Modulated Policy Head
        self.actor_head = nn.Sequential(nn.Linear(128 + 1, 64), nn.GELU(), nn.Linear(64, action_dim))
        self.critic_head = nn.Linear(128, 1)

        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        flat_obs = torch.nan_to_num(flat_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step).transpose(1, 2)

        s_feat = F.gelu(self.conv_short(x))
        m_feat = F.gelu(self.conv_med(x))
        l_feat = F.gelu(self.conv_long(x))

        multi_scale_cat = torch.cat([s_feat, m_feat, l_feat], dim=1)
        fused = self.scale_fusion(multi_scale_cat).transpose(1, 2)

        trans_out = self.transformer(fused)
        flat_repr = trans_out.reshape(b, -1)
        latent = self.fc(flat_repr)

        prediction_error_risk = torch.nan_to_num(self.risk_head(latent), nan=0.01)
        actor_input = torch.cat([latent, prediction_error_risk], dim=-1)

        actor_logits = torch.nan_to_num(self.actor_head(actor_input), nan=0.0)
        value = torch.nan_to_num(self.critic_head(latent), nan=0.0)

        return actor_logits, value, prediction_error_risk

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).to(DEVICE)
                if flat_obs.ndim == 1:
                    flat_obs = flat_obs.unsqueeze(0)
            logits, val, risk = self.forward(flat_obs)
            return logits.squeeze(0).cpu().numpy() if deterministic else Normal(logits, torch.exp(self.log_std)).sample().squeeze(0).cpu().numpy()
