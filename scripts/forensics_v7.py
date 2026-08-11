"""
═══════════════════════════════════════════════════════════════════════════════
  RAI v7 — ALLOCATION FORENSICS: SCENARIO ADAPTATION ANALYSIS
  ════════════════════════════════════════════════════════════
  Check whether exposure to 7 causal scenarios in RAI v7 induced
  dynamic, regime-dependent allocation shifts or whether it also
  converged to a static allocation heuristic.
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V7_MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "v7_scenarios", "models")
V6_MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "robustness", "seeds")
REPORT_DIR = os.path.join(PROJECT_ROOT, "data", "v7_scenarios")

TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]

class DeepEndToEndTradingNet(nn.Module):
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64, nhead=2):
        super().__init__()
        self.history_len, self.features_per_step = history_len, features_per_step
        self.conv1d = nn.Sequential(nn.Conv1d(features_per_step, 32, 3, padding=1), nn.LeakyReLU(0.1),
                                    nn.Conv1d(32, embed_dim, 3, padding=1), nn.LeakyReLU(0.1))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=128, dropout=0.05, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.fc_features = nn.Sequential(nn.Linear(embed_dim, 128), nn.LeakyReLU(0.1))
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step).permute(0, 2, 1)
        x = self.conv1d(x).permute(0, 2, 1)
        x = self.transformer(x)
        return self.actor_head(self.fc_features(x.mean(dim=1))), self.critic_head(self.fc_features(x.mean(dim=1)))

    def get_action(self, flat_obs):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            return self.forward(flat_obs)[0].cpu().numpy().squeeze(0)

def track_weights(model, prices):
    T, N = prices.shape
    cash = 500.0
    shares = (9500.0 / N) / prices[30]
    records = []

    obs_h = []
    for t in range(30):
        p, pp = prices[t], prices[max(0, t-1)]
        np_ = np.pad(p/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.append(np.concatenate([np_, lr, [0.05, 0.]]).astype(np.float32))

    for t in range(30, T):
        act = model.get_action(np.concatenate(obs_h).astype(np.float32))
        cl = np.clip(act[0] - 2.5, -8., 3.)
        tc = 1.0 / (1.0 + np.exp(-cl))
        ts = 1.0 - tc
        n = min(N, 10)
        ea = np.exp(act[1:1+n] - np.max(act[1:1+n]))
        taw = (ea / ea.sum()) * ts

        p = prices[t]
        w = max(1e-4, cash + np.sum(shares * p))
        caw = (shares * p) / w
        ccf = cash / w

        if abs(ccf - tc) + np.sum(np.abs(caw - taw)) > 0.03:
            tv = abs(cash - w*tc) + np.sum(np.abs(shares*p - w*taw))
            net = max(1e-4, w - tv*0.0005)
            cash = net * tc
            shares = (net * taw) / np.maximum(1e-4, p)

        nw = cash + np.sum(shares * p)

        actual_w = np.zeros(N + 1)
        actual_w[:N] = (shares * p) / nw
        actual_w[-1] = cash / nw
        records.append(actual_w)

        pp = prices[t-1]
        np_ = np.pad(p/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.pop(0)
        obs_h.append(np.concatenate([np_, lr, [cash/max(1e-4,nw), 0.]]).astype(np.float32))

    return np.array(records)

def main():
    import yfinance as yf
    df = yf.download(TICKERS, start="2020-01-01", end="2026-08-08", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df = df['Close']
    df = df[TICKERS].dropna()
    prices = df.values

    v6_weights = []
    v7_weights = []

    for seed in range(1, 6):
        p6 = os.path.join(V6_MODELS_DIR, f"rai_v6_seed_{seed:02d}.pt")
        if os.path.exists(p6):
            m6 = DeepEndToEndTradingNet()
            m6.load_state_dict(torch.load(p6, weights_only=True))
            m6.eval()
            v6_weights.append(track_weights(m6, prices))

        p7 = os.path.join(V7_MODELS_DIR, f"rai_v7_scenario_seed_{seed:02d}.pt")
        if os.path.exists(p7):
            m7 = DeepEndToEndTradingNet()
            m7.load_state_dict(torch.load(p7, weights_only=True))
            m7.eval()
            v7_weights.append(track_weights(m7, prices))

    v6_arr = np.array(v6_weights) # (seeds, days, assets+1)
    v7_arr = np.array(v7_weights)

    v6_mean_w = np.mean(v6_arr, axis=0) # (days, assets+1)
    v7_mean_w = np.mean(v7_arr, axis=0)

    # Compute similarity between v6 and v7 allocations
    cos_sim = np.mean([1 - cosine(v6_mean_w[t], v7_mean_w[t]) for t in range(len(v6_mean_w))])
    l1_dist = np.mean(np.sum(np.abs(v6_mean_w - v7_mean_w), axis=1))

    print(f"\n  RAI v6 vs RAI v7 Allocation Similarity:")
    print(f"  Mean Cosine Similarity: {cos_sim:.4f}")
    print(f"  Mean L1 Weight Dist:    {l1_dist:.4f}")

    print(f"\n  Average Asset Weights (Full Period):")
    print(f"  {'Ticker':<8} {'RAI v6':>10} {'RAI v7':>10} {'Diff':>10}")
    print(f"  {'-'*40}")
    for i, t in enumerate(TICKERS):
        w6 = np.mean(v6_mean_w[:, i])
        w7 = np.mean(v7_mean_w[:, i])
        print(f"  {t:<8} {w6:>9.1%} {w7:>9.1%} {w7-w6:>+9.2%}")
    print(f"  {'Cash':<8} {np.mean(v6_mean_w[:, -1]):>9.1%} {np.mean(v7_mean_w[:, -1]):>9.1%} {np.mean(v7_mean_w[:, -1])-np.mean(v6_mean_w[:, -1]):>+9.2%}")

    # Standard deviation of weights over time (adaptation measure)
    std_v6 = np.mean(np.std(v6_mean_w[:, :10], axis=0))
    std_v7 = np.mean(np.std(v7_mean_w[:, :10], axis=0))
    print(f"\n  Weight Variation Over Time (Std):")
    print(f"  RAI v6: {std_v6:.5f}")
    print(f"  RAI v7: {std_v7:.5f}")

if __name__ == "__main__":
    main()
