"""
═══════════════════════════════════════════════════════════════════════════════
  INPUT FEATURE ABLATION DIAGNOSTIC
  ═════════════════════════════════
  Research Question:
    "What specific input information is actually responsible for the policy's
     transferable allocation behavior?"

  Ablation Tests (Masking input components during evaluation):
    1. Full Observation (Control baseline)
    2. Zero Out Price Levels (Keep log returns + portfolio state)
    3. Zero Out Log Returns (Keep price levels + portfolio state)
    4. Zero Out Portfolio State (Keep price levels + log returns, zero cash/dd)
    5. Zero Out History / Memory (Keep only most recent day, zero past 29 days)
    6. Constant Zeros (All inputs masked to zero)

  Measures:
    - Change in Return, Sharpe, Max Drawdown
    - Cosine Similarity of actions vs Unmasked Control
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
V6_DIR = os.path.join(PROJECT_ROOT, "data", "robustness", "seeds")
REPORT_DIR = os.path.join(PROJECT_ROOT, "data", "feature_ablation")
os.makedirs(REPORT_DIR, exist_ok=True)

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


def eval_with_ablation(model, prices, mode='full'):
    """Evaluates model with specific input features masked."""
    T, N = prices.shape
    cash = 500.0
    shares = (9500.0 / N) / prices[30]
    peak = 10000.0
    eq = [10000.0]
    actions_taken = []

    obs_h = []
    for t in range(30):
        p, pp = prices[t], prices[max(0, t-1)]
        np_ = np.pad(p/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.append(np.concatenate([np_, lr, [0.05, 0.]]).astype(np.float32))

    for t in range(30, T):
        full_obs = np.concatenate(obs_h).astype(np.float32)

        # ── Apply Feature Masking ──
        masked_obs = full_obs.copy()
        obs_matrix = masked_obs.reshape(30, 22)

        if mode == 'no_prices':
            obs_matrix[:, :10] = 0.0  # Zero out price ratios
        elif mode == 'no_returns':
            obs_matrix[:, 10:20] = 0.0  # Zero out log returns
        elif mode == 'no_portfolio_state':
            obs_matrix[:, 20:] = 0.0  # Zero out cash fraction & drawdown
        elif mode == 'no_history':
            obs_matrix[:29, :] = 0.0  # Zero out past 29 days (keep only day 30)
        elif mode == 'constant_zeros':
            obs_matrix[:, :] = 0.0  # Zero out EVERYTHING

        act = model.get_action(obs_matrix.flatten())
        actions_taken.append(act)

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
        peak = max(peak, nw)
        eq.append(nw)

        pp = prices[t-1]
        np_ = np.pad(p/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.pop(0)
        obs_h.append(np.concatenate([np_, lr, [cash/max(1e-4,nw), np.clip((nw-peak)/max(1e-4,peak),-1,0)]]).astype(np.float32))

    eq_a = np.array(eq)
    r = np.diff(eq_a) / np.maximum(1e-8, eq_a[:-1])
    pk = np.maximum.accumulate(eq_a)
    return {
        'return_pct': float((eq_a[-1]/eq_a[0]-1)*100),
        'sharpe': float(np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r) > 1e-8 else 0.,
        'max_dd_pct': float(np.min((eq_a-pk)/pk)*100),
        'actions': np.array(actions_taken)
    }


def main():
    W = 85
    print("="*W)
    print("  INPUT FEATURE ABLATION DIAGNOSTIC")
    print("  What information is actually responsible for transferable behavior?")
    print("="*W, flush=True)

    # Load real market prices (full period)
    local_test_path = os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "test_prices.csv")
    if not os.path.exists(local_test_path):
        print("  ⚠ test_prices.csv not found")
        return

    df = pd.read_csv(local_test_path, index_col=0, parse_dates=True)
    prices = df.values

    # Load 5 RAI v6 models
    models = []
    for seed in range(1, 6):
        p6 = os.path.join(V6_DIR, f"rai_v6_seed_{seed:02d}.pt")
        if os.path.exists(p6):
            m = DeepEndToEndTradingNet()
            m.load_state_dict(torch.load(p6, weights_only=True))
            m.eval()
            models.append(m)

    print(f"  ✓ Loaded {len(models)} RAI v6 seed models\n", flush=True)

    modes = [
        ('Full Obs (Control)', 'full'),
        ('Mask Price Levels (Keep returns + port)', 'no_prices'),
        ('Mask Log Returns (Keep prices + port)', 'no_returns'),
        ('Mask Portfolio State (Keep prices + returns)', 'no_portfolio_state'),
        ('Mask History (Keep only Day 30)', 'no_history'),
        ('Mask EVERYTHING (Constant All-Zero Obs)', 'constant_zeros'),
    ]

    print(f"  {'Ablation Mode':<42} | {'Return (%)':<12} | {'Sharpe':<8} | {'Max DD (%)':<10} | {'Action Cosine vs Control':<24}")
    print(f"  {'-'*105}")

    # Control actions to compute cosine similarity
    control_actions_per_seed = [eval_with_ablation(m, prices, mode='full')['actions'] for m in models]

    for mode_name, mode_key in modes:
        rets, shs, dds, cos_sims = [], [], [], []

        for s_idx, m in enumerate(models):
            res = eval_with_ablation(m, prices, mode=mode_key)
            rets.append(res['return_pct'])
            shs.append(res['sharpe'])
            dds.append(res['max_dd_pct'])

            ctrl_act = control_actions_per_seed[s_idx]
            curr_act = res['actions']
            # Compute average cosine similarity across time steps
            sim = np.mean([1 - cosine(ctrl_act[t], curr_act[t]) for t in range(len(ctrl_act))])
            cos_sims.append(sim)

        mean_ret = np.mean(rets)
        std_ret = np.std(rets)
        mean_sh = np.mean(shs)
        mean_dd = np.mean(dds)
        mean_cos = np.mean(cos_sims)

        marker = " ◄ DEGRADATION" if mean_cos < 0.90 else " ◄ UNCHANGED"
        print(f"  {mode_name:<42} | {mean_ret:>+6.2f}±{std_ret:<4.2f}% | {mean_sh:>7.2f} | {mean_dd:>+9.2f}% | {mean_cos:>11.4f}{marker}")

    print(f"\n{'═'*W}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"{'═'*W}\n", flush=True)


if __name__ == "__main__":
    main()
