"""
═══════════════════════════════════════════════════════════════════════════════
  REAL-DATA TRAINED DL MODELS vs. RAI ZERO-SHOT MULTI-DATASET EXPERIMENT
  ═══════════════════════════════════════════════════════════════════════
  Scientific Benchmark:
    • Other DL Models: Trained on 70% REAL HISTORICAL DATA, Tested on 30% OOS
    • RAI v6 Agent   : Trained on 0% REAL DATA (100% Synthetic), Tested ZERO-SHOT on 30% OOS

  Datasets Evaluated:
    1. US ETFs Universe (2007–2026 / 4,863 days)
    2. Crypto Assets Universe (2020–2026 / 2,148 days)
    3. US Mega-Cap Stocks Universe (2015–2026 / 2,916 days)
    4. Global Equity Indices Universe (2015–2026 / 2,916 days)
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6_DIR = os.path.join(PROJECT_ROOT, "data", "robustness", "seeds")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "real_vs_rai_benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    "1. US ETFs Universe": {
        "tickers": ["SPY", "QQQ", "EEM", "VNQ", "HYG", "TLT", "DBC", "GLD", "USO", "UUP"],
        "start": "2007-01-01", "end": "2026-08-08"
    },
    "2. Crypto Assets": {
        "tickers": ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "DOT-USD"],
        "start": "2020-07-01", "end": "2026-08-08"
    },
    "3. US Mega-Cap Stocks": {
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "LLY", "JPM", "JNJ", "WMT"],
        "start": "2015-01-01", "end": "2026-08-08"
    },
    "4. Global Equity Indices": {
        "tickers": ["SPY", "EWJ", "EWG", "EWU", "MCHI", "INDA", "EWZ", "EFA", "EEM", "FXI"],
        "start": "2015-01-01", "end": "2026-08-08"
    }
}


# ═══════════════════════════════════════════════
#  REAL DATA GYM ENVIRONMENT FOR REAL-DATA TRAINING
# ═══════════════════════════════════════════════
class RealMarketTradingEnv(gym.Env):
    def __init__(self, prices, fee_bps=5):
        super().__init__()
        self.prices = prices.astype(np.float32)
        self.T, self.N = prices.shape
        self.fee_bps = fee_bps

        # Observation shape: 30 days * 22 features = 660
        self.observation_space = spaces.Box(low=-10., high=10., shape=(660,), dtype=np.float32)
        self.action_space = spaces.Box(low=-5., high=5., shape=(11,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_step = 30
        self.cash = 500.0
        self.shares = (9500.0 / self.N) / self.prices[30]
        self.peak = 10000.0
        self.obs_h = []
        for t in range(30):
            p, pp = self.prices[t], self.prices[max(0, t-1)]
            np_ = np.pad(p/self.prices[30], (0, max(0, 10-self.N)), constant_values=1.)[:10]
            lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-self.N)), constant_values=0.)[:10]
            self.obs_h.append(np.concatenate([np_, lr, [0.05, 0.]]).astype(np.float32))
        return np.concatenate(self.obs_h).astype(np.float32), {}

    def step(self, act):
        cl = np.clip(act[0] - 2.5, -8., 3.)
        tc = 1.0 / (1.0 + np.exp(-cl))
        ts = 1.0 - tc
        n = min(self.N, 10)
        ea = np.exp(act[1:1+n] - np.max(act[1:1+n]))
        taw = (ea / ea.sum()) * ts

        p = self.prices[self.curr_step].copy()
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        caw = (self.shares * p) / w
        ccf = self.cash / w

        if abs(ccf - tc) + np.sum(np.abs(caw - taw)) > 0.03:
            tv = abs(self.cash - w * tc) + np.sum(np.abs(self.shares * p - w * taw))
            net = max(1e-4, w - tv * (self.fee_bps/10000.0))
            self.cash = net * tc
            self.shares = (net * taw) / np.maximum(1e-4, p)

        nw = self.cash + np.sum(self.shares * self.prices[self.curr_step])
        self.peak = max(self.peak, nw)
        dd = (nw - self.peak) / self.peak

        reward = (nw - w) / w - 0.5 * max(0, -dd)

        self.curr_step += 1
        done = (self.curr_step >= self.T)

        if not done:
            pp = self.prices[self.curr_step-1]
            np_ = np.pad(p/self.prices[30], (0, max(0, 10-self.N)), constant_values=1.)[:10]
            lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-self.N)), constant_values=0.)[:10]
            self.obs_h.pop(0)
            self.obs_h.append(np.concatenate([np_, lr, [self.cash/max(1e-4, nw), np.clip(dd, -1, 0)]]).astype(np.float32))

        obs = np.concatenate(self.obs_h).astype(np.float32) if not done else np.zeros(660, dtype=np.float32)
        return obs, reward, done, False, {}


# ═══════════════════════════════════════════════
#  RAI NEURAL MODEL
# ═══════════════════════════════════════════════
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


def eval_policy(policy_func, prices):
    T, N = prices.shape
    if T < 35: return {"return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "final": 10000.0}

    cash = 500.0
    shares = (9500.0 / N) / prices[30]
    peak = 10000.0
    eq = [10000.0]

    obs_h = []
    for t in range(30):
        p, pp = prices[t], prices[max(0, t-1)]
        np_ = np.pad(p/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.append(np.concatenate([np_, lr, [0.05, 0.]]).astype(np.float32))

    for t in range(30, T):
        act = policy_func(np.concatenate(obs_h).astype(np.float32))
        cl = np.clip(act[0] - 2.5, -8., 3.)
        tc = 1.0 / (1.0 + np.exp(-cl))
        ts = 1.0 - tc
        n = min(N, 10)
        ea = np.exp(act[1:1+n] - np.max(act[1:1+n]))
        taw = (ea / ea.sum()) * ts

        p = prices[t].copy()
        w = max(1e-4, cash + np.sum(shares * p))
        caw = (shares * p) / w
        ccf = cash / w

        if abs(ccf - tc) + np.sum(np.abs(caw - taw)) > 0.03:
            tv = abs(cash - w * tc) + np.sum(np.abs(shares * p - w * taw))
            net = max(1e-4, w - tv * 0.0005)
            cash = net * tc
            shares = (net * taw) / np.maximum(1e-4, p)

        nw = cash + np.sum(shares * prices[t])
        peak = max(peak, nw)
        dd = (nw - peak) / peak
        eq.append(nw)

        pp = prices[t-1]
        np_ = np.pad(prices[t]/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(prices[t]/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.pop(0)
        obs_h.append(np.concatenate([np_, lr, [cash/max(1e-4, nw), np.clip(dd, -1, 0)]]).astype(np.float32))

    eq_a = np.array(eq)
    r = np.diff(eq_a) / np.maximum(1e-8, eq_a[:-1])
    pk = np.maximum.accumulate(eq_a)
    return {
        "final": float(eq_a[-1]),
        "return_pct": float((eq_a[-1]/eq_a[0]-1)*100),
        "sharpe": float(np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r) > 1e-8 else 0.,
        "max_dd_pct": float(np.min((eq_a-pk)/pk)*100)
    }


def main():
    W = 105
    print("="*W)
    print("  REAL-DATA TRAINED DL MODELS vs. RAI ZERO-SHOT MULTI-DATASET EXPERIMENT")
    print("="*W, flush=True)

    # Load 5 RAI v6 models (Trained 100% on synthetic data)
    rai_v6_models = []
    for s in range(1, 6):
        p = os.path.join(V6_DIR, f"rai_v6_seed_{s:02d}.pt")
        if os.path.exists(p):
            m = DeepEndToEndTradingNet()
            m.load_state_dict(torch.load(p, weights_only=True))
            m.eval()
            rai_v6_models.append(m)

    print(f"  ✓ Loaded {len(rai_v6_models)} RAI v6 models (Trained on 0% Real Data / 100% Synthetic)\n", flush=True)

    benchmark_summary = {}

    for d_name, d_info in DATASETS.items():
        print(f"\n{'═'*W}")
        print(f"  EXPERIMENT DATASET: {d_name}")
        print(f"{'═'*W}")

        try:
            df = yf.download(d_info['tickers'], start=d_info['start'], end=d_info['end'], progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df = df['Close']
            df = df.dropna()
            if df.empty or len(df) < 50:
                print(f"  ⚠ Insufficient data for {d_name}, skipping.")
                continue
            prices = df.values
        except Exception as e:
            print(f"  ⚠ Failed download for {d_name}: {e}")
            continue

        n_total = len(prices)
        n_train = int(n_total * 0.70)
        train_prices = prices[:n_train]
        test_prices = prices[n_train:]

        print(f"  Total Data      : {n_total} trading days ({df.index[0].date()} → {df.index[-1].date()})")
        print(f"  70% Real Train  : {n_train} trading days ({df.index[0].date()} → {df.index[n_train-1].date()})")
        print(f"  30% Real OOS Test: {len(test_prices)} trading days ({df.index[n_train].date()} → {df.index[-1].date()})\n")

        # ── 1. Train Real-Data PPO Model on 70% Real Train Split ──
        print(f"  [1/2] Training Deep Learning PPO Model directly on 70% Real Train Data ({n_train} days)...", flush=True)
        env_train = RealMarketTradingEnv(train_prices)
        real_ppo_model = PPO("MlpPolicy", env_train, verbose=0, n_steps=256, batch_size=64, learning_rate=0.0003, seed=42)
        real_ppo_model.learn(total_timesteps=30000)
        print("  ✓ Real-Data PPO Training Complete!\n", flush=True)

        def real_ppo_policy(obs):
            act, _ = real_ppo_model.predict(obs, deterministic=True)
            return act

        # Evaluate Real-Data Trained Model on OOS Test Set
        real_train_oos_res = eval_policy(real_ppo_policy, test_prices)

        # ── 2. Evaluate RAI v6 (Trained on 0 Real Data) Zero-Shot on exact same OOS Test Set ──
        print(f"  [2/2] Evaluating RAI v6 (Trained on 0% Real Data) Zero-Shot on exact same OOS Test Set...", flush=True)
        rai_res_list = [eval_policy(m.get_action, test_prices) for m in rai_v6_models]
        rai_rets = [r['return_pct'] for r in rai_res_list]
        rai_shs = [r['sharpe'] for r in rai_res_list]
        rai_dds = [r['max_dd_pct'] for r in rai_res_list]

        # Equal Weight Baseline on OOS
        def ew_policy(obs):
            return np.zeros(11, dtype=np.float32)
        ew_res = eval_policy(ew_policy, test_prices)

        print(f"\n  {'Model Variant':<40} | {'Real Data Trained?':<20} | {'OOS Return (%)':<16} | {'Sharpe':<8} | {'Max DD (%)':<12}")
        print(f"  {'-'*104}")
        print(f"  {'Equal Weight (1/N Baseline)':<40} | {'No (Rule-Based)':<20} | {ew_res['return_pct']:>+8.2f}%{'':<7} | {ew_res['sharpe']:>7.2f} | {ew_res['max_dd_pct']:>+9.2f}%")
        print(f"  {'Real-Data Trained PPO (70% Real Train)':<40} | {'YES (Trained 70%)':<20} | {real_train_oos_res['return_pct']:>+8.2f}%{'':<7} | {real_train_oos_res['sharpe']:>7.2f} | {real_train_oos_res['max_dd_pct']:>+9.2f}%")
        print(f"  {'RAI v6 (0% Real Data / 100% Synthetic)':<40} | {'NO (0% Real Data)':<20} | {np.mean(rai_rets):>+8.2f}±{np.std(rai_rets):<4.2f}%{'':<1} | {np.mean(rai_shs):>7.2f} | {np.mean(rai_dds):>+9.2f}%")

        diff_ret = np.mean(rai_rets) - real_train_oos_res['return_pct']
        print(f"\n  ► ZERO-SHOT RAI v6 VS REAL-TRAINED PPO RETURN DIFFERENCE: {diff_ret:>+6.2f}%")

        benchmark_summary[d_name] = {
            "real_trained_ppo": real_train_oos_res,
            "rai_v6_zero_shot": {"mean_return": float(np.mean(rai_rets)), "mean_sharpe": float(np.mean(rai_shs)), "mean_max_dd": float(np.mean(rai_dds))},
            "equal_weight": ew_res
        }

    # Save Output
    out_file = os.path.join(RESULTS_DIR, "real_train_vs_rai_benchmark.json")
    with open(out_file, 'w') as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\n{'═'*W}")
    print(f"  ✅ BENCHMARK EXPERIMENT COMPLETE")
    print(f"  Results saved to: {out_file}")
    print(f"{'═'*W}\n", flush=True)

if __name__ == "__main__":
    main()
