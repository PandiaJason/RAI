"""
═══════════════════════════════════════════════════════════════════════════════
  DEEP LEARNING MODEL ZOO MULTI-DATASET BENCHMARK
  ════════════════════════════════════════════════
  Evaluates and compares a rich ZOO of trained deep learning models:

    1. RAI v6 Hybrid Conv1D+Transformer (5 seeds - Synthetic Control)
    2. RAI v7 Macro Scenario Policy (5 seeds - Causal Factor Engine)
    3. RAI v5 Dual-Head Architecture (v0.5 checkpoint)
    4. RAI v1 Early Frozen Model (v1.0 baseline)
    5. Real-Data Trained PPO Agent (100k steps trained directly on real data)
    6. Equal Weight Benchmark (1/N)

  Across 4 Real Asset Classes:
    • Crypto Assets (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, LTC)
    • Global Equity Indices (SPY, EWJ, EWG, EWU, MCHI, INDA, EWZ, EFA, EEM, FXI)
    • US Mega-Cap Stocks (AAPL, MSFT, NVDA, GOOGL, AMZN, META, LLY, JPM, JNJ, WMT)
    • US Sector ETFs (XLK, XLV, XLF, XLE, XLI, XLP, XLU, XLY, XLB, XLC)
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from stable_baselines3 import PPO

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6_DIR = os.path.join(PROJECT_ROOT, "data", "robustness", "seeds")
V7_DIR = os.path.join(PROJECT_ROOT, "data", "v7_scenarios", "models")
V5_PATH = os.path.join(PROJECT_ROOT, "data", "v0.5_rl_checkpoints", "rai_v5_dual_head.pt")
V1_PATH = os.path.join(PROJECT_ROOT, "v1.0_FROZEN", "rai_v1_model.pt")
REAL_PPO_PATH = os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "rai_real_ppo_100000_steps.zip")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "dl_zoo_multi_dataset")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    "1. Crypto Assets": {
        "tickers": ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "SOL-USD", "AVAX-USD", "LINK-USD", "LTC-USD"],
        "start": "2020-01-01", "end": "2026-08-08"
    },
    "2. Global Equity Indices": {
        "tickers": ["SPY", "EWJ", "EWG", "EWU", "MCHI", "INDA", "EWZ", "EFA", "EEM", "FXI"],
        "start": "2015-01-01", "end": "2026-08-08"
    },
    "3. US Mega-Cap Stocks": {
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "LLY", "JPM", "JNJ", "WMT"],
        "start": "2015-01-01", "end": "2026-08-08"
    },
    "4. US Sector ETFs": {
        "tickers": ["XLK", "XLV", "XLF", "XLE", "XLI", "XLP", "XLU", "XLY", "XLB", "XLC"],
        "start": "2015-01-01", "end": "2026-08-08"
    }
}


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


def eval_model(model, prices, fee_bps=5, slippage_pct=0.02, is_sb3=False):
    T, N = prices.shape
    if T < 35: return {"return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "final": 10000.0}

    cash = 500.0
    shares = (9500.0 / N) / prices[30]
    peak = 10000.0
    eq = [10000.0]
    rng = np.random.RandomState(42)

    obs_h = []
    for t in range(30):
        p, pp = prices[t], prices[max(0, t-1)]
        np_ = np.pad(p/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.append(np.concatenate([np_, lr, [0.05, 0.]]).astype(np.float32))

    for t in range(30, T):
        flat_obs = np.concatenate(obs_h).astype(np.float32)

        if is_sb3:
            act, _ = model.predict(flat_obs, deterministic=True)
        else:
            act = model.get_action(flat_obs)

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
            p_ex = p * (1.0 + rng.uniform(-slippage_pct/100., slippage_pct/100., N))
            tv = abs(cash - w * tc) + np.sum(np.abs(shares * p - w * taw))
            fee = fee_bps / 10000.0
            net = max(1e-4, w - tv * fee)
            cash = net * tc
            shares = (net * taw) / np.maximum(1e-4, p_ex)

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


def compute_equal_weight(prices):
    T, N = prices.shape
    sh = (10000.0 / N) / prices[30]
    eq = []
    for t in range(30, T):
        w = np.sum(sh * prices[t])
        if (t - 30) % 21 == 0 and t > 30: sh = (w / N) / prices[t]
        eq.append(w)
    eq_a = np.array(eq)
    r = np.diff(eq_a) / np.maximum(1e-8, eq_a[:-1])
    pk = np.maximum.accumulate(eq_a)
    return {
        "final": float(eq_a[-1]), "return_pct": float((eq_a[-1]/10000-1)*100),
        "sharpe": float(np.mean(r)/np.std(r)*np.sqrt(252)),
        "max_dd_pct": float(np.min((eq_a-pk)/pk)*100)
    }


def main():
    W = 105
    print("="*W)
    print("  DEEP LEARNING MODEL ZOO MULTI-DATASET BENCHMARK")
    print("="*W, flush=True)

    # Load Model Zoo
    zoo = {}

    # 1. RAI v6 (5 seeds)
    v6_models = []
    for s in range(1, 6):
        p = os.path.join(V6_DIR, f"rai_v6_seed_{s:02d}.pt")
        if os.path.exists(p):
            m = DeepEndToEndTradingNet()
            m.load_state_dict(torch.load(p, weights_only=True))
            m.eval()
            v6_models.append(m)
    zoo["RAI v6 Conv1D+Transformer (5-Seed Mean)"] = (v6_models, False)

    # 2. RAI v7 (5 seeds)
    v7_models = []
    for s in range(1, 6):
        p = os.path.join(V7_DIR, f"rai_v7_scenario_seed_{s:02d}.pt")
        if os.path.exists(p):
            m = DeepEndToEndTradingNet()
            m.load_state_dict(torch.load(p, weights_only=True))
            m.eval()
            v7_models.append(m)
    zoo["RAI v7 Scenario Engine (5-Seed Mean)"] = (v7_models, False)

    # 3. RAI v5 Dual Head
    if os.path.exists(V5_PATH):
        try:
            m5 = DeepEndToEndTradingNet()
            m5.load_state_dict(torch.load(V5_PATH, weights_only=True))
            m5.eval()
            zoo["RAI v5 Dual-Head Model"] = ([m5], False)
        except Exception as e:
            print(f"  ⚠ Could not load v5: {e}")

    # 4. RAI v1 Frozen Baseline
    if os.path.exists(V1_PATH):
        try:
            m1 = DeepEndToEndTradingNet()
            m1.load_state_dict(torch.load(V1_PATH, weights_only=True))
            m1.eval()
            zoo["RAI v1 Frozen Baseline"] = ([m1], False)
        except Exception as e:
            print(f"  ⚠ Could not load v1: {e}")

    # 5. Real-Data Trained PPO Model (Stable Baselines 3)
    if os.path.exists(REAL_PPO_PATH):
        try:
            sb3_m = PPO.load(REAL_PPO_PATH)
            zoo["Real-Data Trained PPO Agent (100k steps)"] = ([sb3_m], True)
        except Exception as e:
            print(f"  ⚠ Could not load real PPO model: {e}")

    for name, (models, is_sb) in zoo.items():
        print(f"  ✓ Loaded '{name}' ({len(models)} model(s))")

    print("\n", flush=True)

    master_results = {}

    for d_name, d_info in DATASETS.items():
        print(f"\n{'═'*W}")
        print(f"  EVALUATION DATASET: {d_name}")
        print(f"{'═'*W}")

        try:
            df = yf.download(d_info['tickers'], start=d_info['start'], end=d_info['end'], progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df = df['Close']
            df = df.dropna()
            prices = df.values
            print(f"  Data: {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()}) across {df.shape[1]} assets\n")
        except Exception as e:
            print(f"  ⚠ Download failed for {d_name}: {e}")
            continue

        print(f"  {'Model / Strategy Variant':<42} | {'Return (%)':<18} | {'Sharpe':<10} | {'Max DD (%)':<14} | {'Final Capital':<12}")
        print(f"  {'-'*104}")

        # Baseline
        ew_res = compute_equal_weight(prices)
        print(f"  {'Equal Weight (1/N Benchmark)':<42} | {ew_res['return_pct']:>+8.2f}%{'':<9} | {ew_res['sharpe']:>7.2f}{'':<3} | {ew_res['max_dd_pct']:>+9.2f}%{'':<4} | ${ew_res['final']:>10,.2f}")
        print(f"  {'-'*104}")

        d_res = {"equal_weight": ew_res, "models": {}}

        for model_name, (models, is_sb) in zoo.items():
            if not models: continue
            res_list = [eval_model(m, prices, is_sb3=is_sb) for m in models]
            rets = [r['return_pct'] for r in res_list]
            shs = [r['sharpe'] for r in res_list]
            dds = [r['max_dd_pct'] for r in res_list]

            mean_ret, std_ret = np.mean(rets), np.std(rets)
            mean_sh, std_sh = np.mean(shs), np.std(shs)
            mean_dd, std_dd = np.mean(dds), np.std(dds)
            mean_fin = np.mean([r['final'] for r in res_list])

            std_str = f"±{std_ret:.1f}%" if len(models) > 1 else ""
            print(f"  {model_name:<42} | {mean_ret:>+8.2f}{std_str:<9} | {mean_sh:>7.2f}{'':<3} | {mean_dd:>+9.2f}%{'':<4} | ${mean_fin:>10,.2f}")

            d_res["models"][model_name] = {
                "mean_return": float(mean_ret), "mean_sharpe": float(mean_sh), "mean_max_dd": float(mean_dd)
            }

        master_results[d_name] = d_res

    # Save Output JSON
    out_file = os.path.join(RESULTS_DIR, "dl_zoo_multi_dataset_results.json")
    with open(out_file, 'w') as f:
        json.dump(master_results, f, indent=2)

    print(f"\n{'═'*W}")
    print(f"  ✅ DEEP LEARNING MODEL ZOO MULTI-DATASET EVALUATION COMPLETE")
    print(f"  Results saved to: {out_file}")
    print(f"{'═'*W}\n", flush=True)

if __name__ == "__main__":
    main()
