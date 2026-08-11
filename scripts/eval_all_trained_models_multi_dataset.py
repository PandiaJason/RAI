"""
═══════════════════════════════════════════════════════════════════════════════
  MASTER TRAINED MODELS MULTI-DATASET COMPARISON PROTOCOL
  ════════════════════════════════════════════════════════
  Evaluates and compares MULTIPLE TRAINED MODEL VARIANTS across 4 datasets:

    1. Crypto Assets (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, LTC)
    2. Global Equity Indices (SPY, EWJ, EWG, EWU, MCHI, INDA, EWZ, EFA, EEM, FXI)
    3. US Mega-Cap Stocks (AAPL, MSFT, NVDA, GOOGL, AMZN, META, LLY, JPM, JNJ, WMT)
    4. US Sector ETFs (XLK, XLV, XLF, XLE, XLI, XLP, XLU, XLY, XLB, XLC)

  Trained Models Compared:
    • RAI v6 Control (5 seeds) - Trained on G0 Random Walks
    • RAI v7 Scenarios (5 seeds) - Trained on Factor-Driven Macro Scenarios
    • RAI G0 Ladder (5 seeds) - Pure Random Walk Ablation
    • RAI G6 Ladder (5 seeds) - Full Realistic Stochastic Market Simulator
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6_DIR = os.path.join(PROJECT_ROOT, "data", "robustness", "seeds")
V7_DIR = os.path.join(PROJECT_ROOT, "data", "v7_scenarios", "models")
LADDER_DIR = os.path.join(PROJECT_ROOT, "data", "ablation_ladder", "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "multi_model_multi_dataset")
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


def eval_model(model, prices, fee_bps=5, slippage_pct=0.02):
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
        act = model.get_action(np.concatenate(obs_h).astype(np.float32))
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


def load_model_group(model_dir, prefix, seeds=5):
    models = []
    for s in range(1, seeds + 1):
        path = os.path.join(model_dir, f"{prefix}_{s:02d}.pt")
        if os.path.exists(path):
            m = DeepEndToEndTradingNet()
            m.load_state_dict(torch.load(path, weights_only=True))
            m.eval()
            models.append(m)
    return models


def main():
    W = 105
    print("="*W)
    print("  MASTER TRAINED MODELS MULTI-DATASET COMPARISON PROTOCOL")
    print("="*W, flush=True)

    # Load Model Ensembles
    model_groups = {
        "RAI v6 Control (Random Walk)": load_model_group(V6_DIR, "rai_v6_seed", 5),
        "RAI v7 Scenarios (Macro Engine)": load_model_group(V7_DIR, "rai_v7_scenario_seed", 5),
        "RAI G0 Uncorrelated GBM (5 seeds)": load_model_group(LADDER_DIR, "Level_0_GBM_seed", 5),
        "RAI G6 Full Simulator (5 seeds)": load_model_group(LADDER_DIR, "Level_6_CombinedRealistic_seed", 5)
    }

    for name, group in model_groups.items():
        print(f"  ✓ Loaded {len(group)} models for '{name}'")

    print("\n", flush=True)

    master_results = {}

    for d_name, d_info in DATASETS.items():
        print(f"\n{'═'*W}")
        print(f"  DATASET EVALUATION: {d_name}")
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

        print(f"  {'Model / Strategy Variant':<36} | {'Return (%)':<20} | {'Sharpe':<12} | {'Max DD (%)':<14} | {'Final Capital':<12}")
        print(f"  {'-'*100}")

        # Baseline
        ew_res = compute_equal_weight(prices)
        print(f"  {'Equal Weight (1/N Benchmark)':<36} | {ew_res['return_pct']:>+8.2f}%{'':<11} | {ew_res['sharpe']:>7.2f}{'':<5} | {ew_res['max_dd_pct']:>+9.2f}%{'':<4} | ${ew_res['final']:>10,.2f}")
        print(f"  {'-'*100}")

        d_res = {"equal_weight": ew_res, "models": {}}

        for g_name, group in model_groups.items():
            if not group: continue
            res_list = [eval_model(m, prices) for m in group]
            rets = [r['return_pct'] for r in res_list]
            shs = [r['sharpe'] for r in res_list]
            dds = [r['max_dd_pct'] for r in res_list]
            best_idx = int(np.argmax(shs))

            mean_ret, std_ret = np.mean(rets), np.std(rets)
            mean_sh, std_sh = np.mean(shs), np.std(shs)
            mean_dd, std_dd = np.mean(dds), np.std(dds)
            mean_fin = np.mean([r['final'] for r in res_list])

            print(f"  {g_name:<36} | {mean_ret:>+8.2f}±{std_ret:<4.2f}%{'':<5} | {mean_sh:>7.2f}±{std_sh:<4.2f} | {mean_dd:>+9.2f}±{std_dd:<4.2f}% | ${mean_fin:>10,.2f}")
            print(f"    └ Best Seed ({best_idx+1}){'':<19} | {rets[best_idx]:>+8.2f}%{'':<11} | {shs[best_idx]:>7.2f}{'':<5} | {dds[best_idx]:>+9.2f}%{'':<4} | ${res_list[best_idx]['final']:>10,.2f}")

            d_res["models"][g_name] = {
                "mean_return": float(mean_ret), "std_return": float(std_ret),
                "mean_sharpe": float(mean_sh), "std_sharpe": float(std_sh),
                "mean_max_dd": float(mean_dd), "std_max_dd": float(std_dd),
                "best_seed_return": float(rets[best_idx]), "best_seed_sharpe": float(shs[best_idx]),
                "best_seed_max_dd": float(dds[best_idx])
            }

        master_results[d_name] = d_res

    # Save Output JSON
    out_file = os.path.join(RESULTS_DIR, "all_trained_models_multi_dataset.json")
    with open(out_file, 'w') as f:
        json.dump(master_results, f, indent=2)

    print(f"\n{'═'*W}")
    print(f"  ✅ MASTER TRAINED MODELS MULTI-DATASET EVALUATION COMPLETE")
    print(f"  Results saved to: {out_file}")
    print(f"{'═'*W}\n", flush=True)

if __name__ == "__main__":
    main()
