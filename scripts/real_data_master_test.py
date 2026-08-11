"""
═══════════════════════════════════════════════════════════════════════════════
  MASTER REAL MARKET TEST PROTOCOL: RAI v6 vs RAI v7 ACROSS 4 REAL PERIODS
  ════════════════════════════════════════════════════════════════════════
  Evaluates all 10 trained models (5 seeds of RAI v6, 5 seeds of RAI v7)
  across 4 distinct real market periods spanning 2007 to 2026:

    1. 2007–2019: Historical Crisis Period (GFC 2008, 2011, 2018)
    2. 2020–2024: Primary Out-of-Sample Period (COVID, Inflation, Rate Hikes)
    3. 2024–2026: Untouched Holdout Period (Recent Market Regimes)
    4. 2007–2026: Multi-Decade Full Historical Period (19 Years)

  Baselines:
    - Buy & Hold SPY
    - Equal Weight (1/N)
    - Risk Parity
    - 60/40 Equity/Bond
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "real_master_test")
os.makedirs(RESULTS_DIR, exist_ok=True)

TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]

# ═══════════════════════════════════════════════
#  MODEL ARCHITECTURE
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


# ═══════════════════════════════════════════════
#  EVALUATION ENGINE
# ═══════════════════════════════════════════════
def run_model_eval(model, prices, fee_bps=5, slippage_pct=0.02):
    T, N = prices.shape
    if T < 35:
        return {"return_pct": 0, "sharpe": 0, "max_dd_pct": 0, "final": 10000.0, "trades": 0}

    cash = 500.0
    shares = (9500.0 / N) / prices[30]
    peak = 10000.0
    eq = [10000.0]
    trades_count = 0
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
            trades_count += 1
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
        "vol_pct": float(np.std(r)*np.sqrt(252)*100),
        "sharpe": float(np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r) > 1e-8 else 0.,
        "max_dd_pct": float(np.min((eq_a-pk)/pk)*100),
        "trades": trades_count
    }


def compute_baselines(prices):
    T, N = prices.shape
    baselines = {}

    # Buy & Hold SPY (index 5)
    spy_eq = (10000.0 * prices[30:, 5] / prices[30, 5]).tolist()
    r_spy = np.diff(spy_eq) / np.maximum(1e-8, spy_eq[:-1])
    pk_spy = np.maximum.accumulate(spy_eq)
    baselines['Buy & Hold SPY'] = {
        "final": float(spy_eq[-1]), "return_pct": float((spy_eq[-1]/10000-1)*100),
        "sharpe": float(np.mean(r_spy)/np.std(r_spy)*np.sqrt(252)),
        "max_dd_pct": float(np.min((spy_eq-pk_spy)/pk_spy)*100)
    }

    # Equal Weight (1/N)
    sh = (10000.0 / N) / prices[30]
    eq = []
    for t in range(30, T):
        w = np.sum(sh * prices[t])
        if (t - 30) % 21 == 0 and t > 30: sh = (w / N) / prices[t]
        eq.append(w)
    r_eq = np.diff(eq) / np.maximum(1e-8, eq[:-1])
    pk_eq = np.maximum.accumulate(eq)
    baselines['Equal Weight'] = {
        "final": float(eq[-1]), "return_pct": float((eq[-1]/10000-1)*100),
        "sharpe": float(np.mean(r_eq)/np.std(r_eq)*np.sqrt(252)),
        "max_dd_pct": float(np.min((eq-pk_eq)/pk_eq)*100)
    }

    # 60/40 Equity/Bond
    sh_6040 = np.zeros(N)
    sh_6040[5] = (6000.0) / prices[30, 5]  # SPY
    sh_6040[6] = (4000.0) / prices[30, 6]  # TLT
    eq_6040 = []
    for t in range(30, T):
        w = np.sum(sh_6040 * prices[t])
        if (t - 30) % 21 == 0 and t > 30:
            sh_6040[5] = (w * 0.60) / prices[t, 5]
            sh_6040[6] = (w * 0.40) / prices[t, 6]
        eq_6040.append(w)
    r_6040 = np.diff(eq_6040) / np.maximum(1e-8, eq_6040[:-1])
    pk_6040 = np.maximum.accumulate(eq_6040)
    baselines['Fixed 60/40'] = {
        "final": float(eq_6040[-1]), "return_pct": float((eq_6040[-1]/10000-1)*100),
        "sharpe": float(np.mean(r_6040)/np.std(r_6040)*np.sqrt(252)),
        "max_dd_pct": float(np.min((eq_6040-pk_6040)/pk_6040)*100)
    }

    return baselines


def main():
    W = 105
    print("="*W)
    print("  MASTER REAL MARKET TEST PROTOCOL: RAI v6 vs RAI v7 ACROSS 4 REAL PERIODS")
    print("="*W, flush=True)

    # 1. Load Trained Models
    v6_models, v7_models = [], []
    for seed in range(1, 6):
        p6 = os.path.join(V6_DIR, f"rai_v6_seed_{seed:02d}.pt")
        if os.path.exists(p6):
            m6 = DeepEndToEndTradingNet()
            m6.load_state_dict(torch.load(p6, weights_only=True))
            m6.eval()
            v6_models.append(m6)

        p7 = os.path.join(V7_DIR, f"rai_v7_scenario_seed_{seed:02d}.pt")
        if os.path.exists(p7):
            m7 = DeepEndToEndTradingNet()
            m7.load_state_dict(torch.load(p7, weights_only=True))
            m7.eval()
            v7_models.append(m7)

    print(f"  ✓ Loaded {len(v6_models)} RAI v6 models and {len(v7_models)} RAI v7 models\n", flush=True)

    # 2. Download Real Data for 4 Historical Periods
    period_specs = {
        "Period 1: 2007–2019 (Historical Crisis Period)": ("2007-01-01", "2019-12-31"),
        "Period 2: 2020–2024 (Out-of-Sample Period)": ("2020-01-01", "2023-12-29"),
        "Period 3: 2024–2026 (Untouched Holdout Period)": ("2024-06-01", "2026-08-08"),
        "Period 4: 2007–2026 (Multi-Decade Full Period)": ("2007-01-01", "2026-08-08"),
    }

    real_data = {}
    print("  Downloading real market dataset across 4 timeframes...", flush=True)
    for p_name, (start_d, end_d) in period_specs.items():
        try:
            df = yf.download(TICKERS, start=start_d, end=end_d, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df = df['Close']
            df = df[TICKERS].dropna()
            if len(df) >= 35:
                real_data[p_name] = df.values
                print(f"    ✓ {p_name:<48}: {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})", flush=True)
        except Exception as e:
            print(f"    ⚠ Could not fetch {p_name}: {e}", flush=True)

    # 3. Execute Master Test
    master_results = {}

    for p_name, prices in real_data.items():
        print(f"\n{'═'*W}")
        print(f"  EVALUATION RESULTS: {p_name}")
        print(f"{'═'*W}")
        print(f"  {'Strategy / Variant':<34} | {'Return (%)':<20} | {'Sharpe':<12} | {'Max DD (%)':<14} | {'Final Capital':<12}")
        print(f"  {'-'*100}")

        # Baselines
        base_res = compute_baselines(prices)
        for b_name, b_m in base_res.items():
            print(f"  {b_name:<34} | {b_m['return_pct']:>+8.2f}%{'':<11} | {b_m['sharpe']:>7.2f}{'':<5} | {b_m['max_dd_pct']:>+9.2f}%{'':<4} | ${b_m['final']:>10,.2f}")

        print(f"  {'-'*100}")

        # RAI v6 Evaluation
        v6_res = [run_model_eval(m, prices) for m in v6_models]
        v6_rets = [r['return_pct'] for r in v6_res]
        v6_shs = [r['sharpe'] for r in v6_res]
        v6_dds = [r['max_dd_pct'] for r in v6_res]
        v6_best_idx = int(np.argmax(v6_shs))

        print(f"  {'RAI v6 (Ensemble 5-Seed Mean)':<34} | {np.mean(v6_rets):>+8.2f}±{np.std(v6_rets):<4.2f}%{'':<5} | {np.mean(v6_shs):>7.2f}±{np.std(v6_shs):<4.2f} | {np.mean(v6_dds):>+9.2f}±{np.std(v6_dds):<4.2f}% | ${np.mean([r['final'] for r in v6_res]):>10,.2f}")
        print(f"  {'RAI v6 (Best Seed)':<34} | {v6_rets[v6_best_idx]:>+8.2f}%{'':<11} | {v6_shs[v6_best_idx]:>7.2f}{'':<5} | {v6_dds[v6_best_idx]:>+9.2f}%{'':<4} | ${v6_res[v6_best_idx]['final']:>10,.2f}")

        # RAI v7 Evaluation
        v7_res = [run_model_eval(m, prices) for m in v7_models]
        v7_rets = [r['return_pct'] for r in v7_res]
        v7_shs = [r['sharpe'] for r in v7_res]
        v7_dds = [r['max_dd_pct'] for r in v7_res]
        v7_best_idx = int(np.argmax(v7_shs))

        # Welch's t-test vs v6
        t_stat, p_val = stats.ttest_ind(v7_rets, v6_rets, equal_var=False)
        sig_str = f" (vs v6: p={p_val:.3f} ns)"

        print(f"  {'RAI v7 (Scenario Diversity Mean)':<34} | {np.mean(v7_rets):>+8.2f}±{np.std(v7_rets):<4.2f}%{'':<5} | {np.mean(v7_shs):>7.2f}±{np.std(v7_shs):<4.2f} | {np.mean(v7_dds):>+9.2f}±{np.std(v7_dds):<4.2f}% | ${np.mean([r['final'] for r in v7_res]):>10,.2f}")
        print(f"  {'RAI v7 (Best Seed)':<34} | {v7_rets[v7_best_idx]:>+8.2f}%{'':<11} | {v7_shs[v7_best_idx]:>7.2f}{'':<5} | {v7_dds[v7_best_idx]:>+9.2f}%{'':<4} | ${v7_res[v7_best_idx]['final']:>10,.2f}")

        master_results[p_name] = {
            "baselines": base_res,
            "v6_mean": {"return": float(np.mean(v6_rets)), "sharpe": float(np.mean(v6_shs)), "max_dd": float(np.mean(v6_dds))},
            "v7_mean": {"return": float(np.mean(v7_rets)), "sharpe": float(np.mean(v7_shs)), "max_dd": float(np.mean(v7_dds))},
            "welch_p_val": float(p_val)
        }

    # Save output JSON
    out_file = os.path.join(RESULTS_DIR, "master_real_test_results.json")
    with open(out_file, 'w') as f:
        json.dump(master_results, f, indent=2)

    print(f"\n{'═'*W}")
    print(f"  ✅ MASTER REAL MARKET TEST COMPLETE")
    print(f"  Results saved to: {out_file}")
    print(f"{'═'*W}\n", flush=True)

if __name__ == "__main__":
    main()
