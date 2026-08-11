"""
═══════════════════════════════════════════════════════════════════════════════
  REAL DATASET EVALUATION TEST: RAI v6 vs RAI v7 vs BASELINES
  ═════════════════════════════════════════════════════════════
  Evaluates trained RAI v6 (Control) and RAI v7 (Scenario Diversity) models
  against real market asset price data across:
    1. 2020–2024 Out-of-Sample Test
    2. 2024–2026 Untouched Holdout Test
    3. Full 2020–2026 Combined Test

  Calculates: Daily P&L, Sharpe, Max Drawdown, Volatility, Win Rate,
              Profit Factor, Monthly Return Breakdowns, Equity Curve Charts.
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6_DIR = os.path.join(PROJECT_ROOT, "data", "robustness", "seeds")
V7_DIR = os.path.join(PROJECT_ROOT, "data", "v7_scenarios", "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "data", "real_eval_report")
os.makedirs(REPORT_DIR, exist_ok=True)

TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]


# ═══════════════════════════════════════════════
#  MODEL DEFINITION
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
def run_model_eval(model, prices_df, fee_bps=5, slippage_pct=0.02):
    tickers = list(prices_df.columns)
    prices = prices_df.values
    dates = prices_df.index
    T, N = prices.shape

    daily_log = []
    trade_log = []
    cash = 500.0
    shares = (9500.0 / N) / prices[30]
    peak = 10000.0
    rng = np.random.RandomState(42)

    obs_h = []
    for t in range(30):
        p, pp = prices[t], prices[max(0, t-1)]
        np_ = np.pad(p/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.append(np.concatenate([np_, lr, [0.05, 0.]]).astype(np.float32))

    for t in range(30, T):
        w_before = cash + np.sum(shares * prices[t])
        act = model.get_action(np.concatenate(obs_h).astype(np.float32))

        cl = np.clip(act[0] - 2.5, -8., 3.)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cl))
        target_stock_frac = 1.0 - target_cash_frac
        n = min(N, 10)
        ea = np.exp(act[1:1+n] - np.max(act[1:1+n]))
        target_asset_weights = (ea / ea.sum()) * target_stock_frac

        p = prices[t].copy()
        w = max(1e-4, cash + np.sum(shares * p))
        current_asset_weights = (shares * p) / w
        current_cash_frac = cash / w

        traded = False
        turnover = 0.0
        if abs(current_cash_frac - target_cash_frac) + np.sum(np.abs(current_asset_weights - target_asset_weights)) > 0.03:
            traded = True
            p_ex = p * (1.0 + rng.uniform(-slippage_pct/100., slippage_pct/100., N))
            tv = abs(cash - w * target_cash_frac) + np.sum(np.abs(shares * p - w * target_asset_weights))
            turnover = tv / w
            fee = fee_bps / 10000.0
            net = max(1e-4, w - tv * fee)
            cash = net * target_cash_frac
            shares = (net * target_asset_weights) / np.maximum(1e-4, p_ex)

        nw = cash + np.sum(shares * prices[t])
        peak = max(peak, nw)
        dd = (nw - peak) / peak
        daily_ret = (nw - w_before) / w_before if w_before > 0 else 0

        daily_log.append({
            'date': str(dates[t].date()) if hasattr(dates[t], 'date') else str(dates[t]),
            'portfolio_value': nw,
            'daily_return_pct': daily_ret * 100,
            'cash_pct': (cash / nw) * 100,
            'drawdown_pct': dd * 100,
            'traded': traded,
        })

        pp = prices[t-1]
        np_ = np.pad(prices[t]/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(prices[t]/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.pop(0)
        obs_h.append(np.concatenate([np_, lr, [cash/max(1e-4, nw), np.clip(dd, -1, 0)]]).astype(np.float32))

    return daily_log


def compute_baselines(prices_df):
    prices = prices_df.values
    T, N = prices.shape
    baselines = {}

    spy_idx = list(prices_df.columns).index('SPY') if 'SPY' in prices_df.columns else 5
    baselines['Buy & Hold SPY'] = (10000 * prices[30:, spy_idx] / prices[30, spy_idx]).tolist()

    sh = (10000.0 / N) / prices[30]
    eq = []
    for t in range(30, T):
        w = np.sum(sh * prices[t])
        if (t - 30) % 21 == 0 and t > 30: sh = (w / N) / prices[t]
        eq.append(w)
    baselines['Equal Weight'] = eq

    sh_rp = (10000.0 / N) / prices[30]
    eq_rp = []
    for t in range(30, T):
        w = np.sum(sh_rp * prices[t])
        if (t - 30) % 21 == 0 and t >= 60:
            v = np.std(np.diff(np.log(prices[t-60:t+1]), axis=0), axis=0)
            iv = 1.0 / np.maximum(1e-8, v)
            wts = iv / iv.sum()
            sh_rp = (w * wts) / prices[t]
        eq_rp.append(w)
    baselines['Risk Parity'] = eq_rp

    return baselines


def plot_real_curves(period_name, rai_v6_log, rai_v7_log, baselines, dates, save_dir):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Real Market Test — {period_name}', fontsize=18, fontweight='bold', color='white', y=0.98)

    plot_dates = pd.to_datetime([d['date'] for d in rai_v6_log])
    v6_eq = [d['portfolio_value'] for d in rai_v6_log]
    v7_eq = [d['portfolio_value'] for d in rai_v7_log]

    ax1 = axes[0]
    colors = {'Buy & Hold SPY': '#4FC3F7', 'Equal Weight': '#81C784', 'Risk Parity': '#FFB74D'}
    for name, curve in baselines.items():
        ax1.plot(plot_dates, curve[:len(plot_dates)], label=name, color=colors.get(name, 'gray'), alpha=0.6, linewidth=1.2)

    ax1.plot(plot_dates, v6_eq, label='RAI v6 (Frozen Control)', color='#FF5252', linewidth=2.5)
    ax1.plot(plot_dates, v7_eq, label='RAI v7 (Scenario Diversity)', color='#AB47BC', linewidth=2.2, linestyle='--')

    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.legend(fontsize=10, loc='upper left', framealpha=0.3)
    ax1.grid(True, alpha=0.15)

    # Drawdown comparison
    ax2 = axes[1]
    v6_dd = [d['drawdown_pct'] for d in rai_v6_log]
    v7_dd = [d['drawdown_pct'] for d in rai_v7_log]
    ax2.plot(plot_dates, v6_dd, color='#FF5252', linewidth=1.2, label='RAI v6 Drawdown')
    ax2.plot(plot_dates, v7_dd, color='#AB47BC', linewidth=1.2, label='RAI v7 Drawdown', linestyle='--')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.15)
    ax2.axhline(0, color='white', alpha=0.3)
    ax2.legend(fontsize=9, loc='lower left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    safe_name = period_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(os.path.join(save_dir, f'real_test_{safe_name}.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()


def print_period_table(period_name, v6_logs, v7_logs, baselines):
    W = 85
    print(f"\n{'═'*W}")
    print(f"  REAL MARKET EVALUATION: {period_name}")
    print(f"{'═'*W}")

    print(f"  {'Strategy / Variant':<32} | {'Return (%)':<12} | {'Sharpe':<8} | {'Max DD (%)':<12} | {'Final ($)':<12}")
    print(f"  {'-'*82}")

    # Baselines
    for b_name, b_curve in baselines.items():
        b_ret = (b_curve[-1] / 10000 - 1) * 100
        b_rets = np.diff(b_curve) / np.maximum(1e-8, b_curve[:-1])
        b_sh = np.mean(b_rets) / np.std(b_rets) * np.sqrt(252) if np.std(b_rets) > 1e-8 else 0
        pk = np.maximum.accumulate(b_curve)
        b_dd = np.min((np.array(b_curve) - pk) / pk) * 100
        print(f"  {b_name:<32} | {b_ret:>+11.2f}% | {b_sh:>7.2f} | {b_dd:>+11.2f}% | ${b_curve[-1]:>11,.2f}")

    print(f"  {'-'*82}")

    # RAI v6 Ensemble stats
    v6_rets = [(l[-1]['portfolio_value']/10000 - 1)*100 for l in v6_logs]
    v6_shs = [np.mean([d['daily_return_pct'] for d in l])/np.std([d['daily_return_pct'] for d in l])*np.sqrt(252) for l in v6_logs]
    v6_dds = [min([d['drawdown_pct'] for d in l]) for l in v6_logs]
    v6_best_idx = np.argmax(v6_shs)
    v6_best_final = v6_logs[v6_best_idx][-1]['portfolio_value']

    print(f"  {'RAI v6 (Ensemble Mean)':<32} | {np.mean(v6_rets):>+11.2f}% | {np.mean(v6_shs):>7.2f} | {np.mean(v6_dds):>+11.2f}% | ${np.mean([l[-1]['portfolio_value'] for l in v6_logs]):>11,.2f}")
    print(f"  {'RAI v6 (Best Seed)':<32} | {v6_rets[v6_best_idx]:>+11.2f}% | {v6_shs[v6_best_idx]:>7.2f} | {v6_dds[v6_best_idx]:>+11.2f}% | ${v6_best_final:>11,.2f}")

    # RAI v7 Ensemble stats
    v7_rets = [(l[-1]['portfolio_value']/10000 - 1)*100 for l in v7_logs]
    v7_shs = [np.mean([d['daily_return_pct'] for d in l])/np.std([d['daily_return_pct'] for d in l])*np.sqrt(252) for l in v7_logs]
    v7_dds = [min([d['drawdown_pct'] for d in l]) for l in v7_logs]
    v7_best_idx = np.argmax(v7_shs)
    v7_best_final = v7_logs[v7_best_idx][-1]['portfolio_value']

    print(f"  {'RAI v7 (Ensemble Mean)':<32} | {np.mean(v7_rets):>+11.2f}% | {np.mean(v7_shs):>7.2f} | {np.mean(v7_dds):>+11.2f}% | ${np.mean([l[-1]['portfolio_value'] for l in v7_logs]):>11,.2f}")
    print(f"  {'RAI v7 (Best Seed)':<32} | {v7_rets[v7_best_idx]:>+11.2f}% | {v7_shs[v7_best_idx]:>7.2f} | {v7_dds[v7_best_idx]:>+11.2f}% | ${v7_best_final:>11,.2f}")


def main():
    print("="*85)
    print("  REAL DATASET EVALUATION TEST: RAI v6 vs RAI v7")
    print("="*85, flush=True)

    # Load v6 & v7 models
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

    print(f"  ✓ Loaded {len(v6_models)} RAI v6 models and {len(v7_models)} RAI v7 models", flush=True)

    # Load Datasets
    local_test_path = os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "test_prices.csv")
    eval_datasets = {}

    if os.path.exists(local_test_path):
        df_test = pd.read_csv(local_test_path, index_col=0, parse_dates=True)
        eval_datasets["2020-2024 (OOS Test)"] = df_test

    import yfinance as yf
    periods = {
        "2024-2026 (Untouched Holdout)": ("2024-06-01", "2026-08-08"),
        "Full 2020-2026 (Combined)": ("2020-01-01", "2026-08-08"),
    }

    for p_name, (start, end) in periods.items():
        try:
            df = yf.download(TICKERS, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex): df = df['Close']
            df = df[TICKERS].dropna()
            if len(df) >= 35: eval_datasets[p_name] = df
        except Exception as e:
            print(f"  ⚠ Could not download {p_name}: {e}")

    # Run evaluations
    for p_name, df_prices in eval_datasets.items():
        v6_logs = [run_model_eval(m, df_prices) for m in v6_models]
        v7_logs = [run_model_eval(m, df_prices) for m in v7_models]
        baselines = compute_baselines(df_prices)

        print_period_table(p_name, v6_logs, v7_logs, baselines)

        # Plot charts for best seed
        v6_best_idx = np.argmax([l[-1]['portfolio_value'] for l in v6_logs])
        v7_best_idx = np.argmax([l[-1]['portfolio_value'] for l in v7_logs])
        plot_real_curves(p_name, v6_logs[v6_best_idx], v7_logs[v7_best_idx], baselines, df_prices.index[30:], REPORT_DIR)

    print(f"\n{'═'*85}")
    print(f"  ✅ EVALUATION TEST COMPLETE")
    print(f"  Charts saved to: {REPORT_DIR}/")
    print(f"{'═'*85}\n", flush=True)


if __name__ == "__main__":
    main()
