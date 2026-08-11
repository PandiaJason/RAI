"""
═══════════════════════════════════════════════════════════
  RAI v6 — LIVE REAL DATASET TEST
  ═════════════════════════════════
  Downloads fresh real market data & runs the best model
  with full portfolio tracking, daily P&L, trade log,
  and visual equity curves.
═══════════════════════════════════════════════════════════
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "robustness")
REPORT_DIR = os.path.join(PROJECT_ROOT, "data", "real_test_report")
os.makedirs(REPORT_DIR, exist_ok=True)

TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]


# ═══════════════════════════════════════════════
#  MODEL
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
        feat = self.fc_features(x.mean(dim=1))
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            return self.forward(flat_obs)[0].cpu().numpy().squeeze(0)


# ═══════════════════════════════════════════════
#  EVALUATION ENGINE (with full tracking)
# ═══════════════════════════════════════════════
def run_full_evaluation(model, prices_df, fee_bps=5, slippage_pct=0.02):
    """Run model on real prices with full trade & portfolio tracking."""
    tickers = list(prices_df.columns)
    prices = prices_df.values
    dates = prices_df.index
    T, N = prices.shape

    # Tracking arrays
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

        # Decode action
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
            old_shares = shares.copy()
            cash = net * target_cash_frac
            shares = (net * target_asset_weights) / np.maximum(1e-4, p_ex)

            # Log trades
            for i in range(N):
                delta = shares[i] - old_shares[i]
                if abs(delta) > 0.001:
                    trade_log.append({
                        'date': str(dates[t].date()) if hasattr(dates[t], 'date') else str(dates[t]),
                        'ticker': tickers[i] if i < len(tickers) else f'Asset_{i}',
                        'action': 'BUY' if delta > 0 else 'SELL',
                        'shares': abs(delta),
                        'price': p[i],
                        'value': abs(delta) * p[i]
                    })

        nw = cash + np.sum(shares * prices[t])
        peak = max(peak, nw)
        dd = (nw - peak) / peak
        daily_ret = (nw - w_before) / w_before if w_before > 0 else 0

        daily_log.append({
            'date': str(dates[t].date()) if hasattr(dates[t], 'date') else str(dates[t]),
            'portfolio_value': nw,
            'daily_return_pct': daily_ret * 100,
            'cash': cash,
            'cash_pct': (cash / nw) * 100,
            'drawdown_pct': dd * 100,
            'traded': traded,
            'turnover': turnover * 100 if traded else 0,
            'weights': {tickers[i]: (shares[i] * prices[t][i]) / nw for i in range(N)}
        })

        # Update observation
        pp = prices[t-1]
        np_ = np.pad(prices[t]/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(prices[t]/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.pop(0)
        obs_h.append(np.concatenate([np_, lr, [cash/max(1e-4, nw), np.clip(dd, -1, 0)]]).astype(np.float32))

    return daily_log, trade_log


def compute_baselines(prices_df):
    """Compute all baseline equity curves."""
    prices = prices_df.values
    dates = prices_df.index
    T, N = prices.shape
    baselines = {}

    # Buy & Hold SPY
    spy_idx = list(prices_df.columns).index('SPY') if 'SPY' in prices_df.columns else 5
    spy_eq = (10000 * prices[:, spy_idx] / prices[0, spy_idx]).tolist()
    baselines['Buy & Hold SPY'] = spy_eq

    # Equal Weight
    sh = (10000.0 / N) / prices[0]
    eq = [10000.0]
    for t in range(1, T):
        w = np.sum(sh * prices[t])
        if t % 21 == 0: sh = (w / N) / prices[t]
        eq.append(w)
    baselines['Equal Weight'] = eq

    # Risk Parity
    sh = (10000.0 / N) / prices[0]
    eq = [10000.0]
    for t in range(1, T):
        w = np.sum(sh * prices[t])
        if t % 21 == 0 and t >= 60:
            v = np.std(np.diff(np.log(prices[t-60:t+1]), axis=0), axis=0)
            iv = 1.0 / np.maximum(1e-8, v)
            wts = iv / iv.sum()
            sh = (w * wts) / prices[t]
        eq.append(w)
    baselines['Risk Parity'] = eq

    # Momentum
    sh = (10000.0 / N) / prices[0]
    eq = [10000.0]
    for t in range(1, T):
        w = np.sum(sh * prices[t])
        if t % 21 == 0 and t >= 60:
            mom = prices[t] / prices[t-60] - 1
            top = np.argsort(mom)[-3:]
            wts = np.zeros(N); wts[top] = 1.0/3
            sh = (w * wts) / np.maximum(1e-8, prices[t])
        eq.append(w)
    baselines['Momentum Top-3'] = eq

    return baselines, dates


def plot_results(daily_log, baselines, dates, prices_df, save_dir):
    """Generate comprehensive charts."""
    plt.style.use('dark_background')

    rai_dates = [d['date'] for d in daily_log]
    rai_eq = [d['portfolio_value'] for d in daily_log]

    # ── CHART 1: Equity Curves ──
    fig, axes = plt.subplots(3, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle('RAI v6 — Real Market Test', fontsize=20, fontweight='bold', color='white', y=0.98)

    ax1 = axes[0]
    colors = {'Buy & Hold SPY': '#4FC3F7', 'Equal Weight': '#81C784',
              'Risk Parity': '#FFB74D', 'Momentum Top-3': '#CE93D8'}

    for name, eq_curve in baselines.items():
        bl_dates = dates[:len(eq_curve)]
        ax1.plot(bl_dates, eq_curve, label=name, color=colors.get(name, 'gray'),
                 alpha=0.6, linewidth=1.2)

    # RAI equity - use matching dates
    rai_plot_dates = pd.to_datetime(rai_dates)
    ax1.plot(rai_plot_dates, rai_eq, label='RAI v6 (Synthetic-Trained)',
             color='#FF5252', linewidth=2.5, zorder=10)

    ax1.set_ylabel('Portfolio Value ($)', fontsize=13, color='white')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.3)
    ax1.grid(True, alpha=0.15)
    ax1.set_title(f'Equity Curves — {rai_dates[0]} to {rai_dates[-1]}', fontsize=14, color='#aaa')

    # Final values annotation
    final_rai = rai_eq[-1]
    ret_pct = (final_rai / 10000 - 1) * 100
    ax1.annotate(f'RAI v6: ${final_rai:,.0f} ({ret_pct:+.1f}%)',
                 xy=(rai_plot_dates[-1], final_rai),
                 xytext=(-150, 20), textcoords='offset points',
                 fontsize=12, color='#FF5252', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#FF5252'))

    # ── CHART 2: Drawdown ──
    ax2 = axes[1]
    dd = [d['drawdown_pct'] for d in daily_log]
    ax2.fill_between(rai_plot_dates, dd, 0, color='#FF5252', alpha=0.4)
    ax2.plot(rai_plot_dates, dd, color='#FF5252', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12, color='white')
    ax2.set_ylim(min(dd) * 1.2, 2)
    ax2.grid(True, alpha=0.15)
    ax2.axhline(0, color='white', alpha=0.3)

    # ── CHART 3: Daily Returns ──
    ax3 = axes[2]
    rets = [d['daily_return_pct'] for d in daily_log]
    colors_r = ['#4CAF50' if r >= 0 else '#FF5252' for r in rets]
    ax3.bar(rai_plot_dates, rets, color=colors_r, alpha=0.7, width=1)
    ax3.set_ylabel('Daily Return (%)', fontsize=12, color='white')
    ax3.grid(True, alpha=0.15)
    ax3.axhline(0, color='white', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, 'equity_curves.png'), dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close()

    # ── CHART 4: Portfolio Weights Over Time ──
    fig, ax = plt.subplots(figsize=(18, 6))
    tickers = list(prices_df.columns)
    weight_data = {t: [] for t in tickers}
    cash_weights = []
    sample_dates = []

    for i, d in enumerate(daily_log):
        if i % 5 == 0:  # Sample every 5 days
            sample_dates.append(d['date'])
            cash_weights.append(d['cash_pct'] / 100)
            for t in tickers:
                weight_data[t].append(d['weights'].get(t, 0))

    sd = pd.to_datetime(sample_dates)
    bottom = np.zeros(len(sample_dates))
    ticker_colors = plt.cm.tab10(np.linspace(0, 1, len(tickers) + 1))

    for i, t in enumerate(tickers):
        vals = np.array(weight_data[t])
        ax.fill_between(sd, bottom, bottom + vals, alpha=0.8,
                        color=ticker_colors[i], label=t)
        bottom += vals

    ax.fill_between(sd, bottom, bottom + np.array(cash_weights), alpha=0.8,
                    color='#9E9E9E', label='Cash')

    ax.set_ylabel('Portfolio Weight', fontsize=13)
    ax.set_title('RAI v6 — Portfolio Allocation Over Time', fontsize=16, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'portfolio_weights.png'), dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close()

    print(f"  ✓ Charts saved to {save_dir}/", flush=True)


def print_stats(daily_log, trade_log, baselines, dates):
    """Print comprehensive statistics."""
    eq = [d['portfolio_value'] for d in daily_log]
    rets = np.array([d['daily_return_pct'] for d in daily_log]) / 100
    trade_days = sum(1 for d in daily_log if d['traded'])
    total_days = len(daily_log)

    final = eq[-1]
    total_ret = (final / 10000 - 1) * 100
    ann_ret = ((final / 10000) ** (252 / total_days) - 1) * 100 if total_days > 0 else 0
    vol = np.std(rets) * np.sqrt(252) * 100
    sharpe = (np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 1e-8 else 0
    pk = np.maximum.accumulate(eq)
    max_dd = np.min((np.array(eq) - pk) / pk) * 100
    win_rate = np.mean(rets > 0) * 100
    avg_win = np.mean(rets[rets > 0]) * 100 if np.any(rets > 0) else 0
    avg_loss = np.mean(rets[rets < 0]) * 100 if np.any(rets < 0) else 0
    profit_factor = abs(np.sum(rets[rets > 0]) / np.sum(rets[rets < 0])) if np.any(rets < 0) else float('inf')
    avg_cash = np.mean([d['cash_pct'] for d in daily_log])
    avg_turnover = np.mean([d['turnover'] for d in daily_log if d['traded']]) if trade_days > 0 else 0

    W = 80
    print(f"\n{'═'*W}")
    print(f"  RAI v6 — REAL MARKET TEST RESULTS")
    print(f"  Period: {daily_log[0]['date']} → {daily_log[-1]['date']} ({total_days} trading days)")
    print(f"{'═'*W}")

    print(f"\n  {'PERFORMANCE METRICS':─<{W-2}}")
    print(f"  Starting Capital:       $10,000.00")
    print(f"  Final Value:            ${final:>12,.2f}")
    print(f"  Total Return:           {total_ret:>+10.2f}%")
    print(f"  Annualized Return:      {ann_ret:>+10.2f}%")
    print(f"  Volatility (ann.):      {vol:>10.2f}%")
    print(f"  Sharpe Ratio:           {sharpe:>10.3f}")
    print(f"  Maximum Drawdown:       {max_dd:>+10.2f}%")

    print(f"\n  {'TRADING STATISTICS':─<{W-2}}")
    print(f"  Days Traded:            {trade_days:>10d} / {total_days} ({trade_days/total_days*100:.1f}%)")
    print(f"  Win Rate:               {win_rate:>10.1f}%")
    print(f"  Avg Win:                {avg_win:>+10.4f}%")
    print(f"  Avg Loss:               {avg_loss:>+10.4f}%")
    print(f"  Profit Factor:          {profit_factor:>10.2f}")
    print(f"  Avg Cash Allocation:    {avg_cash:>10.1f}%")
    print(f"  Avg Turnover (trades):  {avg_turnover:>10.2f}%")
    print(f"  Total Trades:           {len(trade_log):>10d}")

    # Best/Worst days
    sorted_days = sorted(daily_log, key=lambda d: d['daily_return_pct'])
    print(f"\n  {'BEST/WORST DAYS':─<{W-2}}")
    print(f"  {'Date':<14} {'Return':>10} {'Value':>14}")
    print(f"  {'-'*40}")
    for d in sorted_days[-3:][::-1]:
        print(f"  {d['date']:<14} {d['daily_return_pct']:>+9.3f}% ${d['portfolio_value']:>12,.2f}  ← BEST")
    print(f"  {'...'}")
    for d in sorted_days[:3]:
        print(f"  {d['date']:<14} {d['daily_return_pct']:>+9.3f}% ${d['portfolio_value']:>12,.2f}  ← WORST")

    # Baseline comparison
    print(f"\n  {'COMPARISON vs BASELINES':─<{W-2}}")
    print(f"  {'Strategy':<25} {'Return':>10} {'Final':>14}")
    print(f"  {'-'*55}")
    for name, eq_curve in baselines.items():
        bl_ret = (eq_curve[-1] / 10000 - 1) * 100
        print(f"  {name:<25} {bl_ret:>+9.2f}% ${eq_curve[-1]:>12,.2f}")
    print(f"  {'-'*55}")
    print(f"  {'RAI v6 (synthetic-trained)':<25} {total_ret:>+9.2f}% ${final:>12,.2f}")

    # Monthly returns
    print(f"\n  {'MONTHLY RETURNS':─<{W-2}}")
    monthly = {}
    for d in daily_log:
        ym = d['date'][:7]
        if ym not in monthly: monthly[ym] = []
        monthly[ym].append(d['daily_return_pct'] / 100)

    print(f"  {'Month':<10} {'Return':>10} {'Days':>6}")
    print(f"  {'-'*30}")
    for ym in sorted(monthly.keys()):
        mr = (np.prod(1 + np.array(monthly[ym])) - 1) * 100
        marker = "📈" if mr > 0 else "📉"
        print(f"  {ym:<10} {mr:>+9.2f}% {len(monthly[ym]):>5d}  {marker}")


def main():
    print("="*80)
    print("  RAI v6 — REAL DATASET TEST")
    print("  Downloading fresh data & running evaluation")
    print("="*80, flush=True)

    # Download latest real data
    import yfinance as yf
    periods = {
        "2020-2024 (OOS Test)": ("2020-01-01", "2024-06-01"),
        "2024-2026 (Holdout)": ("2024-06-01", "2026-08-08"),
        "Full 2020-2026": ("2020-01-01", "2026-08-08"),
    }

    # Load all 10 RAI v6 seeds
    models = {}
    for seed in range(1, 11):
        path = os.path.join(RESULTS_DIR, "seeds", f"rai_v6_seed_{seed:02d}.pt")
        if os.path.exists(path):
            m = DeepEndToEndTradingNet()
            m.load_state_dict(torch.load(path, weights_only=True))
            m.eval()
            models[seed] = m
            print(f"  ✓ Loaded seed {seed:02d}", flush=True)

    best_seed = 4  # From robustness experiment
    print(f"\n  Using best seed: {best_seed} (highest test Sharpe)", flush=True)

    for period_name, (start, end) in periods.items():
        print(f"\n{'═'*80}")
        print(f"  TESTING: {period_name}")
        print(f"{'═'*80}", flush=True)

        print(f"  ↓ Downloading {start} → {end}...", flush=True)
        df = yf.download(TICKERS, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close'] if 'Close' in df.columns.get_level_values(0) else df.iloc[:, :len(TICKERS)]
        df = df[TICKERS].dropna()
        print(f"  ✓ Downloaded: {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})", flush=True)

        if len(df) < 40:
            print(f"  ⚠ Too few data points, skipping", flush=True)
            continue

        # Run best model
        model = models[best_seed]
        daily_log, trade_log = run_full_evaluation(model, df, fee_bps=5, slippage_pct=0.02)
        baselines, dates = compute_baselines(df)

        # Print results
        print_stats(daily_log, trade_log, baselines, dates)

        # Run all 10 seeds for ensemble stats
        print(f"\n  {'ENSEMBLE (10 seeds)':─<78}")
        seed_returns = []
        seed_sharpes = []
        for seed, m in models.items():
            dl, _ = run_full_evaluation(m, df, fee_bps=5, slippage_pct=0.02)
            eq = [d['portfolio_value'] for d in dl]
            ret = (eq[-1] / 10000 - 1) * 100
            rets = np.diff(eq) / np.maximum(1e-8, eq[:-1])
            sh = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 1e-8 else 0
            seed_returns.append(ret)
            seed_sharpes.append(sh)
            print(f"  Seed {seed:02d}: Return={ret:>+8.2f}%  Sharpe={sh:>6.3f}", flush=True)

        print(f"  {'-'*50}")
        print(f"  Mean Return:  {np.mean(seed_returns):>+8.2f}% ± {np.std(seed_returns):.2f}%")
        print(f"  Mean Sharpe:  {np.mean(seed_sharpes):>8.3f} ± {np.std(seed_sharpes):.3f}")
        print(f"  Min Return:   {np.min(seed_returns):>+8.2f}%")
        print(f"  Max Return:   {np.max(seed_returns):>+8.2f}%")
        print(f"  All Positive: {'✅ YES' if all(r > 0 for r in seed_returns) else '❌ NO'}")

        # Plot charts for this period
        safe_name = period_name.replace(' ', '_').replace('(', '').replace(')', '')
        plot_results(daily_log, baselines, dates, df, REPORT_DIR)
        # Rename charts with period name
        for f in ['equity_curves.png', 'portfolio_weights.png']:
            src = os.path.join(REPORT_DIR, f)
            dst = os.path.join(REPORT_DIR, f"{safe_name}_{f}")
            if os.path.exists(src):
                os.rename(src, dst)
                print(f"  ✓ Saved: {dst}", flush=True)

    # Save trade log
    if trade_log:
        tl_path = os.path.join(REPORT_DIR, "trade_log.json")
        with open(tl_path, 'w') as f:
            json.dump(trade_log[:200], f, indent=2, default=str)
        print(f"\n  ✓ Trade log: {tl_path} ({len(trade_log)} trades)")

    print(f"\n{'═'*80}")
    print(f"  ✅ REAL DATA TEST COMPLETE")
    print(f"  Charts & reports saved to: {REPORT_DIR}/")
    print(f"{'═'*80}", flush=True)


if __name__ == "__main__":
    main()
