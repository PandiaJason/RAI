"""
═══════════════════════════════════════════════════════════════════════════════
  RAI v6 — ALLOCATION FORENSICS
  ═════════════════════════════
  The critical question:
    Does RAI do anything beyond equal weighting?

  Compare DAILY WEIGHTS of:
    1. RAI v6 (best seed)
    2. Equal Weight (monthly rebalance)
    3. Fixed 60/40 (equity/bond split)
    4. Simple Volatility Targeting

  If RAI ≈ Equal Weight in allocations → it rediscovered diversification
  If RAI ≠ Equal Weight but similar returns → it learned something distinct
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, warnings, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cosine

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "robustness")
REPORT_DIR = os.path.join(PROJECT_ROOT, "data", "allocation_forensics")
os.makedirs(REPORT_DIR, exist_ok=True)

TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]
# Asset classes
EQUITY_TICKERS = ["EEM", "QQQ", "SPY", "VNQ"]  # Equities + REIT
BOND_TICKERS = ["HYG", "TLT"]                    # Bonds
COMMODITY_TICKERS = ["DBC", "GLD", "USO"]         # Commodities
CURRENCY_TICKERS = ["UUP"]                         # Dollar


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
        return self.actor_head(self.fc_features(x.mean(dim=1))), self.critic_head(self.fc_features(x.mean(dim=1)))

    def get_action(self, flat_obs):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            return self.forward(flat_obs)[0].cpu().numpy().squeeze(0)


# ═══════════════════════════════════════════════
#  STRATEGY TRACKERS (return daily weights)
# ═══════════════════════════════════════════════

def run_rai_v6(model, prices, dates):
    """Run RAI v6, return daily weights + equity curve."""
    T, N = prices.shape
    cash = 500.0
    shares = (9500.0 / N) / prices[30]
    peak = 10000.0
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

        # Record TARGET weights (what the model wants)
        target_w = np.zeros(N + 1)  # N assets + cash
        target_w[:n] = taw
        target_w[-1] = tc

        if abs(ccf - tc) + np.sum(np.abs(caw - taw)) > 0.03:
            tv = abs(cash - w*tc) + np.sum(np.abs(shares*p - w*taw))
            net = max(1e-4, w - tv*0.0005)
            cash = net * tc
            shares = (net * taw) / np.maximum(1e-4, p)

        nw = cash + np.sum(shares * p)
        peak = max(peak, nw)

        # Record ACTUAL weights
        actual_w = np.zeros(N + 1)
        actual_w[:N] = (shares * p) / nw
        actual_w[-1] = cash / nw

        records.append({
            'date': dates[t],
            'wealth': nw,
            'target_weights': target_w.copy(),
            'actual_weights': actual_w.copy(),
            'cash_pct': (cash / nw) * 100,
        })

        pp = prices[t-1]
        np_ = np.pad(p/prices[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.pop(0)
        obs_h.append(np.concatenate([np_, lr, [cash/max(1e-4,nw),
                     np.clip((nw-peak)/max(1e-4,peak),-1,0)]]).astype(np.float32))

    return records


def run_equal_weight(prices, dates):
    """Equal weight monthly rebalance."""
    T, N = prices.shape
    shares = (10000.0 / N) / prices[30]
    records = []
    for t in range(30, T):
        w = np.sum(shares * prices[t])
        if (t - 30) % 21 == 0 and t > 30:
            shares = (w / N) / prices[t]
        weights = np.zeros(N + 1)
        weights[:N] = (shares * prices[t]) / w
        weights[-1] = 0.0
        records.append({'date': dates[t], 'wealth': w, 'actual_weights': weights.copy()})
    return records


def run_sixty_forty(prices, dates):
    """60% equities / 40% bonds, monthly rebalance."""
    T, N = prices.shape
    eq_idx = [TICKERS.index(t) for t in EQUITY_TICKERS if t in TICKERS]
    bd_idx = [TICKERS.index(t) for t in BOND_TICKERS if t in TICKERS]
    ot_idx = [i for i in range(N) if i not in eq_idx and i not in bd_idx]

    w_target = np.zeros(N)
    for i in eq_idx: w_target[i] = 0.60 / len(eq_idx)
    for i in bd_idx: w_target[i] = 0.30 / len(bd_idx)
    for i in ot_idx: w_target[i] = 0.10 / len(ot_idx) if ot_idx else 0

    shares = (10000.0 * w_target) / prices[30]
    records = []
    for t in range(30, T):
        w = np.sum(shares * prices[t])
        if (t - 30) % 21 == 0 and t > 30:
            shares = (w * w_target) / prices[t]
        weights = np.zeros(N + 1)
        weights[:N] = (shares * prices[t]) / w
        weights[-1] = 0.0
        records.append({'date': dates[t], 'wealth': w, 'actual_weights': weights.copy()})
    return records


def run_vol_target(prices, dates, target_vol=0.12, lookback=60):
    """Simple volatility targeting: scale exposure to maintain target vol."""
    T, N = prices.shape
    shares = (10000.0 / N) / prices[30]
    cash = 0.0
    records = []
    for t in range(30, T):
        w = cash + np.sum(shares * prices[t])
        if (t - 30) % 21 == 0 and t > 30 and t >= lookback:
            log_rets = np.diff(np.log(prices[t-lookback:t+1]), axis=0)
            port_rets = np.mean(log_rets, axis=1)  # Equal-weight portfolio returns
            realized_vol = np.std(port_rets) * np.sqrt(252)
            scale = min(1.5, max(0.3, target_vol / max(0.01, realized_vol)))
            invested = w * scale
            cash = w - invested
            shares = (invested / N) / prices[t]
        weights = np.zeros(N + 1)
        weights[:N] = (shares * prices[t]) / max(1e-4, w)
        weights[-1] = cash / max(1e-4, w)
        records.append({'date': dates[t], 'wealth': w, 'actual_weights': weights.copy()})
    return records


# ═══════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════

def analyze_weights(rai_recs, ew_recs, sf_recs, vt_recs, dates):
    """Deep comparison of allocation behavior."""
    n_days = min(len(rai_recs), len(ew_recs), len(sf_recs), len(vt_recs))
    N = len(TICKERS)

    rai_w = np.array([r['actual_weights'][:N] for r in rai_recs[:n_days]])
    ew_w = np.array([r['actual_weights'][:N] for r in ew_recs[:n_days]])
    sf_w = np.array([r['actual_weights'][:N] for r in sf_recs[:n_days]])
    vt_w = np.array([r['actual_weights'][:N] for r in vt_recs[:n_days]])
    rai_cash = np.array([r['actual_weights'][-1] for r in rai_recs[:n_days]])

    results = {}

    # ── 1. Average weights ──
    print(f"\n  {'AVERAGE PORTFOLIO WEIGHTS':─<70}", flush=True)
    print(f"  {'Ticker':<8} {'RAI v6':>10} {'EqualWt':>10} {'60/40':>10} {'VolTgt':>10} {'RAI-EW diff':>12}", flush=True)
    print(f"  {'-'*62}", flush=True)
    avg_diffs = []
    for i, t in enumerate(TICKERS):
        rm, em, sm, vm = np.mean(rai_w[:, i]), np.mean(ew_w[:, i]), np.mean(sf_w[:, i]), np.mean(vt_w[:, i])
        diff = rm - em
        avg_diffs.append(diff)
        marker = "◄" if abs(diff) > 0.02 else ""
        print(f"  {t:<8} {rm:>9.1%} {em:>9.1%} {sm:>9.1%} {vm:>9.1%} {diff:>+11.2%} {marker}", flush=True)
    print(f"  {'Cash':<8} {np.mean(rai_cash):>9.1%} {'0.0%':>10} {'0.0%':>10} {np.mean([r['actual_weights'][-1] for r in vt_recs[:n_days]]):>9.1%}", flush=True)

    results['avg_weights'] = {TICKERS[i]: float(np.mean(rai_w[:, i])) for i in range(N)}
    results['avg_weights']['Cash'] = float(np.mean(rai_cash))
    results['avg_ew_weights'] = {TICKERS[i]: float(np.mean(ew_w[:, i])) for i in range(N)}

    # ── 2. Weight similarity metrics ──
    print(f"\n  {'WEIGHT SIMILARITY METRICS':─<70}", flush=True)

    # Cosine similarity per day
    cos_ew = [1 - cosine(rai_w[t], ew_w[t]) for t in range(n_days) if np.sum(rai_w[t]) > 0]
    cos_sf = [1 - cosine(rai_w[t], sf_w[t]) for t in range(n_days) if np.sum(rai_w[t]) > 0]
    cos_vt = [1 - cosine(rai_w[t], vt_w[t]) for t in range(n_days) if np.sum(rai_w[t]) > 0]

    # L1 distance (total absolute difference)
    l1_ew = np.mean(np.sum(np.abs(rai_w - ew_w), axis=1))
    l1_sf = np.mean(np.sum(np.abs(rai_w - sf_w), axis=1))
    l1_vt = np.mean(np.sum(np.abs(rai_w - vt_w), axis=1))

    # Correlation of weight time series
    corr_ew = np.mean([np.corrcoef(rai_w[:, i], ew_w[:, i])[0, 1] for i in range(N)])
    corr_sf = np.mean([np.corrcoef(rai_w[:, i], sf_w[:, i])[0, 1] for i in range(N)])

    print(f"  {'Metric':<30} {'vs EqualWt':>12} {'vs 60/40':>12} {'vs VolTgt':>12}", flush=True)
    print(f"  {'-'*68}", flush=True)
    print(f"  {'Cosine Similarity (mean)':<30} {np.mean(cos_ew):>11.4f} {np.mean(cos_sf):>11.4f} {np.mean(cos_vt):>11.4f}", flush=True)
    print(f"  {'L1 Distance (mean)':<30} {l1_ew:>11.4f} {l1_sf:>11.4f} {l1_vt:>11.4f}", flush=True)
    print(f"  {'Weight Correlation (mean)':<30} {corr_ew:>11.4f} {corr_sf:>11.4f} {'N/A':>12}", flush=True)

    results['similarity'] = {
        'cosine_vs_ew': float(np.mean(cos_ew)),
        'cosine_vs_6040': float(np.mean(cos_sf)),
        'l1_vs_ew': float(l1_ew),
        'l1_vs_6040': float(l1_sf),
    }

    # ── 3. Weight dispersion ──
    print(f"\n  {'WEIGHT DISPERSION (how concentrated?)':─<70}", flush=True)
    rai_hhi = np.mean([np.sum(rai_w[t]**2) for t in range(n_days)])
    ew_hhi = np.mean([np.sum(ew_w[t]**2) for t in range(n_days)])
    sf_hhi = np.mean([np.sum(sf_w[t]**2) for t in range(n_days)])
    vt_hhi = np.mean([np.sum(vt_w[t]**2) for t in range(n_days)])

    # Effective number of assets = 1/HHI
    print(f"  {'Strategy':<20} {'HHI':>10} {'Eff. Assets':>12} {'Max Weight':>12} {'Min Weight':>12}", flush=True)
    print(f"  {'-'*68}", flush=True)
    print(f"  {'RAI v6':<20} {rai_hhi:>9.4f} {1/max(1e-8,rai_hhi):>11.1f} {np.max(np.mean(rai_w, axis=0)):>11.1%} {np.min(np.mean(rai_w, axis=0)):>11.1%}", flush=True)
    print(f"  {'Equal Weight':<20} {ew_hhi:>9.4f} {1/max(1e-8,ew_hhi):>11.1f} {np.max(np.mean(ew_w, axis=0)):>11.1%} {np.min(np.mean(ew_w, axis=0)):>11.1%}", flush=True)
    print(f"  {'60/40':<20} {sf_hhi:>9.4f} {1/max(1e-8,sf_hhi):>11.1f} {np.max(np.mean(sf_w, axis=0)):>11.1%} {np.min(np.mean(sf_w, axis=0)):>11.1%}", flush=True)
    print(f"  {'Vol Target':<20} {vt_hhi:>9.4f} {1/max(1e-8,vt_hhi):>11.1f} {np.max(np.mean(vt_w, axis=0)):>11.1%} {np.min(np.mean(vt_w, axis=0)):>11.1%}", flush=True)

    results['concentration'] = {
        'rai_hhi': float(rai_hhi), 'ew_hhi': float(ew_hhi),
        'rai_eff_assets': float(1/max(1e-8,rai_hhi)),
        'ew_eff_assets': float(1/max(1e-8,ew_hhi)),
    }

    # ── 4. Asset class exposure ──
    print(f"\n  {'ASSET CLASS EXPOSURE':─<70}", flush=True)
    def class_exposure(w_arr):
        eq_exp = np.mean(np.sum(w_arr[:, [TICKERS.index(t) for t in EQUITY_TICKERS]], axis=1))
        bd_exp = np.mean(np.sum(w_arr[:, [TICKERS.index(t) for t in BOND_TICKERS]], axis=1))
        cm_exp = np.mean(np.sum(w_arr[:, [TICKERS.index(t) for t in COMMODITY_TICKERS]], axis=1))
        cu_exp = np.mean(np.sum(w_arr[:, [TICKERS.index(t) for t in CURRENCY_TICKERS]], axis=1))
        return eq_exp, bd_exp, cm_exp, cu_exp

    classes = ['Equity', 'Bonds', 'Commodity', 'Currency']
    rai_cls = class_exposure(rai_w)
    ew_cls = class_exposure(ew_w)
    sf_cls = class_exposure(sf_w)
    print(f"  {'Class':<15} {'RAI v6':>10} {'EqualWt':>10} {'60/40':>10} {'RAI-EW':>10}", flush=True)
    print(f"  {'-'*57}", flush=True)
    for i, c in enumerate(classes):
        diff = rai_cls[i] - ew_cls[i]
        marker = " ◄ OVERWEIGHT" if diff > 0.03 else (" ◄ UNDERWEIGHT" if diff < -0.03 else "")
        print(f"  {c:<15} {rai_cls[i]:>9.1%} {ew_cls[i]:>9.1%} {sf_cls[i]:>9.1%} {diff:>+9.2%}{marker}", flush=True)
    print(f"  {'Cash':<15} {np.mean(rai_cash):>9.1%} {'0.0%':>10} {'0.0%':>10}", flush=True)

    results['asset_class'] = {c: float(rai_cls[i]) for i, c in enumerate(classes)}
    results['asset_class']['Cash'] = float(np.mean(rai_cash))

    # ── 5. Weight dynamics: does RAI change weights over time? ──
    print(f"\n  {'WEIGHT DYNAMICS (does RAI adapt?)':─<70}", flush=True)

    # Weight change magnitude per day
    rai_turnover = np.sum(np.abs(np.diff(rai_w, axis=0)), axis=1)
    ew_turnover = np.sum(np.abs(np.diff(ew_w, axis=0)), axis=1)

    print(f"  {'Metric':<35} {'RAI v6':>12} {'EqualWt':>12}", flush=True)
    print(f"  {'-'*62}", flush=True)
    print(f"  {'Daily weight change (mean L1)':<35} {np.mean(rai_turnover):>11.5f} {np.mean(ew_turnover):>11.5f}", flush=True)
    print(f"  {'Daily weight change (std)':<35} {np.std(rai_turnover):>11.5f} {np.std(ew_turnover):>11.5f}", flush=True)
    print(f"  {'Max daily weight change':<35} {np.max(rai_turnover):>11.5f} {np.max(ew_turnover):>11.5f}", flush=True)
    print(f"  {'Weight std over time (mean)':<35} {np.mean(np.std(rai_w, axis=0)):>11.5f} {np.mean(np.std(ew_w, axis=0)):>11.5f}", flush=True)

    # ── 6. Regime-dependent behavior ──
    print(f"\n  {'REGIME-DEPENDENT BEHAVIOR':─<70}", flush=True)
    # Split into quarters and check if weights change
    quarter_len = n_days // 4
    for q in range(4):
        s, e = q * quarter_len, (q+1) * quarter_len
        q_rai = np.mean(rai_w[s:e], axis=0)
        q_ew = np.mean(ew_w[s:e], axis=0)
        q_cos = np.mean([1 - cosine(rai_w[t], ew_w[t]) for t in range(s, e) if np.sum(rai_w[t]) > 0])
        d0 = rai_recs[s]['date']
        d1 = rai_recs[min(e-1, len(rai_recs)-1)]['date']
        top_rai = TICKERS[np.argmax(q_rai)]
        top_ew = TICKERS[np.argmax(q_ew)]
        eq_exp = np.sum(q_rai[[TICKERS.index(t) for t in EQUITY_TICKERS]])
        print(f"  Q{q+1} ({str(d0.date())[:10]} → {str(d1.date())[:10]}): "
              f"Cos={q_cos:.4f} | Top RAI={top_rai} | Equity={eq_exp:.1%} | "
              f"Cash={np.mean(rai_cash[s:e]):.1%}", flush=True)

    return results, rai_w, ew_w, sf_w, vt_w, rai_cash, n_days


def plot_forensics(rai_w, ew_w, sf_w, vt_w, rai_cash, n_days, dates_plot, save_dir):
    """Generate weight comparison charts."""
    plt.style.use('dark_background')
    N = len(TICKERS)

    # ── Chart 1: Weight difference heatmap (RAI - EqualWeight) ──
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Sample every 5 days for readability
    step = 5
    diff = rai_w[::step] - ew_w[::step]
    d_dates = dates_plot[::step]

    im = axes[0].imshow(diff.T, aspect='auto', cmap='RdBu_r', vmin=-0.05, vmax=0.05,
                        interpolation='nearest')
    axes[0].set_yticks(range(N))
    axes[0].set_yticklabels(TICKERS, fontsize=10)
    axes[0].set_title('RAI v6 vs Equal Weight — Daily Weight Difference', fontsize=14, fontweight='bold')
    n_ticks = min(8, len(d_dates))
    tick_pos = np.linspace(0, len(d_dates)-1, n_ticks).astype(int)
    axes[0].set_xticks(tick_pos)
    axes[0].set_xticklabels([str(d_dates[i].date())[:10] for i in tick_pos], rotation=45, fontsize=9)
    plt.colorbar(im, ax=axes[0], label='RAI - EqualWeight', shrink=0.8)

    # ── Chart 2: Asset class exposure over time ──
    eq_idx = [TICKERS.index(t) for t in EQUITY_TICKERS]
    bd_idx = [TICKERS.index(t) for t in BOND_TICKERS]
    cm_idx = [TICKERS.index(t) for t in COMMODITY_TICKERS]

    rai_eq_exp = np.sum(rai_w[:, eq_idx], axis=1)
    ew_eq_exp = np.sum(ew_w[:, eq_idx], axis=1)
    rai_bd_exp = np.sum(rai_w[:, bd_idx], axis=1)
    ew_bd_exp = np.sum(ew_w[:, bd_idx], axis=1)

    axes[1].plot(dates_plot, rai_eq_exp, color='#FF5252', linewidth=1.5, label='RAI Equity', alpha=0.9)
    axes[1].plot(dates_plot, ew_eq_exp, color='#4FC3F7', linewidth=1.5, label='EW Equity', alpha=0.9)
    axes[1].plot(dates_plot, rai_bd_exp, color='#FF8A65', linewidth=1.5, label='RAI Bonds', alpha=0.7)
    axes[1].plot(dates_plot, ew_bd_exp, color='#81C784', linewidth=1.5, label='EW Bonds', alpha=0.7)
    axes[1].fill_between(dates_plot, rai_cash, 0, color='gray', alpha=0.3, label='RAI Cash')
    axes[1].set_ylabel('Allocation %', fontsize=12)
    axes[1].set_title('Asset Class Exposure: RAI v6 vs Equal Weight', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9, ncol=3, loc='upper right')
    axes[1].grid(True, alpha=0.15)

    # ── Chart 3: Cosine similarity over time ──
    cos_daily = [1 - cosine(rai_w[t], ew_w[t]) if np.sum(rai_w[t]) > 0 else 1.0 for t in range(n_days)]
    axes[2].plot(dates_plot, cos_daily, color='#AB47BC', linewidth=1, alpha=0.7)
    # Smoothed
    window = 21
    if len(cos_daily) > window:
        smoothed = pd.Series(cos_daily).rolling(window, center=True).mean()
        axes[2].plot(dates_plot, smoothed, color='#AB47BC', linewidth=2.5, label=f'{window}d smoothed')
    axes[2].set_ylabel('Cosine Similarity', fontsize=12)
    axes[2].set_title('RAI v6 vs Equal Weight — Weight Similarity Over Time', fontsize=13, fontweight='bold')
    axes[2].axhline(1.0, color='white', alpha=0.3, linestyle='--', label='Identical')
    axes[2].set_ylim(0.85, 1.02)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_forensics.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()

    # ── Chart 4: Individual ticker weight comparison ──
    fig, axes = plt.subplots(5, 2, figsize=(18, 16), sharex=True)
    for i, (ax, ticker) in enumerate(zip(axes.flat, TICKERS)):
        ax.plot(dates_plot, rai_w[:, i], color='#FF5252', linewidth=1, alpha=0.8, label='RAI v6')
        ax.plot(dates_plot, ew_w[:, i], color='#4FC3F7', linewidth=1, alpha=0.8, label='EqualWt')
        ax.plot(dates_plot, sf_w[:, i], color='#FFB74D', linewidth=1, alpha=0.5, label='60/40')
        ax.set_title(ticker, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(0.25, np.max(rai_w[:, i]) * 1.2))
        ax.grid(True, alpha=0.15)
        if i == 0: ax.legend(fontsize=8)

    fig.suptitle('Per-Ticker Weight Comparison: RAI v6 vs Baselines', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_ticker_weights.png'), dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()

    print(f"  ✓ Charts saved to {save_dir}/", flush=True)


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════

def main():
    W = 80
    print("="*W)
    print("  RAI v6 — ALLOCATION FORENSICS")
    print("  Does RAI do anything beyond equal weighting?")
    print("="*W, flush=True)

    # Download real data (full period)
    import yfinance as yf
    print(f"\n  Downloading 2020-01-01 → 2026-08-08...", flush=True)
    df = yf.download(TICKERS, start="2020-01-01", end="2026-08-08", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    df = df[TICKERS].dropna()
    prices = df.values
    dates = df.index
    print(f"  ✓ {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})", flush=True)

    # Load ALL 10 RAI v6 seeds and run forensics on each
    print(f"\n  Loading RAI v6 models...", flush=True)
    all_seed_results = []

    for seed in range(1, 11):
        path = os.path.join(RESULTS_DIR, "seeds", f"rai_v6_seed_{seed:02d}.pt")
        if not os.path.exists(path):
            continue
        model = DeepEndToEndTradingNet()
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()

        print(f"\n{'─'*W}")
        print(f"  SEED {seed}")
        print(f"{'─'*W}", flush=True)

        rai_recs = run_rai_v6(model, prices, dates)
        ew_recs = run_equal_weight(prices, dates)
        sf_recs = run_sixty_forty(prices, dates)
        vt_recs = run_vol_target(prices, dates)

        res, rai_w, ew_w, sf_w, vt_w, rai_cash, n_days = analyze_weights(
            rai_recs, ew_recs, sf_recs, vt_recs, dates)

        all_seed_results.append(res)

        # Plot for best seed (seed 4) only
        if seed == 4:
            dates_plot = dates[30:30+n_days]
            plot_forensics(rai_w, ew_w, sf_w, vt_w, rai_cash, n_days, dates_plot, REPORT_DIR)

    # ── Cross-seed summary ──
    print(f"\n{'═'*W}")
    print(f"  CROSS-SEED SUMMARY (10 seeds)")
    print(f"{'═'*W}", flush=True)

    cos_vals = [r['similarity']['cosine_vs_ew'] for r in all_seed_results]
    l1_vals = [r['similarity']['l1_vs_ew'] for r in all_seed_results]
    hhi_vals = [r['concentration']['rai_hhi'] for r in all_seed_results]
    eff_vals = [r['concentration']['rai_eff_assets'] for r in all_seed_results]

    print(f"  Cosine similarity to EW:  {np.mean(cos_vals):.4f} ± {np.std(cos_vals):.4f}")
    print(f"  L1 distance from EW:      {np.mean(l1_vals):.4f} ± {np.std(l1_vals):.4f}")
    print(f"  HHI concentration:        {np.mean(hhi_vals):.4f} ± {np.std(hhi_vals):.4f}")
    print(f"  Effective # of assets:    {np.mean(eff_vals):.1f} ± {np.std(eff_vals):.1f}")
    print(f"  (Equal Weight HHI:        {all_seed_results[0]['concentration']['ew_hhi']:.4f})")
    print(f"  (Equal Weight Eff. Assets: {all_seed_results[0]['concentration']['ew_eff_assets']:.1f})")

    # ── Final verdict ──
    mean_cos = np.mean(cos_vals)
    print(f"\n{'═'*W}")
    if mean_cos > 0.99:
        print(f"  VERDICT: RAI v6 ≈ Equal Weight")
        print(f"  Cosine similarity {mean_cos:.4f} → allocations are nearly identical.")
        print(f"  RAI converged to a simple diversification policy.")
    elif mean_cos > 0.95:
        print(f"  VERDICT: RAI v6 ≈ Equal Weight with minor deviations")
        print(f"  Cosine similarity {mean_cos:.4f} → mostly the same, small tilts.")
    elif mean_cos > 0.85:
        print(f"  VERDICT: RAI v6 has LEARNED DISTINCT behavior")
        print(f"  Cosine similarity {mean_cos:.4f} → meaningfully different allocations.")
        print(f"  This suggests the model learned something beyond diversification.")
    else:
        print(f"  VERDICT: RAI v6 is SUBSTANTIALLY DIFFERENT from Equal Weight")
        print(f"  Cosine similarity {mean_cos:.4f} → very different allocation strategy.")
    print(f"{'═'*W}", flush=True)

    # Save results
    rp = os.path.join(REPORT_DIR, "forensics_results.json")
    with open(rp, 'w') as f:
        json.dump({'seeds': all_seed_results, 'summary': {
            'mean_cosine_vs_ew': float(np.mean(cos_vals)),
            'mean_l1_vs_ew': float(np.mean(l1_vals)),
            'mean_hhi': float(np.mean(hhi_vals)),
        }}, f, indent=2, default=str)
    print(f"\n  Results: {rp}")


if __name__ == "__main__":
    main()
