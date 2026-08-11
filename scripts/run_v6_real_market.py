"""
═══════════════════════════════════════════════════════════════════════════════
  FINAL RAI v6 REAL MARKET LIVE DEPLOYMENT EXECUTION ENGINE
  ═══════════════════════════════════════════════════════════════════════════════
  Zero-Shot Sim-to-Real Portfolio Allocator
  - Trained on: 100% Synthetic Price Dynamics (0% Real Historical Data)
  - Architecture: Multi-Scale Conv1D + Transformer Encoder
  - Inputs: Real-time Live Market Data (via yfinance)
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import yfinance as yf

warnings.filterwarnings('ignore')
torch.set_num_threads(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rai.world.v6_env import RawPriceSyntheticEnv
from rai.learning.v6_model import DeepEndToEndTradingNet


DEFAULT_ASSETS = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TLT", "GLD"]


def train_rai_v6_zero_shot(seed=42, total_steps=50_000):
    """Train RAI v6 policy entirely in synthetic world G0."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = RawPriceSyntheticEnv(num_assets=10, episode_len=504)
    model = DeepEndToEndTradingNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    obs, _ = env.reset(seed=seed)
    step = 0
    while step < total_steps:
        obs_b, act_b, rew_b, val_b, logp_b = [], [], [], [], []
        for _ in range(1024):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean, val = model(obs_t)
                dist = torch.distributions.Normal(mean, torch.exp(model.log_std))
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            act_np = action.squeeze(0).numpy()
            nobs, rew, done, _, _ = env.step(act_np)
            obs_b.append(obs); act_b.append(act_np); rew_b.append(rew); val_b.append(val.item()); logp_b.append(logp.item())
            obs = nobs
            step += 1
            if done: obs, _ = env.reset()

        with torch.no_grad():
            _, nval = model(torch.FloatTensor(obs).unsqueeze(0))
            nval = nval.item()

        r = np.array(rew_b)
        v = np.array(val_b + [nval])
        delta = r + 0.99 * v[1:] - v[:-1]
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            gae = delta[t] + 0.99 * 0.95 * gae
            adv[t] = gae
        ret = adv + v[:-1]

        o_t = torch.FloatTensor(np.array(obs_b))
        a_t = torch.FloatTensor(np.array(act_b))
        adv_t = torch.FloatTensor(adv)
        ret_t = torch.FloatTensor(ret)
        old_logp_t = torch.FloatTensor(np.array(logp_b))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(4):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), 64):
                b_idx = idx[s:s + 64]
                mean, val = model(o_t[b_idx])
                dist = torch.distributions.Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp_t[b_idx])
                surr1 = ratio * adv_t[b_idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b_idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * torch.nn.functional.mse_loss(val.squeeze(-1), ret_t[b_idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

    model.eval()
    return model


def fetch_live_market_data(assets):
    """Fetch recent live market prices for portfolio assets."""
    print(f"  Fetching live market prices for: {', '.join(assets)}...")
    df = yf.download(assets, period="90d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    df = df.dropna().ffill().bfill()
    valid = [a for a in assets if a in df.columns]
    return df[valid]


def run_live_rai_v6():
    W = 100
    print("=" * W)
    print("  🚀 FINAL RAI v6 REAL MARKET LIVE DEPLOYMENT ENGINE")
    print("=" * W)

    model = train_rai_v6_zero_shot()
    print("  ✓ RAI v6 Policy initialized (0% Real Data Trained / Zero-Shot)")

    df = fetch_live_market_data(DEFAULT_ASSETS)
    prices = df.values
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    print(f"  ✓ Latest market data as of: {latest_date} ({len(prices)} trading days loaded)\n")

    # Build 30-day observation window
    obs_h = []
    T, N = prices.shape
    base_price = prices[-30]
    for t in range(T - 30, T):
        p = prices[t]
        pp = prices[max(0, t - 1)]
        obs_h.append(np.concatenate([
            p / base_price,
            np.log(p / np.maximum(1e-4, pp)),
            [0.10, 0.0]  # Standard cash & drawdown features
        ]).astype(np.float32))

    flat_obs = np.concatenate(obs_h).astype(np.float32)
    action = model.get_action(flat_obs, deterministic=True)

    cash_logit = np.clip(action[0], -5.0, 5.0)
    target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))
    stock_portion = 1.0 - target_cash_frac

    asset_logits = action[1:]
    exp_a = np.exp(asset_logits - np.max(asset_logits))
    asset_weights = (exp_a / np.sum(exp_a)) * stock_portion

    print(f"{'═'*W}")
    print(f"  REAL MARKET PORTFOLIO ALLOCATION DECISION (Date: {latest_date})")
    print(f"{'═'*W}")
    print(f"  Cash Reserves Allocation : {target_cash_frac*100:>6.2f}%")
    print(f"  Equities Allocation      : {stock_portion*100:>6.2f}%\n")

    print(f"  {'Asset Ticker':<15} | {'Asset Weight (%)':<20} | {'Current Market Price ($)':<25}")
    print(f"  {'-'*65}")
    for i, ticker in enumerate(df.columns):
        print(f"  {ticker:<15} | {asset_weights[i]*100:>18.2f}% | ${prices[-1, i]:>23.2f}")

    print(f"  {'-'*65}")
    print(f"  Total Portfolio Allocation : {(target_cash_frac + stock_portion)*100:.2f}%\n")
    print("=" * W)


if __name__ == "__main__":
    run_live_rai_v6()
