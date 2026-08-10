"""
═══════════════════════════════════════════════════════════════════════════════
  RAI v7 vs RAI v6: REAL-DATA BENCHMARK EVALUATION SCRIPT
  ═══════════════════════════════════════════════════════════════════════════════
  Trains 5 seeds of RAI v7 (G6 Jump-Diffusion + Spatio-Temporal Transformer)
  and compares out-of-sample real market performance against RAI v6.
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')
torch.set_num_threads(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rai.world.v7_g6_env import G6EnhancedSyntheticEnv
from rai.learning.v7_model import SpatioTemporalTradingNet
from scripts.train_v6_fast import DeepEndToEndTradingNet, RawPriceSyntheticEnv

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "v7_benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "EEM", "VNQ", "HYG", "TLT", "DBC", "GLD", "USO", "UUP"]
N_SEEDS = 5
TOTAL_STEPS = 100_000
ROLLOUT = 1024
BATCH = 64
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
LR = 3e-4


def train_proper_ppo(model, env, seed, total_steps=TOTAL_STEPS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    obs, _ = env.reset(seed=seed)
    step = 0
    t0 = time.time()

    while step < total_steps:
        obs_b, act_b, rew_b, val_b, logp_b = [], [], [], [], []
        for _ in range(ROLLOUT):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean, val = model(obs_t)
                dist = Normal(mean, torch.exp(model.log_std))
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            act_np = action.squeeze(0).numpy()
            nobs, rew, done, _, _ = env.step(act_np)
            obs_b.append(obs); act_b.append(act_np); rew_b.append(rew); val_b.append(val.item()); logp_b.append(logp.item())
            obs = nobs
            step += 1
            if done:
                obs, _ = env.reset()

        with torch.no_grad():
            _, nval = model(torch.FloatTensor(obs).unsqueeze(0))
            nval = nval.item()

        r = np.array(rew_b)
        v = np.array(val_b + [nval])
        delta = r + GAMMA * v[1:] - v[:-1]
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            gae = delta[t] + GAMMA * GAE_LAMBDA * gae
            adv[t] = gae
        ret = adv + v[:-1]

        o_t = torch.FloatTensor(np.array(obs_b))
        a_t = torch.FloatTensor(np.array(act_b))
        adv_t = torch.FloatTensor(adv)
        ret_t = torch.FloatTensor(ret)
        old_logp_t = torch.FloatTensor(np.array(logp_b))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(EPOCHS):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), BATCH):
                b_idx = idx[s:s + BATCH]
                mean, val = model(o_t[b_idx])
                dist = Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp_t[b_idx])
                surr1 = ratio * adv_t[b_idx]
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * adv_t[b_idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(val.squeeze(-1), ret_t[b_idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

    elapsed = time.time() - t0
    model.eval()
    return model, elapsed


def eval_policy(model, prices):
    T, N = prices.shape
    initial_wealth = 10000.0
    cash = initial_wealth * 0.05
    init_p = prices[30]
    shares = (initial_wealth * 0.95 / N) / init_p
    peak = initial_wealth
    eq = [initial_wealth]

    obs_h = []
    for t in range(30):
        p, pp = prices[t], prices[max(0, t - 1)]
        w = max(1e-4, cash + np.sum(shares * p))
        obs_h.append(np.concatenate([
            p / prices[30],
            np.log(p / np.maximum(1e-4, pp)),
            [cash / w, np.clip((w - peak) / max(1e-4, peak), -1, 0)]
        ]).astype(np.float32))

    for t in range(30, T):
        flat_obs = np.concatenate(obs_h).astype(np.float32)
        act = model.get_action(flat_obs)

        cash_logit = np.clip(act[0], -5.0, 5.0)
        tc = 1.0 / (1.0 + np.exp(-cash_logit))
        ts = 1.0 - tc
        ea = np.exp(act[1:] - np.max(act[1:]))
        taw = (ea / ea.sum()) * ts

        p = prices[t].copy()
        w = max(1e-4, cash + np.sum(shares * p))
        caw = (shares * p) / w
        ccf = cash / w

        if abs(ccf - tc) + np.sum(np.abs(caw - taw)) > 0.03:
            tv = abs(cash - w * tc) + np.sum(np.abs(shares * p - w * taw))
            net = max(1e-4, w - tv * 0.001)
            cash = net * tc
            shares = (net * taw) / np.maximum(1e-4, p)

        nw = cash + np.sum(shares * prices[t])
        peak = max(peak, nw)
        eq.append(nw)

        pp = prices[t - 1]
        obs_h.pop(0)
        obs_h.append(np.concatenate([
            prices[t] / prices[30],
            np.log(prices[t] / np.maximum(1e-4, pp)),
            [cash / max(1e-4, nw), np.clip((nw - peak) / max(1e-4, peak), -1, 0)]
        ]).astype(np.float32))

    eq_a = np.array(eq)
    r = np.diff(eq_a) / np.maximum(1e-8, eq_a[:-1])
    pk = np.maximum.accumulate(eq_a)
    return {
        "final": float(eq_a[-1]),
        "return_pct": float((eq_a[-1] / eq_a[0] - 1) * 100),
        "sharpe": float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 1e-8 else 0.,
        "max_dd_pct": float(np.min((eq_a - pk) / pk) * 100)
    }


def main():
    W = 100
    print("=" * W)
    print("  RAI v7 vs RAI v6: REAL-DATA BENCHMARK EVALUATION")
    print("=" * W)
    print("  Evaluating G6 Jump-Diffusion + Spatio-Temporal Transformer (v7) vs Baseline (v6)")
    print(f"  Training Budget : {TOTAL_STEPS:,} steps per model | Seeds: {N_SEEDS} per model", flush=True)

    print("\n  Downloading real test market data...", flush=True)
    df = yf.download(TICKERS, start="2007-01-01", end="2026-08-08", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    df = df[TICKERS].dropna()
    prices = df.values

    n_total = len(prices)
    n_train = int(n_total * 0.70)
    test_prices = prices[n_train:]
    print(f"  Evaluation Set  : {len(test_prices)} OOS trading days ({df.index[n_train].date()} → {df.index[-1].date()})\n")

    # 1. Train RAI v6 models
    print(f"{'─'*W}")
    print(f"  Training {N_SEEDS} seeds of RAI v6 (Standard Synthetic + Conv1D+Transformer)...")
    print(f"{'─'*W}", flush=True)
    v6_models = []
    for s in range(1, N_SEEDS + 1):
        env_v6 = RawPriceSyntheticEnv(num_assets=10, episode_len=504)
        m_v6 = DeepEndToEndTradingNet()
        m_v6, elapsed = train_proper_ppo(m_v6, env_v6, seed=s)
        v6_models.append(m_v6)
        print(f"    ✓ RAI v6 Seed {s}/{N_SEEDS} trained in {elapsed:.0f}s", flush=True)

    # 2. Train RAI v7 models
    print(f"\n{'─'*W}")
    print(f"  Training {N_SEEDS} seeds of RAI v7 (G6 Jump-Diffusion + Spatio-Temporal Transformer)...")
    print(f"{'─'*W}", flush=True)
    v7_models = []
    for s in range(1, N_SEEDS + 1):
        env_v7 = G6EnhancedSyntheticEnv(num_assets=10, episode_len=504)
        m_v7 = SpatioTemporalTradingNet()
        m_v7, elapsed = train_proper_ppo(m_v7, env_v7, seed=s)
        v7_models.append(m_v7)
        print(f"    ✓ RAI v7 Seed {s}/{N_SEEDS} trained in {elapsed:.0f}s", flush=True)

    # 3. Evaluate on OOS Real Market Data
    print(f"\n{'─'*W}")
    print(f"  Evaluating both ensembles on unseen real market test data ({len(test_prices)} days)...")
    print(f"{'─'*W}", flush=True)

    v6_evals = [eval_policy(m, test_prices) for m in v6_models]
    v7_evals = [eval_policy(m, test_prices) for m in v7_models]

    v6_rets = np.array([r['return_pct'] for r in v6_evals])
    v6_shs = np.array([r['sharpe'] for r in v6_evals])
    v6_dds = np.array([r['max_dd_pct'] for r in v6_evals])

    v7_rets = np.array([r['return_pct'] for r in v7_evals])
    v7_shs = np.array([r['sharpe'] for r in v7_evals])
    v7_dds = np.array([r['max_dd_pct'] for r in v7_evals])

    print(f"\n{'═'*W}")
    print("  EMPIRICAL BENCHMARK RESULTS: RAI v6 vs RAI v7")
    print(f"{'═'*W}")
    print(f"  {'Metric':<30} | {'RAI v6 (Baseline)':<30} | {'RAI v7 (G6 Enhanced)':<30} | Improvement")
    print(f"  {'-'*98}")
    print(f"  {'Return (%) Mean ± SD':<30} | {np.mean(v6_rets):>+8.2f} ± {np.std(v6_rets):<5.2f}%         | {np.mean(v7_rets):>+8.2f} ± {np.std(v7_rets):<5.2f}%         | {np.mean(v7_rets)-np.mean(v6_rets):>+6.2f}%")
    print(f"  {'Sharpe Ratio Mean ± SD':<30} | {np.mean(v6_shs):>8.3f} ± {np.std(v6_shs):<5.3f}          | {np.mean(v7_shs):>8.3f} ± {np.std(v7_shs):<5.3f}          | {np.mean(v7_shs)-np.mean(v6_shs):>+6.3f}")
    print(f"  {'Max Drawdown (%) Mean ± SD':<30} | {np.mean(v6_dds):>+8.2f} ± {np.std(v6_dds):<5.2f}%         | {np.mean(v7_dds):>+8.2f} ± {np.std(v7_dds):<5.2f}%         | {abs(np.mean(v6_dds))-abs(np.mean(v7_dds)):>+6.2f}% lower DD")
    print(f"  {'-'*98}")

    output = {
        "v6_baseline": {"mean_return": float(np.mean(v6_rets)), "mean_sharpe": float(np.mean(v6_shs)), "mean_max_dd": float(np.mean(v6_dds))},
        "v7_enhanced": {"mean_return": float(np.mean(v7_rets)), "mean_sharpe": float(np.mean(v7_shs)), "mean_max_dd": float(np.mean(v7_dds))},
    }
    with open(os.path.join(RESULTS_DIR, "v7_vs_v6_results.json"), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✅ Benchmark complete. Saved to: {RESULTS_DIR}/v7_vs_v6_results.json\n")


if __name__ == "__main__":
    main()
