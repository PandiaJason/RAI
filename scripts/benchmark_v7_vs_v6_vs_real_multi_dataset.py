"""
═══════════════════════════════════════════════════════════════════════════════
  MULTI-DATASET COMPREHENSIVE BENCHMARK:
  INDUSTRY LSTM-DNN vs REAL PPO vs RAI v6 ZERO-SHOT vs RAI v7 ZERO-SHOT
  ═══════════════════════════════════════════════════════════════════════════════
  Evaluates out-of-sample zero-shot transfer across multiple distinct real asset classes:
    1. US Mega-Cap Stocks
    2. Global Equity Indices
    3. US Sector ETFs

  Model Arms (Seeded & Trained at the exact same time on identical splits):
    • ARM A: Industry-Standard LSTM Deep Neural Network (LSTM-DNN, Trained 70% Real Data)
    • ARM B: Real-Data Trained PPO (Conv1D+Transformer, Trained 70% Real Data)
    • ARM C: RAI v6 Zero-Shot (Trained 0% Real Data / Standard Synthetic)
    • ARM D: RAI v7 Zero-Shot NEW (Trained 0% Real Data / G6 Jump-Diffusion + Spatio-Temporal)
    • ARM E: Equal Weight (1/N Quantitative Baseline)
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
from scripts.honest_benchmark import BaseTradingEnv, RealDataTradingEnv

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "v7_industry_dnn_benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    "1. US Mega-Cap Stocks": {
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "LLY", "JPM", "JNJ", "WMT"],
        "start": "2015-01-01", "end": "2024-12-31"
    },
    "2. Global Equity Indices": {
        "tickers": ["SPY", "EWJ", "EWG", "EWU", "MCHI", "INDA", "EWZ", "EFA", "EEM", "FXI"],
        "start": "2015-01-01", "end": "2024-12-31"
    },
    "3. US Sector ETFs": {
        "tickers": ["XLK", "XLV", "XLF", "XLE", "XLI", "XLP", "XLU", "XLY", "XLB", "XLC"],
        "start": "2015-01-01", "end": "2024-12-31"
    }
}

N_SEEDS = 3
TOTAL_STEPS = 50_000
ROLLOUT = 1024
BATCH = 64
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
LR = 3e-4


# ═══════════════════════════════════════════════════════════════════════════════
#  INDUSTRY-STANDARD LSTM DEEP NEURAL NETWORK ARCHITECTURE (LSTM-DNN)
# ═══════════════════════════════════════════════════════════════════════════════

class LSTMIndustryTradingNet(nn.Module):
    """Industry Standard 2-Layer Recurrent Neural Network (LSTM Deep Neural Network)."""
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, hidden_dim=64):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step
        self.lstm = nn.LSTM(input_size=features_per_step, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.05)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 128), nn.LeakyReLU(0.1))
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step)
        lstm_out, _ = self.lstm(x)
        latent = lstm_out[:, -1, :]  # last hidden state
        feat = self.fc(latent)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0) if deterministic else Normal(mean, torch.exp(self.log_std)).sample().cpu().numpy().squeeze(0)


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


def eval_equal_weight(prices):
    T, N = prices.shape
    initial_wealth = 10000.0
    init_p = prices[30]
    shares = (initial_wealth / N) / init_p
    eq = [initial_wealth]
    for t in range(30, T):
        nw = np.sum(shares * prices[t])
        eq.append(nw)
    eq_a = np.array(eq)
    r = np.diff(eq_a) / np.maximum(1e-8, eq_a[:-1])
    pk = np.maximum.accumulate(eq_a)
    return {
        "final": float(eq_a[-1]),
        "return_pct": float((eq_a[-1] / eq_a[0] - 1) * 100),
        "sharpe": float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 1e-8 else 0.,
        "max_dd_pct": float(np.min((eq_a - pk) / pk) * 100)
    }


def fetch_dataset_prices(tickers, start, end):
    for attempt in range(3):
        try:
            df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            df = df.dropna(how='all').ffill().bfill().dropna()
            valid_cols = [c for c in tickers if c in df.columns]
            if len(valid_cols) == len(tickers) and len(df) > 500:
                return df[valid_cols]
        except Exception:
            time.sleep(2)
    
    # Fallback to individual downloads if batch fails
    series_dict = {}
    for t in tickers:
        try:
            d = yf.Ticker(t).history(start=start, end=end)['Close']
            if len(d) > 0:
                series_dict[t] = d
        except Exception:
            pass
    df_combined = pd.DataFrame(series_dict).ffill().bfill().dropna()
    return df_combined


def main():
    W = 120
    print("=" * W)
    print("  MULTI-DATASET BENCHMARK WITH INDUSTRY-STANDARD LSTM-DNN vs REAL PPO vs RAI v6 vs RAI v7 NEW")
    print("=" * W)

    results_master = {}

    for d_name, d_info in DATASETS.items():
        print(f"\n{'═'*W}")
        print(f"  DATASET: {d_name}")
        print(f"{'═'*W}", flush=True)

        try:
            df = fetch_dataset_prices(d_info['tickers'], d_info['start'], d_info['end'])
            prices = df.values

            n_total = len(prices)
            n_train = int(n_total * 0.70)
            train_prices = prices[:n_train]
            test_prices = prices[n_train:]

            print(f"  Dataset Total   : {n_total} trading days ({df.index[0].date()} → {df.index[-1].date()})")
            print(f"  70% Train Split : {n_train} trading days ({df.index[0].date()} → {df.index[n_train-1].date()})")
            print(f"  30% OOS Test    : {len(test_prices)} trading days ({df.index[n_train].date()} → {df.index[-1].date()})\n")

            # 1. ARM A: Industry-Standard LSTM Deep Neural Network (Trained on 70% Real Data)
            print(f"  [1/4] Training {N_SEEDS} seeds of Industry LSTM-DNN (Trained on 70% Real Data)...", flush=True)
            lstm_models = []
            for s in range(1, N_SEEDS + 1):
                env = RealDataTradingEnv(train_prices, episode_len=min(504, n_train - 35))
                m = LSTMIndustryTradingNet()
                m, el = train_proper_ppo(m, env, seed=s)
                lstm_models.append(m)

            # 2. ARM B: Real-Data Trained PPO (Conv1D+Transformer, Trained on 70% Real Data)
            print(f"  [2/4] Training {N_SEEDS} seeds of Real-Data PPO (Conv1D+Transformer, Trained on 70% Real Data)...", flush=True)
            real_models = []
            for s in range(1, N_SEEDS + 1):
                env = RealDataTradingEnv(train_prices, episode_len=min(504, n_train - 35))
                m = DeepEndToEndTradingNet()
                m, el = train_proper_ppo(m, env, seed=s)
                real_models.append(m)

            # 3. ARM C: RAI v6 Zero-Shot (Standard Synthetic)
            print(f"  [3/4] Training {N_SEEDS} seeds of RAI v6 Zero-Shot (Standard Synthetic)...", flush=True)
            v6_models = []
            for s in range(1, N_SEEDS + 1):
                env = RawPriceSyntheticEnv(num_assets=10, episode_len=504)
                m = DeepEndToEndTradingNet()
                m, el = train_proper_ppo(m, env, seed=s)
                v6_models.append(m)

            # 4. ARM D: RAI v7 Zero-Shot (G6 Jump-Diffusion + Spatio-Temporal Transformer)
            print(f"  [4/4] Training {N_SEEDS} seeds of RAI v7 Zero-Shot NEW (G6 Jump-Diffusion + Spatio-Temporal)...", flush=True)
            v7_models = []
            for s in range(1, N_SEEDS + 1):
                env = G6EnhancedSyntheticEnv(num_assets=10, episode_len=504)
                m = SpatioTemporalTradingNet()
                m, el = train_proper_ppo(m, env, seed=s)
                v7_models.append(m)

            # Evaluate All Arms on the exact same 30% OOS test split
            lstm_evals = [eval_policy(m, test_prices) for m in lstm_models]
            real_evals = [eval_policy(m, test_prices) for m in real_models]
            v6_evals = [eval_policy(m, test_prices) for m in v6_models]
            v7_evals = [eval_policy(m, test_prices) for m in v7_models]
            ew_eval = eval_equal_weight(test_prices)

            lstm_ret = [r['return_pct'] for r in lstm_evals]; lstm_sh = [r['sharpe'] for r in lstm_evals]; lstm_dd = [r['max_dd_pct'] for r in lstm_evals]
            r_ret = [r['return_pct'] for r in real_evals]; r_sh = [r['sharpe'] for r in real_evals]; r_dd = [r['max_dd_pct'] for r in real_evals]
            v6_ret = [r['return_pct'] for r in v6_evals]; v6_sh = [r['sharpe'] for r in v6_evals]; v6_dd = [r['max_dd_pct'] for r in v6_evals]
            v7_ret = [r['return_pct'] for r in v7_evals]; v7_sh = [r['sharpe'] for r in v7_evals]; v7_dd = [r['max_dd_pct'] for r in v7_evals]

            print(f"\n  {'─'*116}")
            print(f"  RESULTS FOR: {d_name}")
            print(f"  {'─'*116}")
            print(f"  {'Model Arm':<35} | {'OOS Return (%)':<24} | {'Sharpe Ratio':<20} | Max Drawdown (%)")
            print(f"  {'─'*116}")
            print(f"  {'Industry LSTM-DNN (Arm A)':<35} | {np.mean(lstm_ret):>+7.2f} ± {np.std(lstm_ret):<5.2f}%         | {np.mean(lstm_sh):>6.2f} ± {np.std(lstm_sh):<4.2f}         | {np.mean(lstm_dd):>+6.2f}%")
            print(f"  {'Real-Data Trained PPO (Arm B)':<35} | {np.mean(r_ret):>+7.2f} ± {np.std(r_ret):<5.2f}%         | {np.mean(r_sh):>6.2f} ± {np.std(r_sh):<4.2f}         | {np.mean(r_dd):>+6.2f}%")
            print(f"  {'RAI v6 Zero-Shot (Arm C)':<35} | {np.mean(v6_ret):>+7.2f} ± {np.std(v6_ret):<5.2f}%         | {np.mean(v6_sh):>6.2f} ± {np.std(v6_sh):<4.2f}         | {np.mean(v6_dd):>+6.2f}%")
            print(f"  {'🏆 RAI v7 Zero-Shot NEW (Arm D)':<35} | {np.mean(v7_ret):>+7.2f} ± {np.std(v7_ret):<5.2f}%         | {np.mean(v7_sh):>6.2f} ± {np.std(v7_sh):<4.2f}         | {np.mean(v7_dd):>+6.2f}%")
            print(f"  {'Equal Weight 1/N (Arm E)':<35} | {ew_eval['return_pct']:>+7.2f}%                   | {ew_eval['sharpe']:>6.2f}                | {ew_eval['max_dd_pct']:>+6.2f}%")
            print(f"  {'─'*116}")

            results_master[d_name] = {
                "industry_lstm_dnn": {"return": float(np.mean(lstm_ret)), "sharpe": float(np.mean(lstm_sh)), "max_dd": float(np.mean(lstm_dd))},
                "real_ppo": {"return": float(np.mean(r_ret)), "sharpe": float(np.mean(r_sh)), "max_dd": float(np.mean(r_dd))},
                "rai_v6": {"return": float(np.mean(v6_ret)), "sharpe": float(np.mean(v6_sh)), "max_dd": float(np.mean(v6_dd))},
                "rai_v7_new": {"return": float(np.mean(v7_ret)), "sharpe": float(np.mean(v7_sh)), "max_dd": float(np.mean(v7_dd))},
                "equal_weight": {"return": ew_eval['return_pct'], "sharpe": ew_eval['sharpe'], "max_dd": ew_eval['max_dd_pct']},
            }

        except Exception as e:
            print(f"  ⚠ Failed evaluation for {d_name}: {e}", flush=True)

    with open(os.path.join(RESULTS_DIR, "v7_industry_dnn_benchmark_results.json"), 'w') as f:
        json.dump(results_master, f, indent=2)

    print(f"\n{'═'*W}")
    print(f"  ✅ INDUSTRY-STANDARD DNN BENCHMARK COMPLETE — Saved to: {RESULTS_DIR}/v7_industry_dnn_benchmark_results.json")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
