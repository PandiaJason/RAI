"""
═══════════════════════════════════════════════════════════════════════════════
  FAST PARALLEL MULTI-DATASET MASTER BENCHMARK
  ═══════════════════════════════════════════════════════════════════════════════
  Runs parallel CPU multiprocessing across 3 distinct real asset classes:
    1. US Market ETFs
    2. US Mega-Cap Stocks
    3. Global Equity Indices

  Model Arms (Seeded & Trained on Parallel CPU Cores):
    1. Industry LSTM-DNN (PyTorch LSTM, Trained 70% Real Data)
    2. Real-Data Trained PPO (v5 Conv1D Arch, Trained 70% Real Data)
    3. Real-Data Trained PPO (v7 Spatio-Temporal Arch, Trained 70% Real Data)
    4. RAI v6 Zero-Shot (Trained 0% Real Data / Standard Synthetic)
    5. RAI v7 Zero-Shot NEW (Trained 0% Real Data / G6 Jump-Diffusion)
    6. Equal Weight 1/N (Quantitative Baseline)
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, warnings
import concurrent.futures
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import yfinance as yf

warnings.filterwarnings('ignore')
torch.set_num_threads(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rai.world.v7_g6_env import G6EnhancedSyntheticEnv
from rai.learning.v7_model import SpatioTemporalTradingNet
from scripts.train_v6_fast import DeepEndToEndTradingNet, RawPriceSyntheticEnv
from scripts.honest_benchmark import BaseTradingEnv, RealDataTradingEnv

DATASETS = {
    "1. US Market ETFs": {
        "tickers": ["SPY", "QQQ", "EEM", "VNQ", "HYG", "TLT", "GLD", "USO", "UUP", "IWM"],
        "start": "2010-01-01", "end": "2026-08-08"
    },
    "2. US Mega-Cap Stocks": {
        "tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "LLY", "JPM", "JNJ", "WMT"],
        "start": "2015-01-01", "end": "2024-12-31"
    },
    "3. Global Equity Indices": {
        "tickers": ["SPY", "EWJ", "EWG", "EWU", "MCHI", "INDA", "EWZ", "EFA", "EEM", "FXI"],
        "start": "2015-01-01", "end": "2024-12-31"
    }
}

N_SEEDS = 3
TOTAL_STEPS = 30_000  # Fast execution across all datasets
ROLLOUT = 1024
BATCH = 64
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
LR = 3e-4

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "parallel_multi_dataset_benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)


class LSTMIndustryTradingNet(nn.Module):
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
        latent = lstm_out[:, -1, :]
        feat = self.fc(latent)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0) if deterministic else Normal(mean, torch.exp(self.log_std)).sample().cpu().numpy().squeeze(0)


def train_single_worker(args):
    d_name, model_type, seed, train_prices = args
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    if model_type == "lstm_real":
        env = RealDataTradingEnv(train_prices, episode_len=min(504, len(train_prices) - 35))
        model = LSTMIndustryTradingNet()
    elif model_type == "ppo_real":
        env = RealDataTradingEnv(train_prices, episode_len=min(504, len(train_prices) - 35))
        model = DeepEndToEndTradingNet()
    elif model_type == "ppo_v7_real":
        env = RealDataTradingEnv(train_prices, episode_len=min(504, len(train_prices) - 35))
        model = SpatioTemporalTradingNet()
    elif model_type == "rai_v6":
        env = RawPriceSyntheticEnv(num_assets=10, episode_len=504)
        model = DeepEndToEndTradingNet()
    elif model_type == "rai_v7":
        env = G6EnhancedSyntheticEnv(num_assets=10, episode_len=504)
        model = SpatioTemporalTradingNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    obs, _ = env.reset(seed=seed)
    step = 0

    while step < TOTAL_STEPS:
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

    model.eval()
    return d_name, model_type, seed, model


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
    n_cores = os.cpu_count() or 4
    W = 120
    print("=" * W)
    print(f"  FAST PARALLEL MULTI-DATASET MASTER BENCHMARK ON {n_cores} CPU CORES")
    print("=" * W)

    dataset_prices = {}
    tasks = []
    model_types = ["lstm_real", "ppo_real", "ppo_v7_real", "rai_v6", "rai_v7"]

    for d_name, d_info in DATASETS.items():
        df = fetch_dataset_prices(d_info['tickers'], d_info['start'], d_info['end'])
        prices = df.values
        n_total = len(prices)
        n_train = int(n_total * 0.70)
        train_prices = prices[:n_train]
        test_prices = prices[n_train:]
        dataset_prices[d_name] = {"full": df, "prices": prices, "train": train_prices, "test": test_prices}

        for mtype in model_types:
            for seed in range(1, N_SEEDS + 1):
                tasks.append((d_name, mtype, seed, train_prices))

    print(f"  Loaded {len(DATASETS)} real market datasets.")
    print(f"  Dispatching {len(tasks)} training tasks across {n_cores} parallel CPU workers...", flush=True)
    t0 = time.time()

    trained_results = {d_name: {mtype: [] for mtype in model_types} for d_name in DATASETS}

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = [executor.submit(train_single_worker, task) for task in tasks]
        for f in concurrent.futures.as_completed(futures):
            d_name, mtype, seed, model = f.result()
            trained_results[d_name][mtype].append(model)
            print(f"    ✓ [{d_name[:15]}] Trained {mtype} (Seed {seed})", flush=True)

    elapsed_train = time.time() - t0
    print(f"\n  ✅ All {len(tasks)} models trained in parallel in {elapsed_train:.1f} seconds! ({elapsed_train/len(tasks):.1f}s / model)\n", flush=True)

    master_summary = {}

    labels = {
        "lstm_real": ("Industry LSTM-DNN (PyTorch)", "YES (70% Real)"),
        "ppo_real": ("Real-Data Trained PPO (v5 Arch)", "YES (70% Real)"),
        "ppo_v7_real": ("Real-Data Trained PPO (v7 Arch)", "YES (70% Real)"),
        "rai_v6": ("RAI v6 Zero-Shot", "NO (0% Real)"),
        "rai_v7": ("🏆 RAI v7 Zero-Shot (NEW G6 Jump)", "NO (0% Real)"),
    }

    for d_name, d_data in dataset_prices.items():
        test_prices = d_data['test']
        print(f"\n{'═'*W}")
        print(f"  MASTER EVALUATION RESULTS: {d_name} ({len(test_prices)} OOS Trading Days)")
        print(f"{'═'*W}")
        print(f"  {'Model / Baseline Name':<44} | {'Real Trained?':<15} | {'OOS Return (%)':<22} | {'Sharpe Ratio':<18} | Max DD (%)")
        print(f"  {'-'*118}")

        d_eval = {}
        for mtype in model_types:
            evals = [eval_policy(m, test_prices) for m in trained_results[d_name][mtype]]
            rets = [r['return_pct'] for r in evals]
            shs = [r['sharpe'] for r in evals]
            dds = [r['max_dd_pct'] for r in evals]
            d_eval[mtype] = {
                "mean_ret": float(np.mean(rets)), "std_ret": float(np.std(rets)),
                "mean_sh": float(np.mean(shs)), "std_sh": float(np.std(shs)),
                "mean_dd": float(np.mean(dds)), "std_dd": float(np.std(dds)),
            }
            name, tr_status = labels[mtype]
            print(f"  {name:<44} | {tr_status:<15} | {np.mean(rets):>+7.2f} ± {np.std(rets):<5.2f}%        | {np.mean(shs):>6.2f} ± {np.std(shs):<4.2f}        | {np.mean(dds):>+6.2f}%")

        ew = eval_equal_weight(test_prices)
        d_eval["equal_weight"] = ew
        print(f"  {'Equal Weight 1/N Baseline':<44} | {'Passive':<15} | {ew['return_pct']:>+7.2f}%                  | {ew['sharpe']:>6.2f}               | {ew['max_dd_pct']:>+6.2f}%")
        print(f"  {'-'*118}")
        master_summary[d_name] = d_eval

    with open(os.path.join(RESULTS_DIR, "parallel_multi_dataset_benchmark_results.json"), 'w') as f:
        json.dump(master_summary, f, indent=2)

    print(f"\n  ✅ MULTI-DATASET MASTER BENCHMARK COMPLETE — Saved to: {RESULTS_DIR}/parallel_multi_dataset_benchmark_results.json\n")


if __name__ == "__main__":
    main()
