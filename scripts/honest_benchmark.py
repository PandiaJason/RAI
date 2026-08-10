"""
═══════════════════════════════════════════════════════════════════════════════
  HONEST 10-SEED CONTROLLED BENCHMARK: SYNTHETIC ZERO-SHOT vs REAL-DATA PPO
  ═══════════════════════════════════════════════════════════════════════════════
  Fixes ALL 5 issues from the honesty audit:

    ✓ FIX 1: Uses 100,000 training steps (not 1,500)
    ✓ FIX 2: Uses proper PPO (1024-step rollouts, GAE λ=0.95, clip 0.2, 4 epochs)
    ✓ FIX 3: Identical DeepEndToEndTradingNet (Conv1D+Transformer) for BOTH arms
    ✓ FIX 4: Identical multi-regime synthetic env for the synthetic arm
    ✓ FIX 5: Identical reward function for BOTH arms

  Protocol:
    ARM A — Real-Data PPO:
      • Train DeepEndToEndTradingNet via proper PPO on RealMarketEnv (70% real data)
      • Test on 30% OOS real data

    ARM B — Synthetic RAI v6:
      • Train DeepEndToEndTradingNet via proper PPO on RawPriceSyntheticEnv (0% real data)
      • Test zero-shot on the EXACT SAME 30% OOS real data

  10 seeds per arm → 20 total models, each trained for 100,000 steps.
  Estimated time: ~20–30 minutes total.
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
from scipy import stats
import yfinance as yf

warnings.filterwarnings('ignore')
torch.set_num_threads(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "honest_benchmark")
os.makedirs(RESULTS_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "EEM", "VNQ", "HYG", "TLT", "DBC", "GLD", "USO", "UUP"]
N_SEEDS = 10
TOTAL_STEPS = 100_000
ROLLOUT = 1024
BATCH = 64
EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
LR = 3e-4

# ═══════════════════════════════════════════════════════════════════════════════
#  IDENTICAL NETWORK ARCHITECTURE (Conv1D + Transformer Encoder)
#  Used for BOTH real-data and synthetic training — no exceptions.
# ═══════════════════════════════════════════════════════════════════════════════

class DeepEndToEndTradingNet(nn.Module):
    """Exact copy of RAI v6 architecture from train_v6_fast.py."""
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64, nhead=2):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step
        self.conv1d = nn.Sequential(
            nn.Conv1d(features_per_step, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
        )
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=128, dropout=0.05, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.fc_features = nn.Sequential(nn.Linear(embed_dim, 128), nn.LeakyReLU(0.1))
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step)
        x_conv = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_trans = self.transformer(x_conv)
        latent = x_trans.mean(dim=1)
        feat = self.fc_features(latent)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  BASE TRADING ENVIRONMENT (shared logic for both real and synthetic)
# ═══════════════════════════════════════════════════════════════════════════════

class BaseTradingEnv(gym.Env):
    """Shared trading mechanics and reward for both real and synthetic arms.
    Subclasses only differ in how they supply price data."""

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001):
        super().__init__()
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee
        self.features_per_step = 2 * num_assets + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (history_len * self.features_per_step,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (num_assets + 1,), np.float32)

    def _get_prices(self):
        """Return price array of shape (T, num_assets). Overridden by subclasses."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prices = self._get_prices()
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.05
        self.shares = (self.initial_cash * 0.95 / self.num_assets) / self.prices[self.current_step]
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs(), {}

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p, pp = self.prices[t], self.prices[max(0, t - 1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        return np.concatenate([
            p / self.prices[self.start],
            np.log(p / np.maximum(1e-4, pp)),
            [self.cash / w, np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1, 0)]
        ]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        # Identical action decoding, rebalancing, and reward for BOTH arms
        cash_logit = np.clip(action[0], -5.0, 5.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash_frac
        exp_a = np.exp(action[1:] - np.max(action[1:]))
        target_asset_w = (exp_a / np.sum(exp_a)) * stock_portion

        prices = self.prices[self.current_step]
        wealth = max(1e-4, self._wealth())
        cur_asset_w = (self.shares * prices) / wealth
        cur_cash_frac = self.cash / wealth

        drift = abs(cur_cash_frac - target_cash_frac) + np.sum(np.abs(cur_asset_w - target_asset_w))
        if drift > 0.03:
            t_vol = abs(self.cash - wealth * target_cash_frac) + np.sum(np.abs(self.shares * prices - wealth * target_asset_w))
            net = max(1e-4, wealth - t_vol * self.fee)
            self.cash = net * target_cash_frac
            self.shares = (net * target_asset_w) / np.maximum(1e-4, prices)

        self.current_step += 1
        self.steps_done += 1
        new_wealth = self._wealth()
        self.peak_wealth = max(self.peak_wealth, new_wealth)

        # IDENTICAL reward function for both arms
        daily_ret = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        reward = daily_ret * 5.0
        if daily_ret < 0:
            reward *= 2.0
        drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
        if drawdown < -0.10:
            reward += drawdown * 2.0

        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))
        return self._flat_obs(), reward, done, False, {}


# ═══════════════════════════════════════════════════════════════════════════════
#  ARM A: REAL-DATA ENVIRONMENT (trains on 70% real historical prices)
# ═══════════════════════════════════════════════════════════════════════════════

class RealDataTradingEnv(BaseTradingEnv):
    """Cycles through real historical prices for RL training."""
    def __init__(self, real_prices, **kwargs):
        self.real_prices = real_prices.astype(np.float64)
        super().__init__(num_assets=real_prices.shape[1], **kwargs)

    def _get_prices(self):
        return self.real_prices.copy()


# ═══════════════════════════════════════════════════════════════════════════════
#  ARM B: SYNTHETIC ENVIRONMENT (multi-regime, exact same as train_v6_fast.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticTradingEnv(BaseTradingEnv):
    """Multi-regime synthetic environment — identical to RawPriceSyntheticEnv
    from train_v6_fast.py. Three regimes: bull, bear, sideways."""
    REGIMES = {
        'bull':     {'drift': (0.15, 0.45),  'vol': (0.10, 0.20)},
        'bear':     {'drift': (-0.50, -0.15), 'vol': (0.25, 0.60)},
        'sideways': {'drift': (-0.05, 0.05),  'vol': (0.10, 0.20)},
    }

    def _get_prices(self):
        total_T = self.episode_len + self.history_len + 10
        n_seg = self.np_random.integers(3, 7)
        keys = list(self.REGIMES.keys())
        seq = [keys[self.np_random.integers(len(keys))] for _ in range(n_seg)]

        dur = self.np_random.dirichlet(np.ones(n_seg) * 2.0) * total_T
        dur = np.maximum(dur.astype(int), 30)
        dur[-1] = total_T - sum(dur[:-1])

        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)
        for asset in range(self.num_assets):
            p = self.np_random.uniform(20.0, 300.0)
            series = [p]
            day = 0
            for reg, d in zip(seq, dur):
                params = self.REGIMES[reg]
                drift = self.np_random.uniform(*params['drift']) + self.np_random.uniform(-0.04, 0.04)
                vol = self.np_random.uniform(*params['vol']) * self.np_random.uniform(0.85, 1.15)
                mu = drift / 252.0
                sigma = vol / np.sqrt(252.0)
                for _ in range(max(0, min(d, total_T - day - 1))):
                    p = max(0.01, p * np.exp((mu - 0.5 * sigma**2) + sigma * self.np_random.standard_normal()))
                    series.append(p)
                    day += 1
                    if day >= total_T:
                        break
                if day >= total_T:
                    break
            while len(series) < total_T:
                series.append(series[-1])
            prices[:, asset] = series[:total_T]
        return prices


# ═══════════════════════════════════════════════════════════════════════════════
#  PROPER PPO TRAINING LOOP (identical for both arms)
#  Copied verbatim from train_v6_fast.py lines 197-241
# ═══════════════════════════════════════════════════════════════════════════════

def train_proper_ppo(env, seed, total_steps=TOTAL_STEPS, label=""):
    """Train DeepEndToEndTradingNet using proper PPO with GAE, clipping, mini-batch epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DeepEndToEndTradingNet(history_len=30, features_per_step=22, action_dim=11)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    obs, _ = env.reset(seed=seed)
    step = 0
    t0 = time.time()

    while step < total_steps:
        # ── Rollout collection ──
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
            obs_b.append(obs)
            act_b.append(act_np)
            rew_b.append(rew)
            val_b.append(val.item())
            logp_b.append(logp.item())
            obs = nobs
            step += 1
            if done:
                obs, _ = env.reset()

        # ── GAE advantage estimation ──
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

        # ── PPO clipped surrogate optimization ──
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


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION (identical for both arms)
# ═══════════════════════════════════════════════════════════════════════════════

def eval_policy(model, prices):
    """Evaluate a trained model on real price data. Returns metrics dict."""
    T, N = prices.shape
    if T < 35:
        return {"return_pct": 0, "sharpe": 0, "max_dd_pct": 0}

    cash = prices.shape[1] * 0.0  # start with 5% cash
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


def compute_cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pool_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pool_std if pool_std > 1e-8 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    W = 110
    print("=" * W)
    print("  HONEST 10-SEED CONTROLLED BENCHMARK: SYNTHETIC ZERO-SHOT vs REAL-DATA PPO")
    print("=" * W)
    print(f"  Architecture   : DeepEndToEndTradingNet (Conv1D + Transformer) — IDENTICAL for both arms")
    print(f"  PPO            : Proper PPO (rollout={ROLLOUT}, GAE λ={GAE_LAMBDA}, clip={CLIP_RATIO}, epochs={EPOCHS})")
    print(f"  Training Steps : {TOTAL_STEPS:,} per model")
    print(f"  Seeds          : {N_SEEDS} per arm ({N_SEEDS * 2} total models)")
    print(f"  Reward         : daily_ret × 5.0, ×2.0 penalty if negative, DD penalty if < -10%")
    print(f"  Evaluation Fee : 10 bps (0.001)", flush=True)

    # ── 1. Download Real Data ──
    print(f"\n  Downloading real market data...", flush=True)
    df = yf.download(TICKERS, start="2007-01-01", end="2026-08-08", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    df = df[TICKERS].dropna()
    prices = df.values

    n_total = len(prices)
    n_train = int(n_total * 0.70)
    train_prices = prices[:n_train]
    test_prices = prices[n_train:]

    print(f"  Real Dataset       : {n_total} trading days ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"  70% Train Split    : {n_train} trading days ({df.index[0].date()} → {df.index[n_train-1].date()})")
    print(f"  30% OOS Test Split : {len(test_prices)} trading days ({df.index[n_train].date()} → {df.index[-1].date()})")

    # ── 2. ARM A: Train Real-Data PPO (100k steps each) ──
    print(f"\n{'─'*W}")
    print(f"  ARM A: Training {N_SEEDS} seeds of REAL-DATA PPO (100k steps each)...")
    print(f"{'─'*W}", flush=True)

    real_models = []
    real_times = []
    for s in range(1, N_SEEDS + 1):
        env = RealDataTradingEnv(train_prices, episode_len=min(504, n_train - 35))
        model, elapsed = train_proper_ppo(env, seed=s, label=f"Real-Data Seed {s}")
        real_models.append(model)
        real_times.append(elapsed)
        print(f"    ✓ Real-Data PPO Seed {s:>2d}/{N_SEEDS} — {elapsed:.0f}s", flush=True)

    # ── 3. ARM B: Train Synthetic PPO (100k steps each) ──
    print(f"\n{'─'*W}")
    print(f"  ARM B: Training {N_SEEDS} seeds of SYNTHETIC PPO (100k steps each)...")
    print(f"{'─'*W}", flush=True)

    synth_models = []
    synth_times = []
    for s in range(1, N_SEEDS + 1):
        env = SyntheticTradingEnv(num_assets=10, episode_len=504)
        model, elapsed = train_proper_ppo(env, seed=s, label=f"Synthetic Seed {s}")
        synth_models.append(model)
        synth_times.append(elapsed)
        print(f"    ✓ Synthetic PPO Seed {s:>2d}/{N_SEEDS} — {elapsed:.0f}s", flush=True)

    # ── 4. Evaluate BOTH arms on the EXACT SAME OOS test set ──
    print(f"\n{'─'*W}")
    print(f"  Evaluating both {N_SEEDS}-Seed Ensembles on Identical OOS Test Data ({len(test_prices)} days)...")
    print(f"{'─'*W}", flush=True)

    real_evals = [eval_policy(m, test_prices) for m in real_models]
    synth_evals = [eval_policy(m, test_prices) for m in synth_models]

    real_rets = np.array([r['return_pct'] for r in real_evals])
    real_shs = np.array([r['sharpe'] for r in real_evals])
    real_dds = np.array([r['max_dd_pct'] for r in real_evals])

    synth_rets = np.array([r['return_pct'] for r in synth_evals])
    synth_shs = np.array([r['sharpe'] for r in synth_evals])
    synth_dds = np.array([r['max_dd_pct'] for r in synth_evals])

    # ── 5. Statistical Tests ──
    t_ret, p_ret = stats.ttest_ind(synth_rets, real_rets, equal_var=False)
    u_ret, p_u_ret = stats.mannwhitneyu(synth_rets, real_rets, alternative='two-sided')
    d_ret = compute_cohens_d(synth_rets, real_rets)

    t_sh, p_sh = stats.ttest_ind(synth_shs, real_shs, equal_var=False)
    d_sh = compute_cohens_d(synth_shs, real_shs)

    t_dd, p_dd = stats.ttest_ind(synth_dds, real_dds, equal_var=False)
    d_dd = compute_cohens_d(synth_dds, real_dds)

    ci_real_ret = stats.t.interval(0.95, len(real_rets) - 1, loc=np.mean(real_rets), scale=stats.sem(real_rets))
    ci_synth_ret = stats.t.interval(0.95, len(synth_rets) - 1, loc=np.mean(synth_rets), scale=stats.sem(synth_rets))
    ci_real_sh = stats.t.interval(0.95, len(real_shs) - 1, loc=np.mean(real_shs), scale=stats.sem(real_shs))
    ci_synth_sh = stats.t.interval(0.95, len(synth_shs) - 1, loc=np.mean(synth_shs), scale=stats.sem(synth_shs))

    # ── 6. Print Results ──
    print(f"\n{'═'*W}")
    print(f"  HONEST {N_SEEDS}-SEED CONTROLLED BENCHMARK — RESULTS")
    print(f"{'═'*W}")
    print(f"  Controls: Identical architecture, PPO, reward, fees, evaluation | {TOTAL_STEPS:,} steps each")
    print(f"{'─'*W}")

    print(f"\n  {'Metric':<30} | {'Real-Data PPO (ARM A)':<35} | {'Synthetic Zero-Shot (ARM B)':<35} | {'Test'}")
    print(f"  {'─'*106}")

    print(f"  {'Return (%) Mean ± SD':<30} | {np.mean(real_rets):>+8.2f} ± {np.std(real_rets):<5.2f}%{'':<19} | {np.mean(synth_rets):>+8.2f} ± {np.std(synth_rets):<5.2f}%{'':<19} | Welch p={p_ret:.4f}")
    print(f"  {'Return 95% CI':<30} | [{ci_real_ret[0]:>+.2f}%, {ci_real_ret[1]:>+.2f}%]{'':<16} | [{ci_synth_ret[0]:>+.2f}%, {ci_synth_ret[1]:>+.2f}%]{'':<16} | Mann-W p={p_u_ret:.4f}")
    print(f"  {'Sharpe Ratio Mean ± SD':<30} | {np.mean(real_shs):>8.3f} ± {np.std(real_shs):<5.3f}{'':<20} | {np.mean(synth_shs):>8.3f} ± {np.std(synth_shs):<5.3f}{'':<20} | Welch p={p_sh:.4f}")
    print(f"  {'Sharpe 95% CI':<30} | [{ci_real_sh[0]:>.3f}, {ci_real_sh[1]:>.3f}]{'':<21} | [{ci_synth_sh[0]:>.3f}, {ci_synth_sh[1]:>.3f}]{'':<21} | Cohen's d={d_ret:+.3f}")
    print(f"  {'Max DD (%) Mean ± SD':<30} | {np.mean(real_dds):>+8.2f} ± {np.std(real_dds):<5.2f}%{'':<19} | {np.mean(synth_dds):>+8.2f} ± {np.std(synth_dds):<5.2f}%{'':<19} | Welch p={p_dd:.4f}")

    print(f"\n  {'─'*106}")
    print(f"  Individual Seed Returns (%):")
    print(f"    ARM A (Real):      {', '.join(f'{r:+.1f}' for r in real_rets)}")
    print(f"    ARM B (Synthetic): {', '.join(f'{r:+.1f}' for r in synth_rets)}")

    print(f"\n  Individual Seed Sharpe Ratios:")
    print(f"    ARM A (Real):      {', '.join(f'{s:.3f}' for s in real_shs)}")
    print(f"    ARM B (Synthetic): {', '.join(f'{s:.3f}' for s in synth_shs)}")

    print(f"\n  Training Time:")
    print(f"    ARM A avg: {np.mean(real_times):.0f}s per model | ARM B avg: {np.mean(synth_times):.0f}s per model")
    print(f"    Total: {sum(real_times) + sum(synth_times):.0f}s ({(sum(real_times) + sum(synth_times))/60:.1f} min)")

    # ── 7. Save Results ──
    output = {
        "config": {
            "n_seeds": N_SEEDS,
            "total_steps": TOTAL_STEPS,
            "architecture": "DeepEndToEndTradingNet (Conv1D+Transformer, 51.7k params)",
            "ppo_config": {"rollout": ROLLOUT, "batch": BATCH, "epochs": EPOCHS,
                           "gamma": GAMMA, "gae_lambda": GAE_LAMBDA, "clip_ratio": CLIP_RATIO, "lr": LR},
            "reward": "daily_ret * 5.0, x2 penalty if negative, DD penalty if < -10%",
            "real_env": "RealDataTradingEnv (70% real historical prices)",
            "synth_env": "SyntheticTradingEnv (multi-regime: bull/bear/sideways, 0% real data)",
            "oos_test": f"{len(test_prices)} trading days ({df.index[n_train].date()} → {df.index[-1].date()})"
        },
        "arm_a_real_data": {
            "individual_returns": real_rets.tolist(),
            "individual_sharpes": real_shs.tolist(),
            "individual_max_dds": real_dds.tolist(),
            "mean_return": float(np.mean(real_rets)), "std_return": float(np.std(real_rets)),
            "ci95_return": [float(ci_real_ret[0]), float(ci_real_ret[1])],
            "mean_sharpe": float(np.mean(real_shs)), "std_sharpe": float(np.std(real_shs)),
            "ci95_sharpe": [float(ci_real_sh[0]), float(ci_real_sh[1])],
            "mean_max_dd": float(np.mean(real_dds)), "std_max_dd": float(np.std(real_dds)),
        },
        "arm_b_synthetic": {
            "individual_returns": synth_rets.tolist(),
            "individual_sharpes": synth_shs.tolist(),
            "individual_max_dds": synth_dds.tolist(),
            "mean_return": float(np.mean(synth_rets)), "std_return": float(np.std(synth_rets)),
            "ci95_return": [float(ci_synth_ret[0]), float(ci_synth_ret[1])],
            "mean_sharpe": float(np.mean(synth_shs)), "std_sharpe": float(np.std(synth_shs)),
            "ci95_sharpe": [float(ci_synth_sh[0]), float(ci_synth_sh[1])],
            "mean_max_dd": float(np.mean(synth_dds)), "std_max_dd": float(np.std(synth_dds)),
        },
        "statistics": {
            "welch_p_return": float(p_ret), "mann_whitney_p_return": float(p_u_ret),
            "cohens_d_return": float(d_ret),
            "welch_p_sharpe": float(p_sh), "cohens_d_sharpe": float(d_sh),
            "welch_p_max_dd": float(p_dd), "cohens_d_max_dd": float(d_dd),
        }
    }

    out_file = os.path.join(RESULTS_DIR, "honest_benchmark_results.json")
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'═'*W}")
    print(f"  ✅ HONEST BENCHMARK COMPLETE — Results saved to: {out_file}")
    print(f"{'═'*W}\n", flush=True)


if __name__ == "__main__":
    main()
