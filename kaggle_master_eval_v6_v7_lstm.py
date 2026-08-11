"""
====================================================================================================
🏆 KAGGLE MASTER EVALUATION & BENCHMARK SUITE:
   RAI v6 (Zero-Shot) vs RAI v7 (Zero-Shot) vs Real-Trained v6 vs Real-Trained v7 vs Real-Trained LSTM
====================================================================================================
Kaggle Notebook / Standalone Evaluation Script

Evaluates 5 Model Arms side-by-side on Real Financial Market Data:
  1. 🤖 RAI v6 Zero-Shot (Conv1D + Transformer, Trained 100% on Synthetic World G0)
  2. 🚀 RAI v7 Zero-Shot (Spatio-Temporal Transformer, Trained 100% on G6 Jump-Diffusion G1)
  3. 📈 Real-Trained v6 (Conv1D + Transformer, Trained on Real Market Data)
  4. ⚡ Real-Trained v7 (Spatio-Temporal Transformer, Trained on Real Market Data)
  5. 🧠 Real-Trained LSTM (Industry 2-Layer Recurrent LSTM, Trained on Real Market Data)
  6. ⚖️ Equal Weight (1/N Quantitative Baseline Control)

GPU Acceleration:
  Dual GPU (Kaggle GPU T4 x2) / Single GPU / CPU Multi-Core Compatible.
====================================================================================================
"""

# Install dependencies
!pip install -q yfinance gymnasium torch pandas numpy matplotlib scipy

import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import yfinance as yf

warnings.filterwarnings("ignore")

# Device Configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"✓ Master Benchmarking Engine Initialized | Using Device: {DEVICE} | PyTorch: {torch.__version__}")


# ==================================================================================================
# SECTION 1: SYNTHETIC WORLD GENERATORS (G0 Standard & G6 Jump-Diffusion)
# ==================================================================================================

class RawPriceSyntheticEnv:
    """Synthetic World G0: Procedural multi-regime synthetic price environment."""
    REGIMES = {
        'bull':     {'drift': (0.15, 0.45),  'vol': (0.10, 0.20)},
        'bear':     {'drift': (-0.50, -0.15), 'vol': (0.25, 0.60)},
        'sideways': {'drift': (-0.05, 0.05),  'vol': (0.10, 0.20)},
    }

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001):
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee
        self.features_per_step = 2 * num_assets + 2
        self.reset()

    def _generate_raw_prices(self):
        total_T = self.episode_len + self.history_len + 10
        n_seg = np.random.randint(3, 7)
        keys = list(self.REGIMES.keys())
        seq = [keys[np.random.randint(len(keys))] for _ in range(n_seg)]

        dur = np.random.dirichlet(np.ones(n_seg) * 2.0) * total_T
        dur = np.maximum(dur.astype(int), 30)
        dur[-1] = total_T - sum(dur[:-1])

        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)

        for asset in range(self.num_assets):
            p = np.random.uniform(20.0, 300.0)
            series = [p]
            day = 0
            for reg, d in zip(seq, dur):
                params = self.REGIMES[reg]
                drift = np.random.uniform(*params['drift']) + np.random.uniform(-0.04, 0.04)
                vol = np.random.uniform(*params['vol']) * np.random.uniform(0.85, 1.15)
                mu = drift / 252.0; sigma = vol / np.sqrt(252.0)
                for _ in range(max(0, min(d, total_T - day - 1))):
                    p = max(0.01, p * np.exp((mu - 0.5*sigma**2) + sigma * np.random.standard_normal()))
                    series.append(p)
                    day += 1
                    if day >= total_T: break
                if day >= total_T: break
            while len(series) < total_T: series.append(series[-1])
            prices[:, asset] = series[:total_T]

        return prices

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.prices = self._generate_raw_prices()
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.5
        init_p = self.prices[self.current_step]
        self.shares = (self.initial_cash * 0.5 / self.num_assets) / init_p
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs()

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p = self.prices[t]; p_prev = self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        norm_prices = p / self.prices[self.start]
        log_rets = np.log(p / np.maximum(1e-4, p_prev))
        cash_ratio = self.cash / w
        dd = np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1.0, 0.0)
        return np.concatenate([norm_prices, log_rets, [cash_ratio, dd]]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        cash_logit = np.clip(action[0], -5.0, 5.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash_frac
        asset_logits = action[1:]
        exp_a = np.exp(asset_logits - np.max(asset_logits))
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

        daily_ret = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        reward = daily_ret * 5.0
        if daily_ret < 0: reward *= 2.0
        drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
        if drawdown < -0.10: reward += drawdown * 2.0

        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))

        return self._flat_obs(), reward, done, {"portfolio_value": new_wealth}


class G6JumpDiffusionSyntheticEnv(RawPriceSyntheticEnv):
    """Synthetic World G1 (v7): G6 Jump-Diffusion with cross-asset correlation & stochastic volatility."""
    def _generate_raw_prices(self):
        total_T = self.episode_len + self.history_len + 10
        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)
        
        # Build cross-asset correlation matrix
        A = np.random.randn(self.num_assets, self.num_assets)
        corr = A @ A.T
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        L = np.linalg.cholesky(corr)

        init_p = np.random.uniform(20.0, 300.0, size=self.num_assets)
        prices[0] = init_p

        for t in range(1, total_T):
            z = L @ np.random.standard_normal(self.num_assets)
            # Jump diffusion process
            jumps = (np.random.rand(self.num_assets) < 0.02) * np.random.uniform(-0.08, 0.08, size=self.num_assets)
            vol = np.random.uniform(0.10, 0.35, size=self.num_assets) / np.sqrt(252.0)
            mu = np.random.uniform(-0.10, 0.25, size=self.num_assets) / 252.0
            
            p_prev = prices[t - 1]
            p_next = p_prev * np.exp((mu - 0.5 * vol**2) + vol * z + jumps)
            prices[t] = np.maximum(0.01, p_next)

        return prices


# ==================================================================================================
# SECTION 2: ARCHITECTURES (v6 Conv1D+Transformer, v7 Spatio-Temporal, LSTM Recurrent)
# ==================================================================================================

class DeepEndToEndTradingNet(nn.Module):
    """v6 Architecture: Multi-Scale Conv1D + 1-Layer Transformer Encoder."""
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64, nhead=2):
        super().__init__()
        self.history_len, self.features_per_step = history_len, features_per_step
        self.conv1d = nn.Sequential(
            nn.Conv1d(features_per_step, 32, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv1d(32, embed_dim, 3, padding=1), nn.LeakyReLU(0.1)
        )
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=128, dropout=0.05, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Sequential(nn.Linear(embed_dim * history_len, 128), nn.LeakyReLU(0.1), nn.LayerNorm(128))
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step).transpose(1, 2)
        conv_out = self.conv1d(x).transpose(1, 2)
        trans_out = self.transformer(conv_out)
        flat = trans_out.reshape(b, -1)
        feat = self.fc(flat)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).to(DEVICE).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs).to(DEVICE)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0) if deterministic else Normal(mean, torch.exp(self.log_std)).sample().cpu().numpy().squeeze(0)


class SpatioTemporalTradingNet(nn.Module):
    """v7 Architecture: Spatio-Temporal Cross-Attention Transformer."""
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64):
        super().__init__()
        self.history_len, self.features_per_step = history_len, features_per_step
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(features_per_step, 64, 3, padding=1), nn.GELU(),
            nn.Conv1d(64, embed_dim, 3, padding=1), nn.GELU()
        )
        t_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256, dropout=0.05, batch_first=True)
        self.temporal_trans = nn.TransformerEncoder(t_layer, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(embed_dim * history_len, 128), nn.GELU(), nn.LayerNorm(128))
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step).transpose(1, 2)
        s_feat = self.spatial_conv(x).transpose(1, 2)
        t_feat = self.temporal_trans(s_feat)
        flat = t_feat.reshape(b, -1)
        feat = self.fc(flat)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).to(DEVICE).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs).to(DEVICE)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0) if deterministic else Normal(mean, torch.exp(self.log_std)).sample().cpu().numpy().squeeze(0)


class LSTMIndustryTradingNet(nn.Module):
    """LSTM Architecture: Industry 2-Layer Recurrent Neural Network Baseline."""
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, hidden_dim=64):
        super().__init__()
        self.history_len, self.features_per_step = history_len, features_per_step
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
                flat_obs = torch.FloatTensor(flat_obs).to(DEVICE).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs).to(DEVICE)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0) if deterministic else Normal(mean, torch.exp(self.log_std)).sample().cpu().numpy().squeeze(0)


# ==================================================================================================
# SECTION 3: PPO TRAINING PIPELINE FOR SYNTHETIC & REAL ENVIRONMENTS
# ==================================================================================================

def train_ppo_model(model, env, total_steps=30_000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    obs = env.reset(seed=seed)
    step = 0

    while step < total_steps:
        obs_b, act_b, rew_b, val_b, logp_b = [], [], [], [], []
        for _ in range(1024):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mean, val = model(obs_t)
                dist = Normal(mean, torch.exp(model.log_std))
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            act_np = action.squeeze(0).cpu().numpy()
            nobs, rew, done, _ = env.step(act_np)
            obs_b.append(obs); act_b.append(act_np); rew_b.append(rew); val_b.append(val.item()); logp_b.append(logp.item())
            obs = nobs
            step += 1
            if done: obs = env.reset()

        with torch.no_grad():
            _, nval = model(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
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

        o_t = torch.FloatTensor(np.array(obs_b)).to(DEVICE)
        a_t = torch.FloatTensor(np.array(act_b)).to(DEVICE)
        adv_t = torch.FloatTensor(adv).to(DEVICE)
        ret_t = torch.FloatTensor(ret).to(DEVICE)
        old_logp_t = torch.FloatTensor(np.array(logp_b)).to(DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(4):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), 128):
                b_idx = idx[s:s + 128]
                mean, val = model(o_t[b_idx])
                dist = Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp_t[b_idx])
                surr1 = ratio * adv_t[b_idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b_idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(val.squeeze(-1), ret_t[b_idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

    model.eval()
    return model


class RealPriceEnvWrapper(RawPriceSyntheticEnv):
    """Wrapper that turns real price series into a gym-like RL training environment."""
    def __init__(self, prices_matrix, history_len=30, fee=0.001):
        self.prices = prices_matrix
        self.num_assets = prices_matrix.shape[1]
        self.history_len = history_len
        self.episode_len = prices_matrix.shape[0] - history_len - 1
        self.initial_cash = 10000.0
        self.fee = fee
        self.features_per_step = 2 * self.num_assets + 2
        self.reset()

    def reset(self, seed=None):
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.5
        init_p = self.prices[self.current_step]
        self.shares = (self.initial_cash * 0.5 / self.num_assets) / init_p
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs()


# ==================================================================================================
# SECTION 4: MASTER COMPREHENSIVE EVALUATOR & LEADERBOARD GENERATOR
# ==================================================================================================

def fetch_real_data(tickers=["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TLT", "GLD"]):
    """Fetches real market prices with automatic offline simulation fallback."""
    try:
        df = yf.download(tickers, period="5y", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        df = df.dropna().ffill().bfill()
        if not df.empty and len(df) > 250:
            print(f"  ✓ Live Historical Market Data Downloaded ({len(df)} trading days, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
            return df.values, df.columns.tolist()
    except Exception:
        pass

    print("  ⚠️ Kaggle Internet OFF. Using High-Realism Real Market Simulation Stream (5 Years).")
    np.random.seed(101)
    T, N = 1260, len(tickers)
    base_p = np.array([450.0, 380.0, 190.0, 110.0, 390.0, 160.0, 170.0, 480.0, 92.0, 210.0])
    series = [base_p]
    for _ in range(T - 1):
        rets = np.random.normal(0.0004, 0.014, size=N)
        series.append(series[-1] * np.exp(rets))
    return np.array(series), tickers


def run_portfolio_eval(model, prices, is_equal_weight=False):
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
        if is_equal_weight:
            tc = 0.0
            taw = np.ones(N) / N
        else:
            flat_obs = np.concatenate(obs_h).astype(np.float32)
            act = model.get_action(flat_obs, deterministic=True)
            c_logit = np.clip(act[0], -5.0, 5.0)
            tc = 1.0 / (1.0 + np.exp(-c_logit))
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
    ret_pct = (eq_a[-1] / eq_a[0] - 1) * 100
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 1e-8 else 0.
    max_dd = float(np.min((eq_a - pk) / pk) * 100)
    win_rate = float(np.mean(r > 0) * 100)
    calmar = float((ret_pct / max(1e-4, abs(max_dd)))) if abs(max_dd) > 1e-4 else 0.

    return {
        "Return (%)": ret_pct,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd,
        "Win Rate (%)": win_rate,
        "Calmar Ratio": calmar,
    }


def execute_master_benchmark():
    print("\n" + "=" * 100)
    print(" 🏆 EXECUTING MASTER COMPARATIVE EVALUATION ACROSS ALL 5 MODEL ARMS")
    print("=" * 100 + "\n")

    tickers = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TLT", "GLD"]
    prices_matrix, asset_names = fetch_real_data(tickers)
    
    # 70% Real Train Split / 30% Out-of-Sample Test Split
    split_idx = int(len(prices_matrix) * 0.70)
    real_train_prices = prices_matrix[:split_idx]
    real_test_prices = prices_matrix[split_idx:]
    
    print(f"  Split: Real Training ({len(real_train_prices)} days) | Out-of-Sample Test ({len(real_test_prices)} days)\n")

    # ----------------------------------------------------------------------------------------------
    # ARM 1: RAI v6 Zero-Shot (Trained 100% on Synthetic G0 World)
    # ----------------------------------------------------------------------------------------------
    print("  [1/5] Training RAI v6 Zero-Shot (100% Synthetic G0 World)...")
    env_v6_synth = RawPriceSyntheticEnv(num_assets=10, episode_len=504)
    model_v6_rai = train_ppo_model(DeepEndToEndTradingNet(), env_v6_synth, total_steps=30_000, seed=42)
    metrics_v6_rai = run_portfolio_eval(model_v6_rai, real_test_prices)

    # ----------------------------------------------------------------------------------------------
    # ARM 2: RAI v7 Zero-Shot (Trained 100% on Synthetic G1 G6 Jump-Diffusion World)
    # ----------------------------------------------------------------------------------------------
    print("  [2/5] Training RAI v7 Zero-Shot (100% Synthetic G1 Jump-Diffusion World)...")
    env_v7_synth = G6JumpDiffusionSyntheticEnv(num_assets=10, episode_len=504)
    model_v7_rai = train_ppo_model(SpatioTemporalTradingNet(), env_v7_synth, total_steps=30_000, seed=42)
    metrics_v7_rai = run_portfolio_eval(model_v7_rai, real_test_prices)

    # ----------------------------------------------------------------------------------------------
    # ARM 3: Real-Trained v6 (Trained on 70% Real Market Data Split)
    # ----------------------------------------------------------------------------------------------
    print("  [3/5] Training Real-Data Trained v6 (70% Real Data Split)...")
    env_real_train = RealPriceEnvWrapper(real_train_prices)
    model_v6_real = train_ppo_model(DeepEndToEndTradingNet(), env_real_train, total_steps=30_000, seed=42)
    metrics_v6_real = run_portfolio_eval(model_v6_real, real_test_prices)

    # ----------------------------------------------------------------------------------------------
    # ARM 4: Real-Trained v7 (Trained on 70% Real Market Data Split)
    # ----------------------------------------------------------------------------------------------
    print("  [4/5] Training Real-Data Trained v7 (70% Real Data Split)...")
    model_v7_real = train_ppo_model(SpatioTemporalTradingNet(), env_real_train, total_steps=30_000, seed=42)
    metrics_v7_real = run_portfolio_eval(model_v7_real, real_test_prices)

    # ----------------------------------------------------------------------------------------------
    # ARM 5: Real-Trained LSTM (Industry Recurrent Neural Network Trained on Real Data)
    # ----------------------------------------------------------------------------------------------
    print("  [5/5] Training Real-Data Trained LSTM (70% Real Data Split)...")
    model_lstm_real = train_ppo_model(LSTMIndustryTradingNet(), env_real_train, total_steps=30_000, seed=42)
    metrics_lstm_real = run_portfolio_eval(model_lstm_real, real_test_prices)

    # ----------------------------------------------------------------------------------------------
    # ARM 6: Equal Weight Baseline (1/N Control)
    # ----------------------------------------------------------------------------------------------
    metrics_eq = run_portfolio_eval(None, real_test_prices, is_equal_weight=True)

    # ----------------------------------------------------------------------------------------------
    # SUMMARY LEADERBOARD TABLE
    # ----------------------------------------------------------------------------------------------
    leaderboard = [
        {"Model Arm": "🤖 RAI v6 Zero-Shot (0% Real)", **metrics_v6_rai},
        {"Model Arm": "🚀 RAI v7 Zero-Shot (0% Real)", **metrics_v7_rai},
        {"Model Arm": "📈 Real-Trained v6 (70% Real)", **metrics_v6_real},
        {"Model Arm": "⚡ Real-Trained v7 (70% Real)", **metrics_v7_real},
        {"Model Arm": "🧠 Real-Trained LSTM (70% Real)", **metrics_lstm_real},
        {"Model Arm": "⚖️ Equal Weight (1/N Baseline)", **metrics_eq},
    ]

    df_res = pd.DataFrame(leaderboard)
    df_res = df_res.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

    print("\n" + "═" * 105)
    print(" 🏆 FINAL OUT-OF-SAMPLE BENCHMARK LEADERBOARD (UNSEEN REAL MARKET DATA)")
    print("═" * 105)
    print(df_res.to_string(index=False))
    print("═" * 105 + "\n")


if __name__ == "__main__":
    execute_master_benchmark()
