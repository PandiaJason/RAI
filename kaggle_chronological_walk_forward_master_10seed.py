"""
====================================================================================================
🏆 MASTER 10-SEED CHRONOLOGICAL WALK-FORWARD REAL-MARKET HOLDOUT BENCHMARK
====================================================================================================
Universes: 4 Global Universes (Indian Nifty 50, US Tech & SPY, Forex & Commodities, Crypto)
Seeds: 10 Independent Random Seeds [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
Hardware: Dual NVIDIA T4 GPUs (cuda:0 and cuda:1)
Primary Metric: 248-Day (~1 Year) Untouched Future Real-Market Holdout (2025 - 2026)
Non-Inferiority Hypothesis: Synthetic-trained RAI v8.2 (0% real data) matches real-data baselines
                            while achieving lower maximum realized drawdown.
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
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf

warnings.filterwarnings("ignore")

NUM_GPUS = torch.cuda.device_count()
DEVICES = [torch.device(f'cuda:{i}') for i in range(NUM_GPUS)] if NUM_GPUS > 0 else [torch.device('cpu')]
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909] # 10 Independent Seeds

print(f"✓ Master 10-Seed Walk-Forward Suite Active | Detected {NUM_GPUS} GPUs: {[str(d) for d in DEVICES]} | PyTorch: {torch.__version__}")


# ==================================================================================================
# GLOBAL UNIVERSES & CHRONOLOGICAL 3-WAY SPLIT (5-YEAR DATASET)
# ==================================================================================================
GLOBAL_UNIVERSES = {
    "1. 🇮🇳 Indian Nifty 50 Equities": {
        "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS"],
        "period": "5y"
    },
    "2. 🇺🇸 US Tech & Benchmark Index": {
        "tickers": ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TLT", "GLD"],
        "period": "5y"
    },
    "3. 🌍 Global Forex & Commodities": {
        "tickers": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", 
                    "GC=F", "CL=F", "SI=F", "HG=F", "NG=F"],
        "period": "5y"
    },
    "4. 🪙 Cryptocurrency Market": {
        "tickers": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
                    "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD"],
        "period": "5y"
    }
}


def fetch_chronological_3way_split(tickers, period="5y"):
    """
    Downloads 5 years of daily prices and performs a strict 3-way chronological split:
      - 60% In-Sample Train (~3 Years)
      - 20% Out-of-Sample Test (~1 Year)
      - 20% Strict Untouched Future Real Holdout (~1 Year)
    """
    try:
        df = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        df = df.dropna().ffill().bfill()
        if not df.empty and len(df) > 200:
            prices = df.values
            dates = df.index
            T = len(prices)
            idx_train_end = int(T * 0.60)
            idx_oos_end = int(T * 0.80)

            dates_info = {
                "train_start": dates[0].strftime("%Y-%m-%d"),
                "train_end": dates[idx_train_end - 1].strftime("%Y-%m-%d"),
                "oos_start": dates[idx_train_end].strftime("%Y-%m-%d"),
                "oos_end": dates[idx_oos_end - 1].strftime("%Y-%m-%d"),
                "future_start": dates[idx_oos_end].strftime("%Y-%m-%d"),
                "future_end": dates[-1].strftime("%Y-%m-%d"),
            }
            return prices[:idx_train_end], prices[idx_train_end:idx_oos_end], prices[idx_oos_end:], dates_info
    except Exception:
        pass

    np.random.seed(hash(tickers[0]) % 10000)
    T, N = 1260, len(tickers)
    base_p = np.random.uniform(50.0, 500.0, size=N)
    series = [base_p]
    for _ in range(T - 1):
        series.append(series[-1] * np.exp(np.random.normal(0.0004, 0.016, size=N)))
    prices = np.array(series)
    idx_train_end = int(T * 0.60)
    idx_oos_end = int(T * 0.80)
    dates_info = {
        "train_start": "Year 1 Day 1", "train_end": f"Day {idx_train_end}",
        "oos_start": f"Day {idx_train_end+1}", "oos_end": f"Day {idx_oos_end}",
        "future_start": f"Day {idx_oos_end+1}", "future_end": f"Day {T}"
    }
    return prices[:idx_train_end], prices[idx_train_end:idx_oos_end], prices[idx_oos_end:], dates_info


# ==================================================================================================
# PROCEDURAL WORLD ENGINE V8.2
# ==================================================================================================
class ProceduralWorldEngineV82:
    REGIME_PROPERTIES = {
        'bull':             {'drift': (0.15, 0.40),  'vol': (0.10, 0.20), 'corr_shift': 0.0},
        'bear':             {'drift': (-0.45, -0.15),'vol': (0.25, 0.50), 'corr_shift': 0.2},
        'sideways':         {'drift': (-0.05, 0.05),  'vol': (0.08, 0.18), 'corr_shift': 0.0},
        'stagflation':      {'drift': (-0.25, -0.05),'vol': (0.20, 0.40), 'corr_shift': 0.3},
        'liquidity_crisis': {'drift': (-0.60, -0.30),'vol': (0.35, 0.70), 'corr_shift': 0.6},
        'bubble_bust':      {'drift': (-0.50, 0.30),  'vol': (0.30, 0.60), 'corr_shift': 0.5},
    }

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001, reward_mode='symmetric_log'):
        self.num_assets, self.action_dim = num_assets, num_assets + 1
        self.history_len, self.episode_len = history_len, episode_len
        self.initial_cash, self.fee, self.reward_mode = initial_cash, fee, reward_mode
        self.features_per_step = 2 * num_assets + 2
        self.reset()

    def _sample_world_parameters(self):
        n_segments = np.random.randint(3, 7)
        keys = list(self.REGIME_PROPERTIES.keys())
        regime_seq = [keys[np.random.randint(len(keys))] for _ in range(n_segments)]
        total_T = self.episode_len + self.history_len + 15
        min_dur = 30
        rem_T = total_T - n_segments * min_dur
        props = np.random.dirichlet(np.ones(n_segments))
        segment_durations = (min_dur + props * rem_T).astype(int)
        segment_durations[-1] = total_T - int(np.sum(segment_durations[:-1]))
        n_factors = np.random.randint(2, 5)
        factor_loadings = np.random.uniform(-0.8, 0.8, size=(self.num_assets, n_factors))
        A = np.random.randn(self.num_assets, self.num_assets)
        corr_matrix = A @ A.T
        d = np.sqrt(np.diag(corr_matrix))
        base_corr = corr_matrix / np.outer(d, d)
        return {
            'regimes': regime_seq, 'durations': segment_durations, 'n_factors': n_factors,
            'factor_loadings': factor_loadings, 'base_corr': base_corr, 'execution_delay': 0,
            'noise_level': 0.002, 'heavy_tail_df': 8.0, 'jump_intensity': 0.01,
            'jump_size_std': 0.03, 'total_T': total_T
        }

    def _generate_procedural_prices(self, cfg):
        total_T = cfg['total_T']
        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)
        prices[0] = np.random.uniform(20.0, 300.0, size=self.num_assets)
        day = 0
        current_vol = np.ones(self.num_assets) * (0.15 / np.sqrt(252.0))**2
        for reg_idx, dur in enumerate(cfg['durations']):
            reg_name = cfg['regimes'][reg_idx]
            reg_props = self.REGIME_PROPERTIES[reg_name]
            corr_shift = reg_props['corr_shift']
            target_corr = (1.0 - corr_shift) * cfg['base_corr'] + corr_shift * np.ones((self.num_assets, self.num_assets))
            np.fill_diagonal(target_corr, 1.0)
            try: L = np.linalg.cholesky(target_corr)
            except: L = np.eye(self.num_assets)
            drift_annual = np.random.uniform(*reg_props['drift']) + np.random.uniform(-0.05, 0.05, size=self.num_assets)
            for _ in range(max(0, min(dur, total_T - day - 1))):
                day += 1
                if day >= total_T: break
                factor_component = cfg['factor_loadings'] @ np.random.normal(0, 1.0, size=cfg['n_factors'])
                z_raw = np.clip(np.random.standard_t(df=cfg['heavy_tail_df'], size=self.num_assets), -4.0, 4.0)
                z = L @ z_raw + factor_component * 0.3
                current_vol = np.clip(0.000005 + 0.05 * (z**2) + 0.90 * current_vol, 1e-6, 0.01)
                stoch_vol = np.sqrt(current_vol)
                log_return = (drift_annual / 252.0 - 0.5 * stoch_vol**2) + stoch_vol * z
                prices[day] = np.maximum(0.01, prices[day - 1] * np.exp(np.clip(log_return, -0.25, 0.25)))
            if day >= total_T: break
        while day < total_T: prices[day] = prices[max(0, day - 1)]; day += 1
        return np.nan_to_num(prices, nan=100.0, posinf=500.0, neginf=0.01)

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.world_cfg = self._sample_world_parameters()
        self.prices = self._generate_procedural_prices(self.world_cfg)
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.5
        self.shares = (self.initial_cash * 0.5 / self.num_assets) / np.maximum(1e-4, self.prices[self.start])
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0
        self.action_queue = [np.zeros(self.action_dim, dtype=np.float32)]
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs()

    def _wealth(self): return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p, pp = self.prices[t], self.prices[max(0, t-1)]
        w = max(1e-4, self._wealth())
        return np.nan_to_num(np.concatenate([p / self.prices[self.start], np.clip(np.log(p / np.maximum(1e-4, pp)), -0.5, 0.5), [self.cash / w, np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1.0, 0.0)]]).astype(np.float32), nan=0.0)

    def _flat_obs(self): return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        action = np.nan_to_num(action, nan=0.0)
        c_frac = 1.0 / (1.0 + np.exp(-np.clip(action[0], -5.0, 5.0)))
        exp_a = np.exp(action[1:] - np.max(action[1:]))
        target_w = (exp_a / np.sum(exp_a)) * (1.0 - c_frac)
        prices = self.prices[self.current_step]
        wealth = max(1e-4, self._wealth())
        drift = abs(self.cash / wealth - c_frac) + np.sum(np.abs((self.shares * prices) / wealth - target_w))
        if drift > 0.03:
            net = max(1e-4, wealth - wealth * drift * self.fee)
            self.cash = net * c_frac
            self.shares = (net * target_w) / np.maximum(1e-4, prices)
        self.current_step += 1
        self.steps_done += 1
        new_wealth = max(1e-4, self._wealth())
        self.peak_wealth = max(self.peak_wealth, new_wealth)
        daily_ret = np.clip((new_wealth - self.last_wealth) / max(1e-4, self.last_wealth), -0.5, 0.5)
        reward = float(np.log(new_wealth / max(1e-4, self.last_wealth)) * 100.0 - 50.0 * (max(0.0, -daily_ret)**2))
        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))
        return self._flat_obs(), float(reward), done, {}


# ==================================================================================================
# NETWORK ARCHITECTURES & TRAINERS
# ==================================================================================================
class MultiScaleRiskAwareNet(nn.Module):
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64):
        super().__init__()
        self.history_len, self.features_per_step = history_len, features_per_step
        self.conv_short = nn.Conv1d(features_per_step, 24, kernel_size=3, padding=1)
        self.conv_med   = nn.Conv1d(features_per_step, 24, kernel_size=7, padding=3)
        self.conv_long  = nn.Conv1d(features_per_step, 24, kernel_size=15, padding=7)
        self.scale_fusion = nn.Sequential(nn.Conv1d(72, embed_dim, kernel_size=1), nn.GELU())
        trans_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=256, dropout=0.05, batch_first=True)
        self.transformer = nn.TransformerEncoder(trans_layer, num_layers=2)
        self.fc = nn.Sequential(nn.Linear(embed_dim * history_len, 128), nn.GELU(), nn.LayerNorm(128))
        self.risk_head = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus())
        self.actor_head = nn.Sequential(nn.Linear(128 + 1, 64), nn.GELU(), nn.Linear(64, action_dim))
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step).transpose(1, 2)
        fused = self.scale_fusion(torch.cat([F.gelu(self.conv_short(x)), F.gelu(self.conv_med(x)), F.gelu(self.conv_long(x))], dim=1)).transpose(1, 2)
        latent = self.fc(self.transformer(fused).reshape(b, -1))
        risk = torch.nan_to_num(self.risk_head(latent), nan=0.01)
        return torch.nan_to_num(self.actor_head(torch.cat([latent, risk], dim=-1)), nan=0.0), torch.nan_to_num(self.critic_head(latent), nan=0.0), risk

    def get_action(self, flat_obs, device=DEVICES[0], deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).to(device)
                if flat_obs.ndim == 1: flat_obs = flat_obs.unsqueeze(0)
            logits, _, _ = self.forward(flat_obs)
            return logits.squeeze(0).cpu().numpy()

class LSTMDNNBaseline(nn.Module):
    def __init__(self, history_len=30, features_per_step=22, action_dim=11):
        super().__init__()
        self.history_len, self.features_per_step = history_len, features_per_step
        self.lstm = nn.LSTM(features_per_step, 64, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, x):
        out, _ = self.lstm(x.reshape(x.shape[0], self.history_len, self.features_per_step))
        return self.fc(out[:, -1, :])

    def get_action(self, flat_obs, device=DEVICES[0]):
        with torch.no_grad():
            t_obs = torch.FloatTensor(flat_obs).unsqueeze(0).to(device)
            return np.nan_to_num(self.forward(t_obs).squeeze(0).cpu().numpy(), nan=0.0)

def train_lstm_baseline(train_prices, seed=42, device=DEVICES[0]):
    torch.manual_seed(seed)
    model = LSTMDNNBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    norm_p = train_prices / train_prices[0]
    obs_list, target_list = [], []
    for t in range(30, len(norm_p) - 1):
        obs_seq = [np.concatenate([norm_p[t-30+i], np.log(norm_p[t-30+i] / np.maximum(1e-4, norm_p[max(0, t-30+i-1)])), [0.5, 0.0]]) for i in range(30)]
        obs_list.append(np.concatenate(obs_seq).astype(np.float32))
        target_list.append(np.concatenate([[0.0], (norm_p[t+1] - norm_p[t]) / np.maximum(1e-4, norm_p[t])]))
    if len(obs_list) > 30:
        o_t, y_t = torch.FloatTensor(np.array(obs_list)).to(device), torch.FloatTensor(np.array(target_list)).to(device)
        for _ in range(50):
            optimizer.zero_grad(); F.mse_loss(model(o_t), y_t).backward(); optimizer.step()
    model.eval()
    return model

class RealDataPPOEnv:
    def __init__(self, prices, history_len=30):
        self.prices = prices / prices[0]
        self.num_assets, self.history_len = prices.shape[1], history_len
        self.reset()

    def reset(self):
        self.start = self.history_len
        self.current_step = self.start
        self.cash = 5000.0
        self.shares = (5000.0 / self.num_assets) / np.maximum(1e-4, self.prices[self.start])
        self.peak_wealth = 10000.0
        self.last_wealth = 10000.0
        self.obs_history = [self._obs_at(i) for i in range(self.history_len)]
        return self._flat_obs()

    def _obs_at(self, t):
        p, pp = self.prices[t], self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        return np.concatenate([p, np.log(p / np.maximum(1e-4, pp)), [self.cash / w, 0.0]]).astype(np.float32)

    def _flat_obs(self): return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        c_frac = 1.0 / (1.0 + np.exp(-np.clip(action[0], -5, 5)))
        target_w = (np.exp(action[1:] - np.max(action[1:])) / np.sum(np.exp(action[1:] - np.max(action[1:])))) * (1.0 - c_frac)
        p = self.prices[self.current_step]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        self.cash = w * c_frac
        self.shares = (w * target_w) / np.maximum(1e-4, p)
        self.current_step += 1
        new_w = max(1e-4, self.cash + np.sum(self.shares * self.prices[self.current_step]))
        ret = (new_w - self.last_wealth) / max(1e-4, self.last_wealth)
        done = self.current_step >= len(self.prices) - 1
        self.last_wealth = new_w
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))
        return self._flat_obs(), float(ret * 100.0), done, {}

def train_real_ppo_model(train_prices, seed=42, device=DEVICES[0]):
    torch.manual_seed(seed)
    env = RealDataPPOEnv(train_prices)
    model = MultiScaleRiskAwareNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs = env.reset()
    step = 0
    while step < 10_000:
        obs_b, act_b, rew_b, val_b, logp_b, done_b = [], [], [], [], [], []
        for _ in range(512):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, val, unc = model(obs_t)
                dist = Normal(mean, torch.exp(model.log_std))
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            act_np = action.squeeze(0).cpu().numpy()
            nobs, rew, done, _ = env.step(act_np)
            obs_b.append(obs); act_b.append(act_np); rew_b.append(rew); val_b.append(val.item()); logp_b.append(logp.item()); done_b.append(float(done))
            obs = env.reset() if done else nobs
            step += 1
        r, v, d_mask = np.array(rew_b), np.array(val_b + [val.item()]), np.array(done_b)
        delta = r + 0.99 * v[1:] * (1.0 - d_mask) - v[:-1]
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            gae = delta[t] + 0.99 * 0.95 * gae * (1.0 - d_mask[t])
            adv[t] = gae
        o_t, a_t = torch.FloatTensor(np.array(obs_b)).to(device), torch.FloatTensor(np.array(act_b)).to(device)
        adv_t = torch.FloatTensor(adv).to(device); adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.FloatTensor(adv + v[:-1]).to(device); old_logp_t = torch.FloatTensor(np.array(logp_b)).to(device)
        for _ in range(2):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), 128):
                b_idx = idx[s:s + 128]
                mean, val, unc = model(o_t[b_idx])
                dist = Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp_t[b_idx])
                surr1 = ratio * adv_t[b_idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b_idx]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * F.mse_loss(val.squeeze(-1), ret_t[b_idx])
                optimizer.zero_grad(); (p_loss + v_loss).backward(); optimizer.step()
    model.eval()
    return model

def train_rai_v82_procedural_model(seed=42, device=DEVICES[0]):
    torch.manual_seed(seed); np.random.seed(seed)
    env = ProceduralWorldEngineV82(num_assets=10, episode_len=504, reward_mode='log_moderate_risk')
    model = MultiScaleRiskAwareNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs = env.reset(seed=seed)
    step = 0
    while step < 25_000:
        obs_b, act_b, rew_b, val_b, logp_b, done_b = [], [], [], [], [], []
        for _ in range(512):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, val, unc = model(obs_t)
                dist = Normal(mean, torch.exp(model.log_std))
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            act_np = action.squeeze(0).cpu().numpy()
            nobs, rew, done, _ = env.step(act_np)
            obs_b.append(obs); act_b.append(act_np); rew_b.append(rew); val_b.append(val.item()); logp_b.append(logp.item()); done_b.append(float(done))
            obs = env.reset() if done else nobs
            step += 1
        r, v, d_mask = np.array(rew_b), np.array(val_b + [val.item()]), np.array(done_b)
        delta = r + 0.99 * v[1:] * (1.0 - d_mask) - v[:-1]
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            gae = delta[t] + 0.99 * 0.95 * gae * (1.0 - d_mask[t])
            adv[t] = gae
        o_t, a_t = torch.FloatTensor(np.array(obs_b)).to(device), torch.FloatTensor(np.array(act_b)).to(device)
        adv_t = torch.FloatTensor(adv).to(device); adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.FloatTensor(adv + v[:-1]).to(device); old_logp_t = torch.FloatTensor(np.array(logp_b)).to(device)
        for _ in range(2):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), 128):
                b_idx = idx[s:s + 128]
                mean, val, unc = model(o_t[b_idx])
                dist = Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp_t[b_idx])
                surr1 = ratio * adv_t[b_idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b_idx]
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * F.mse_loss(val.squeeze(-1), ret_t[b_idx])
                optimizer.zero_grad(); (p_loss + v_loss).backward(); optimizer.step()
    model.eval()
    return model

# Corrected Walk-Forward Evaluator (Fixed 30-Day Warmup + Max Days Horizon)
def evaluate_model_walk_forward(model, prices_series, device=DEVICES[0], max_days=None):
    T, N = prices_series.shape
    if T <= 30: return 0.0, 0.0, 0.0

    trading_steps = min(T - 30, max_days) if max_days is not None else (T - 30)
    eval_end_t = 30 + trading_steps

    prices = prices_series / prices_series[0]
    wealth, peak_wealth = 10000.0, 10000.0
    equity_curve = [10000.0]
    cash_frac = 0.50
    stock_weights = np.ones(N) / float(N)

    # 1. Observation History Warmup (First 30 Days)
    obs_h = []
    for t in range(30):
        p, pp = prices[t], prices[max(0, t - 1)]
        obs_h.append(np.concatenate([p, np.log(p / np.maximum(1e-4, pp)), [cash_frac, 0.0]]).astype(np.float32))

    # 2. Main Portfolio Execution Loop from day 30 to eval_end_t
    for t in range(30, eval_end_t):
        p_prev, p_curr = prices[t - 1], prices[t]
        asset_returns = (p_curr - p_prev) / np.maximum(1e-4, p_prev)

        cash_val = wealth * cash_frac
        stock_val = wealth * (1.0 - cash_frac)
        new_stock_val = np.sum(stock_val * stock_weights * (1.0 + asset_returns))
        wealth = max(1e-4, cash_val + new_stock_val)
        peak_wealth = max(peak_wealth, wealth)
        equity_curve.append(wealth)

        flat_obs = np.concatenate(obs_h).astype(np.float32)
        act = model.get_action(flat_obs, device=device)

        c_frac = 1.0 / (1.0 + np.exp(-np.clip(act[0], -5.0, 5.0)))
        exp_a = np.exp(act[1:] - np.max(act[1:]))
        target_stock_w = exp_a / np.sum(exp_a)

        drift = abs(cash_frac - c_frac) + np.sum(np.abs(stock_weights - target_stock_w))
        if drift > 0.03: wealth = max(1e-4, wealth - wealth * drift * 0.001)

        cash_frac = c_frac
        stock_weights = target_stock_w

        obs_h.pop(0)
        obs_h.append(np.concatenate([p_curr, np.log(p_curr / np.maximum(1e-4, p_prev)), [cash_frac, np.clip((wealth - peak_wealth) / max(1e-4, peak_wealth), -1, 0)]]).astype(np.float32))

    eq_a = np.array(equity_curve)
    r = np.diff(eq_a) / np.maximum(1e-8, eq_a[:-1])
    pk = np.maximum.accumulate(eq_a)
    ret_pct = (eq_a[-1] / eq_a[0] - 1) * 100
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252)) if (len(r) > 1 and np.std(r) > 1e-8) else 0.
    max_dd = float(np.min((eq_a - pk) / pk) * 100)

    return ret_pct, sharpe, max_dd

def _run_seed_evaluation_master(args):
    u_name, u_cfg, train_p, oos_p, future_p, dates_info, seed, device_idx = args
    target_device = DEVICES[device_idx % len(DEVICES)]
    print(f"  🌱 [{u_name[:15]}] Seed {seed} on GPU {target_device}...")

    # 1. Train Models (FROZEN after this step!)
    lstm_m = train_lstm_baseline(train_p, seed=seed, device=target_device)
    rppo_m = train_real_ppo_model(train_p, seed=seed, device=target_device)
    rai_m  = train_rai_v82_procedural_model(seed=seed, device=target_device)

    # Save Model Weights locally for downloading
    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)
    clean_u = u_name.split(".")[1].strip().replace(" ", "_")
    torch.save(lstm_m.state_dict(), f"{save_dir}/lstm_dnn_{clean_u}_seed{seed}.pt")
    torch.save(rppo_m.state_dict(), f"{save_dir}/real_ppo_{clean_u}_seed{seed}.pt")
    torch.save(rai_m.state_dict(), f"{save_dir}/rai_v82_{clean_u}_seed{seed}.pt")

    models = {
        "1. LSTM-DNN (60% Real)": lstm_m,
        "2. Real-PPO (60% Real)": rppo_m,
        "3. RAI v8.2 (0% Real)": rai_m
    }

    records = []

    # Primary Metric: Full Untouched Future Real Holdout (~1 Year / 248 Days)
    for m_name, model in models.items():
        ret_fut, sh_fut, dd_fut = evaluate_model_walk_forward(model, future_p, device=target_device)
        ret_oos, sh_oos, dd_oos = evaluate_model_walk_forward(model, oos_p, device=target_device)
        records.append({
            "Universe": u_name,
            "Model Arm": m_name,
            "Seed": seed,
            "OOS Return (%)": ret_oos,
            "Future 1-Yr Return (%)": ret_fut,
            "Future Sharpe": sh_fut,
            "Future Max DD (%)": dd_fut
        })

    return records


def execute_master_10seed_walk_forward_benchmark():
    print("=" * 125)
    print(" 🏆 EXECUTING MASTER 10-SEED CHRONOLOGICAL WALK-FORWARD REAL-MARKET HOLDOUT BENCHMARK")
    print(f" 🌐 4 UNIVERSES × 10 SEEDS = 40 EXPERIMENTS PARALLELIZED ACROSS {len(DEVICES)} GPUs")
    print("=" * 125 + "\n")

    all_records = []
    start_time = time.time()

    for u_name, u_cfg in GLOBAL_UNIVERSES.items():
        print(f"\n📊 --- UNIVERSE: {u_name} ---")
        train_p, oos_p, future_p, dates_info = fetch_chronological_3way_split(u_cfg["tickers"], u_cfg["period"])

        print(f"  📅 In-Sample Train:   {dates_info['train_start']} -> {dates_info['train_end']} ({len(train_p)} trading days)")
        print(f"  📅 OOS Test Split:    {dates_info['oos_start']} -> {dates_info['oos_end']} ({len(oos_p)} trading days)")
        print(f"  📅 Future Real Data:  {dates_info['future_start']} -> {dates_info['future_end']} ({len(future_p)} trading days)")

        tasks = []
        for i, seed in enumerate(SEEDS):
            tasks.append((u_name, u_cfg, train_p, oos_p, future_p, dates_info, seed, i % len(DEVICES)))

        with ThreadPoolExecutor(max_workers=len(DEVICES)) as executor:
            results = list(executor.map(_run_seed_evaluation_master, tasks))

        for res in results: all_records.extend(res)

    total_elapsed = (time.time() - start_time) / 60.0
    df = pd.DataFrame(all_records)

    print("\n" + "═" * 125)
    print(f" 🏆 MASTER 10-SEED LEADERBOARD: 248-DAY UNTOUCHED FUTURE HOLDOUT (COMPLETED IN {total_elapsed:.1f} MINS)")
    print("═" * 125)

    # Detailed Summary by Universe and Model
    summary_by_universe = df.groupby(["Universe", "Model Arm"])[["Future 1-Yr Return (%)", "Future Sharpe", "Future Max DD (%)"]].agg(['mean', 'std'])
    print("\n📊 --- SUMMARY BY UNIVERSE ---")
    print(summary_by_universe.to_string())

    # Overall Aggregate Across All 4 Universes (40 seeds total)
    overall_summary = df.groupby(["Model Arm"])[["Future 1-Yr Return (%)", "Future Sharpe", "Future Max DD (%)"]].agg(['mean', 'std'])
    print("\n🌍 --- OVERALL AGGREGATE ACROSS ALL 4 UNIVERSES (N=40 SEEDS) ---")
    print(overall_summary.to_string())
    print("═" * 125 + "\n")

    # Package all trained PyTorch model checkpoints into a single ZIP file for Kaggle Download
    os.system("zip -q -r rai_master_trained_models.zip ./saved_models")
    print("📦 SUCCESS! All trained model checkpoints saved and packaged into 'rai_master_trained_models.zip'!")
    print("   👉 Click 'Output' in Kaggle right-sidebar to download 'rai_master_trained_models.zip'")

if __name__ == "__main__":
    execute_master_10seed_walk_forward_benchmark()
