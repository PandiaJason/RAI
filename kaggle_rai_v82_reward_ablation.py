"""
====================================================================================================
🧪 KAGGLE MASTER SUITE: RAI v8.2 REWARD FORMULATION ABLATION STUDY
====================================================================================================
Tests the primary scientific hypothesis:
"Is the ~78% cash defensive posture caused by asymmetric downside reward penalties, or by the 
extreme volatility of procedural worlds?"

Reward Arms Evaluated:
  - Arm A (Asymmetric Penalties)  : 5.0 * ret - 2.0 * max(0,-ret) + 2.0 * DD
  - Arm B (Symmetric Log-Growth)  : log(W_t / W_{t-1}) - 0.001 * Turnover
  - Arm C (Log-Growth + Mod Risk) : log(W_t / W_{t-1}) - 0.5 * max(0,-ret)^2 - 0.001 * Turnover
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

print(f"✓ RAI v8.2 Reward Ablation Suite Initialized | Device: {DEVICE} | PyTorch: {torch.__version__}")


# ==================================================================================================
# MULTI-DATASET DEFINITIONS FOR UNTOUCHED ZERO-SHOT EVALUATION
# ==================================================================================================
GLOBAL_UNIVERSES = {
    "1. 🇮🇳 Indian Nifty 50 Equities": {
        "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
                    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS"],
        "period": "2y"
    },
    "2. 🇺🇸 US Tech & Benchmark Index": {
        "tickers": ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TLT", "GLD"],
        "period": "2y"
    },
    "3. 🌍 Global Forex & Commodities": {
        "tickers": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", 
                    "GC=F", "CL=F", "SI=F", "HG=F", "NG=F"],
        "period": "2y"
    },
    "4. 🪙 Cryptocurrency Market": {
        "tickers": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
                    "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD", "LTC-USD"],
        "period": "2y"
    }
}


# ==================================================================================================
# SECTION 1: PROCEDURAL WORLD ENGINE WITH PARAMETRIC REWARD (W_proc)
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
        self.num_assets = num_assets
        self.action_dim = num_assets + 1
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee
        self.reward_mode = reward_mode
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

        execution_delay = int(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
        noise_level = float(np.random.uniform(0.001, 0.005))
        heavy_tail_df = float(np.random.uniform(4.0, 15.0))
        jump_intensity = float(np.random.uniform(0.005, 0.03))
        jump_size_std = float(np.random.uniform(0.01, 0.06))

        return {
            'regimes': regime_seq,
            'durations': segment_durations,
            'n_factors': n_factors,
            'factor_loadings': factor_loadings,
            'base_corr': base_corr,
            'execution_delay': execution_delay,
            'noise_level': noise_level,
            'heavy_tail_df': heavy_tail_df,
            'jump_intensity': jump_intensity,
            'jump_size_std': jump_size_std,
            'total_T': total_T
        }

    def _generate_procedural_prices(self, cfg):
        total_T = cfg['total_T']
        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)

        init_prices = np.random.uniform(20.0, 300.0, size=self.num_assets)
        prices[0] = init_prices

        day = 0
        current_vol = np.ones(self.num_assets) * (0.15 / np.sqrt(252.0))**2
        omega, alpha, beta = 0.000005, 0.05, 0.90

        for reg_idx, dur in enumerate(cfg['durations']):
            reg_name = cfg['regimes'][reg_idx]
            reg_props = self.REGIME_PROPERTIES[reg_name]

            corr_shift = reg_props['corr_shift']
            target_corr = (1.0 - corr_shift) * cfg['base_corr'] + corr_shift * np.ones((self.num_assets, self.num_assets))
            np.fill_diagonal(target_corr, 1.0)
            
            try:
                L = np.linalg.cholesky(target_corr)
            except np.linalg.LinAlgError:
                L = np.eye(self.num_assets)

            drift_annual = np.random.uniform(*reg_props['drift']) + np.random.uniform(-0.05, 0.05, size=self.num_assets)

            for _ in range(max(0, min(dur, total_T - day - 1))):
                day += 1
                if day >= total_T: break

                factor_returns = np.random.normal(0, 1.0, size=cfg['n_factors'])
                factor_component = cfg['factor_loadings'] @ factor_returns

                z_raw = np.random.standard_t(df=cfg['heavy_tail_df'], size=self.num_assets)
                z_raw = np.clip(z_raw, -4.0, 4.0)
                z = L @ z_raw + factor_component * 0.3

                current_vol = omega + alpha * (z**2) + beta * current_vol
                current_vol = np.clip(current_vol, 1e-6, 0.01)
                stoch_vol = np.sqrt(current_vol)

                jump_occured = (np.random.rand(self.num_assets) < cfg['jump_intensity'])
                jumps = jump_occured * np.random.normal(0, cfg['jump_size_std'], size=self.num_assets)
                jumps = np.clip(jumps, -0.15, 0.15)

                total_drift = drift_annual / 252.0
                p_prev = prices[day - 1]
                log_return = (total_drift - 0.5 * stoch_vol**2) + stoch_vol * z + jumps

                if cfg['noise_level'] > 0:
                    log_return += np.random.normal(0, cfg['noise_level'], size=self.num_assets)

                log_return = np.clip(log_return, -0.25, 0.25)
                prices[day] = np.maximum(0.01, p_prev * np.exp(log_return))

            if day >= total_T: break

        while day < total_T:
            prices[day] = prices[max(0, day - 1)]
            day += 1

        return np.nan_to_num(prices, nan=100.0, posinf=500.0, neginf=0.01)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.world_cfg = self._sample_world_parameters()
        self.prices = self._generate_procedural_prices(self.world_cfg)
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.5
        init_p = self.prices[self.current_step]
        self.shares = (self.initial_cash * 0.5 / self.num_assets) / np.maximum(1e-4, init_p)
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0

        self.delay = self.world_cfg['execution_delay']
        self.action_queue = [np.zeros(self.action_dim, dtype=np.float32) for _ in range(self.delay)]

        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs()

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p = self.prices[t]; p_prev = self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        norm_prices = p / np.maximum(1e-4, self.prices[self.start])
        log_rets = np.log(p / np.maximum(1e-4, p_prev))
        log_rets = np.clip(log_rets, -0.5, 0.5)
        cash_ratio = self.cash / w
        dd = np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1.0, 0.0)
        obs = np.concatenate([norm_prices, log_rets, [cash_ratio, dd]]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        action = np.nan_to_num(action, nan=0.0)
        self.action_queue.append(action)
        exec_action = self.action_queue.pop(0)

        cash_logit = np.clip(exec_action[0], -5.0, 5.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash_frac
        asset_logits = exec_action[1:]
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
        new_wealth = max(1e-4, self._wealth())
        self.peak_wealth = max(self.peak_wealth, new_wealth)

        daily_ret = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        daily_ret = np.clip(daily_ret, -0.5, 0.5)

        # PARAMETRIC REWARD ABLATION MODES
        if self.reward_mode == 'asymmetric':
            reward = daily_ret * 5.0
            if daily_ret < 0: reward *= 2.0
            drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
            if drawdown < -0.10: reward += drawdown * 2.0

        elif self.reward_mode == 'symmetric_log':
            log_growth = np.log(new_wealth / max(1e-4, self.last_wealth))
            reward = float(log_growth * 100.0) - (drift * 0.1)

        elif self.reward_mode == 'log_moderate_risk':
            log_growth = np.log(new_wealth / max(1e-4, self.last_wealth))
            downside_sq = max(0.0, -daily_ret)**2
            reward = float(log_growth * 100.0) - 50.0 * downside_sq - (drift * 0.1)

        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))

        return self._flat_obs(), float(reward), done, {"portfolio_value": new_wealth}


# ==================================================================================================
# SECTION 2: MULTI-SCALE RISK-AWARE ARCHITECTURE
# ==================================================================================================
class MultiScaleRiskAwareNet(nn.Module):
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step
        self.action_dim = action_dim

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
        flat_obs = torch.nan_to_num(flat_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step).transpose(1, 2)

        s_feat = F.gelu(self.conv_short(x))
        m_feat = F.gelu(self.conv_med(x))
        l_feat = F.gelu(self.conv_long(x))

        multi_scale_cat = torch.cat([s_feat, m_feat, l_feat], dim=1)
        fused = self.scale_fusion(multi_scale_cat).transpose(1, 2)

        trans_out = self.transformer(fused)
        flat_repr = trans_out.reshape(b, -1)
        latent = self.fc(flat_repr)

        prediction_error_risk = torch.nan_to_num(self.risk_head(latent), nan=0.01)
        actor_input = torch.cat([latent, prediction_error_risk], dim=-1)

        actor_logits = torch.nan_to_num(self.actor_head(actor_input), nan=0.0)
        value = torch.nan_to_num(self.critic_head(latent), nan=0.0)

        return actor_logits, value, prediction_error_risk

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).to(DEVICE)
                if flat_obs.ndim == 1:
                    flat_obs = flat_obs.unsqueeze(0)
            logits, val, risk = self.forward(flat_obs)
            return logits.squeeze(0).cpu().numpy() if deterministic else Normal(logits, torch.exp(self.log_std)).sample().squeeze(0).cpu().numpy()


# ==================================================================================================
# SECTION 3: PPO TRAINER SUPPORTING PARAMETRIC REWARD MODES
# ==================================================================================================
def train_ablation_model(reward_mode='symmetric_log', total_steps=100_000, seed=42):
    print("=" * 100)
    print(f" 🧪 TRAINING REWARD ABLATION ARM: '{reward_mode.upper()}' ON {DEVICE}")
    print("=" * 100)
    t0 = time.time()

    env = ProceduralWorldEngineV82(num_assets=10, episode_len=504, reward_mode=reward_mode)
    model = MultiScaleRiskAwareNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    obs = env.reset(seed=seed)
    step = 0

    while step < total_steps:
        obs_b, act_b, rew_b, val_b, logp_b, done_b = [], [], [], [], [], []
        for _ in range(1024):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mean, val, unc = model(obs_t)
                dist = Normal(mean, torch.exp(model.log_std))
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            act_np = action.squeeze(0).cpu().numpy()
            nobs, rew, done, _ = env.step(act_np)
            obs_b.append(obs); act_b.append(act_np); rew_b.append(rew); val_b.append(val.item()); logp_b.append(logp.item()); done_b.append(float(done))
            obs = nobs
            step += 1
            if done:
                obs = env.reset()

        with torch.no_grad():
            _, nval, _ = model(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
            nval = nval.item()

        r = np.array(rew_b)
        v = np.array(val_b + [nval])
        d_mask = np.array(done_b)

        delta = r + 0.99 * v[1:] * (1.0 - d_mask) - v[:-1]
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            gae = delta[t] + 0.99 * 0.95 * gae * (1.0 - d_mask[t])
            adv[t] = gae
        ret = adv + v[:-1]

        o_t = torch.FloatTensor(np.array(obs_b)).to(DEVICE)
        a_t = torch.FloatTensor(np.array(act_b)).to(DEVICE)
        adv_t = torch.FloatTensor(adv)
        ret_t = torch.FloatTensor(ret).to(DEVICE)
        old_logp_t = torch.FloatTensor(np.array(logp_b)).to(DEVICE)
        adv_t = ((adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)).to(DEVICE)

        for _ in range(4):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), 128):
                b_idx = idx[s:s + 128]
                mean, val, unc = model(o_t[b_idx])
                dist = Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp_t[b_idx])
                surr1 = ratio * adv_t[b_idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b_idx]
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(val.squeeze(-1), ret_t[b_idx])
                
                target_uncertainty = torch.abs(ret_t[b_idx] - val.squeeze(-1)).detach()
                uncertainty_loss = 0.2 * F.mse_loss(unc.squeeze(-1), target_uncertainty)

                total_loss = policy_loss + value_loss + uncertainty_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

    elapsed = time.time() - t0
    print(f"  ✅ Arm '{reward_mode}' Training Complete in {elapsed:.1f}s!\n")
    model.eval()
    return model


# ==================================================================================================
# SECTION 4: UNTOUCHED ZERO-SHOT EVALUATION ACROSS GLOBAL DOMAINS
# ==================================================================================================
def fetch_universe_prices(tickers, period="2y"):
    try:
        df = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        df = df.dropna().ffill().bfill()
        if not df.empty and len(df) > 50:
            print(f"    ✓ Downloaded {len(df)} trading days via Yahoo Finance.")
            return df.values, df.columns.tolist(), df.index[-1].strftime('%Y-%m-%d')
    except Exception:
        pass

    print("    ⚠️ Kaggle Internet OFF. Generating High-Realism Simulation Stream.")
    np.random.seed(hash(tickers[0]) % 10000)
    T, N = 504, len(tickers)
    base_p = np.random.uniform(50.0, 500.0, size=N)
    series = [base_p]
    for _ in range(T - 1):
        rets = np.random.normal(0.0005, 0.018, size=N)
        series.append(series[-1] * np.exp(rets))
    return np.array(series), tickers, "2026-08-11 (Offline)"


def evaluate_universe(model, universe_name, universe_cfg, arm_label):
    raw_prices, tickers, _ = fetch_universe_prices(universe_cfg["tickers"], period=universe_cfg["period"])
    T, N = raw_prices.shape
    prices = raw_prices / raw_prices[0]

    wealth, peak_wealth = 10000.0, 10000.0
    equity_curve = [10000.0]
    cash_amount = wealth * 0.10
    asset_weights = np.ones(N) * (0.90 / N)

    obs_h = []
    for t in range(30):
        p, pp = prices[t], prices[max(0, t - 1)]
        obs_h.append(np.concatenate([
            p,
            np.log(p / np.maximum(1e-4, pp)),
            [cash_amount / wealth, np.clip((wealth - peak_wealth) / max(1e-4, peak_wealth), -1, 0)]
        ]).astype(np.float32))

    for t in range(30, T):
        p_prev, p_curr = prices[t - 1], prices[t]
        asset_returns = (p_curr - p_prev) / np.maximum(1e-4, p_prev)

        invested_wealth = wealth * (1.0 - (cash_amount / wealth))
        wealth = cash_amount + np.sum(invested_wealth * asset_weights * (1.0 + asset_returns))
        wealth = max(1e-4, wealth)
        peak_wealth = max(peak_wealth, wealth)
        equity_curve.append(wealth)

        flat_obs = np.concatenate(obs_h).astype(np.float32)
        act = model.get_action(flat_obs, deterministic=True)

        cash_logit = np.clip(act[0], -5.0, 5.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash_frac
        exp_a = np.exp(act[1:] - np.max(act[1:]))
        target_asset_w = exp_a / np.sum(exp_a)

        drift = abs((cash_amount / wealth) - target_cash_frac) + np.sum(np.abs(asset_weights - target_asset_w))
        if drift > 0.03:
            wealth -= wealth * drift * 0.001
            wealth = max(1e-4, wealth)

        cash_amount = wealth * target_cash_frac
        asset_weights = target_asset_w

        obs_h.pop(0)
        obs_h.append(np.concatenate([
            p_curr,
            np.log(p_curr / np.maximum(1e-4, p_prev)),
            [target_cash_frac, np.clip((wealth - peak_wealth) / max(1e-4, peak_wealth), -1, 0)]
        ]).astype(np.float32))

    eq_a = np.array(equity_curve)
    r = np.diff(eq_a) / np.maximum(1e-8, eq_a[:-1])
    pk = np.maximum.accumulate(eq_a)
    ret_pct = (eq_a[-1] / eq_a[0] - 1) * 100
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 1e-8 else 0.
    max_dd = float(np.min((eq_a - pk) / pk) * 100)

    last_act = model.get_action(np.concatenate(obs_h).astype(np.float32), deterministic=True)
    c_frac = 1.0 / (1.0 + np.exp(-np.clip(last_act[0], -5.0, 5.0)))

    return {
        "Reward Arm": arm_label,
        "Asset Universe": universe_name,
        "Return (%)": ret_pct,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd,
        "Cash Reserves (%)": c_frac * 100
    }


def execute_ablation_study():
    arms = [
        ('asymmetric', 'Arm A: Asymmetric Penalties'),
        ('symmetric_log', 'Arm B: Pure Symmetric Log-Growth'),
        ('log_moderate_risk', 'Arm C: Log-Growth + Mod Risk')
    ]

    all_results = []

    for mode, label in arms:
        model = train_ablation_model(reward_mode=mode, total_steps=100_000, seed=SEED)
        for u_name, u_cfg in GLOBAL_UNIVERSES.items():
            metrics = evaluate_universe(model, u_name, u_cfg, arm_label=label)
            all_results.append(metrics)

    df_res = pd.DataFrame(all_results)
    print("═" * 110)
    print(" 🏆 FINAL RAI v8.2 REWARD ABLATION STUDY LEADERBOARD (PROCEDURAL WORLD ENGINE)")
    print("═" * 110)
    print(df_res.to_string(index=False))
    print("═" * 110 + "\n")


if __name__ == "__main__":
    execute_ablation_study()
