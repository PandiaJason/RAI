"""
═══════════════════════════════════════════════════════════════════════════════
  SYNTHETIC GENERATOR ABLATION LADDER PROTOCOL
  ═══════════════════════════════════════════════════════════════════════════════
  Research Question:
    "Which statistical properties of synthetic market data, if any,
     are necessary and sufficient for zero-shot real-market transfer?"

  Ablation Ladder Levels:
    Level 0: Pure Random Walk (GBM baseline)
    Level 1: Fat Tails (Student-t heavy-tailed return distributions)
    Level 2: Volatility Clustering (GARCH(1,1) / stochastic volatility)
    Level 3: Static Cross-Asset Correlation (Cholesky empirical covariance)
    Level 4: Dynamic Correlation & Volatility (Regime-dependent covariance)
    Level 5: Jump-Diffusion (Poisson jump process / tail discontinuities)
    Level 6: Combined Realistic Simulator (All stylized facts combined)

  Experimental Constraints:
    - Identical RAI v6 hybrid network architecture
    - Identical PPO hyperparameters & 100k training steps
    - 5 seeds per generator level (35 total models)
    - 8-core parallel training execution
    - Identical real-market evaluation & untouched holdout test sets
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, hashlib, warnings, multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
from scipy import stats

warnings.filterwarnings('ignore')
torch.set_num_threads(1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "ablation_ladder")
NUM_WORKERS = 8
N_SEEDS = 5
TOTAL_STEPS = 100_000

os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE (Strictly identical RAI v6 Hybrid)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepEndToEndTradingNet(nn.Module):
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
        feat = self.fc_features(x_trans.mean(dim=1))
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  BASE SYNTHETIC ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class BaseLadderEnv(gym.Env):
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

    def _generate_raw_prices(self):
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prices = self._generate_raw_prices()
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.05
        self.shares = (self.initial_cash * 0.95 / self.num_assets) / self.prices[self.current_step]
        self.peak_wealth = self.last_wealth = self.initial_cash
        self.steps_done = 0
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs(), {}

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p, pp = self.prices[t], self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        return np.concatenate([p/self.prices[self.start], np.log(p/np.maximum(1e-4, pp)),
                               [self.cash/w, np.clip((w-self.peak_wealth)/max(1e-4, self.peak_wealth), -1, 0)]]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        cl = np.clip(action[0] - 2.5, -8., 3.)
        tc = 1./(1.+np.exp(-cl)); ts = 1.-tc
        ea = np.exp(action[1:] - np.max(action[1:])); taw = (ea/ea.sum()) * ts
        p, w = self.prices[self.current_step], max(1e-4, self._wealth())
        caw, ccf = (self.shares*p)/w, self.cash/w
        if abs(ccf-tc)+np.sum(np.abs(caw-taw)) > 0.03:
            tv = abs(self.cash-w*tc)+np.sum(np.abs(self.shares*p-w*taw))
            net = max(1e-4, w-tv*self.fee); self.cash = net*tc; self.shares = (net*taw)/np.maximum(1e-4, p)
        self.current_step += 1; self.steps_done += 1
        nw = self._wealth(); self.peak_wealth = max(self.peak_wealth, nw)
        reward = ((nw-self.last_wealth)/max(1e-4, self.last_wealth)) * 20.
        done = self.current_step >= self.prices.shape[0]-1 or self.steps_done >= self.episode_len
        self.last_wealth = nw; self.obs_history.pop(0); self.obs_history.append(self._obs_at(self.current_step))
        return self._flat_obs(), reward, done, False, {}


# ═══════════════════════════════════════════════════════════════════════════════
#  ABLATION LADDER GENERATORS (LEVELS 0 - 6)
# ═══════════════════════════════════════════════════════════════════════════════

class Level0_GBM(BaseLadderEnv):
    """Level 0: Pure Geometric Brownian Motion (Uncorrelated, Normal innovation)."""
    def _generate_raw_prices(self):
        T = self.episode_len + self.history_len + 10
        prices = np.zeros((T, self.num_assets), np.float64)
        for a in range(self.num_assets):
            p = self.np_random.uniform(20., 300.)
            mu = self.np_random.uniform(-0.10, 0.25) / 252.
            sigma = self.np_random.uniform(0.10, 0.30) / np.sqrt(252.)
            series = [p]
            for _ in range(T-1):
                z = self.np_random.standard_normal()
                p = max(0.01, p * np.exp((mu - 0.5*sigma**2) + sigma * z))
                series.append(p)
            prices[:, a] = series[:T]
        return prices


class Level1_FatTails(BaseLadderEnv):
    """Level 1: Fat-Tailed Innovations (Student-t, df=4)."""
    def _generate_raw_prices(self):
        T = self.episode_len + self.history_len + 10
        prices = np.zeros((T, self.num_assets), np.float64)
        for a in range(self.num_assets):
            p = self.np_random.uniform(20., 300.)
            mu = self.np_random.uniform(-0.10, 0.25) / 252.
            sigma = self.np_random.uniform(0.10, 0.30) / np.sqrt(252.)
            series = [p]
            df = 4.0  # Heavy tails
            scale = np.sqrt((df - 2) / df)  # Normalize variance to 1
            for _ in range(T-1):
                t_noise = self.np_random.standard_t(df) * scale
                p = max(0.01, p * np.exp((mu - 0.5*sigma**2) + sigma * t_noise))
                series.append(p)
            prices[:, a] = series[:T]
        return prices


class Level2_VolClustering(BaseLadderEnv):
    """Level 2: Volatility Clustering (GARCH(1,1) dynamics)."""
    def _generate_raw_prices(self):
        T = self.episode_len + self.history_len + 10
        prices = np.zeros((T, self.num_assets), np.float64)
        for a in range(self.num_assets):
            p = self.np_random.uniform(20., 300.)
            mu = self.np_random.uniform(-0.10, 0.25) / 252.
            base_vol = self.np_random.uniform(0.10, 0.30) / np.sqrt(252.)

            # GARCH(1,1) parameters: omega + alpha + beta = 1
            omega = 0.05 * (base_vol**2)
            alpha = self.np_random.uniform(0.05, 0.15)
            beta = self.np_random.uniform(0.75, 0.88)

            current_var = base_vol**2
            last_eps_sq = base_vol**2
            series = [p]

            for _ in range(T-1):
                current_var = omega + alpha * last_eps_sq + beta * current_var
                sigma = np.sqrt(max(1e-8, current_var))
                z = self.np_random.standard_normal()
                eps = sigma * z
                last_eps_sq = eps**2

                p = max(0.01, p * np.exp((mu - 0.5*sigma**2) + eps))
                series.append(p)
            prices[:, a] = series[:T]
        return prices


class Level3_StaticCorrelation(BaseLadderEnv):
    """Level 3: Static Empirical Cross-Asset Correlation Structure."""
    def _generate_raw_prices(self):
        T = self.episode_len + self.history_len + 10
        prices = np.zeros((T, self.num_assets), np.float64)

        # 6 equities, 4 defensives
        n_eq = 6
        base_corr = self.np_random.uniform(0.3, 0.6)
        C = np.eye(self.num_assets)
        for i in range(n_eq):
            for j in range(i+1, n_eq):
                C[i, j] = C[j, i] = base_corr * self.np_random.uniform(0.7, 1.0)
        for i in range(n_eq, self.num_assets):
            for j in range(i+1, self.num_assets):
                C[i, j] = C[j, i] = base_corr * self.np_random.uniform(0.2, 0.5)
        for i in range(n_eq):
            for j in range(n_eq, self.num_assets):
                C[i, j] = C[j, i] = -base_corr * self.np_random.uniform(0.1, 0.4)

        eigvals = np.linalg.eigvalsh(C)
        if eigvals.min() < 0.01:
            C += (0.02 - eigvals.min()) * np.eye(self.num_assets)
        L = np.linalg.cholesky(C)

        init_p = self.np_random.uniform(20., 300., self.num_assets)
        drifts = self.np_random.uniform(-0.10, 0.25, self.num_assets) / 252.
        vols = self.np_random.uniform(0.10, 0.30, self.num_assets) / np.sqrt(252.)

        prices[0] = init_p
        for t in range(1, T):
            z = self.np_random.standard_normal(self.num_assets)
            cz = L @ z
            prices[t] = np.maximum(0.01, prices[t-1] * np.exp((drifts - 0.5*vols**2) + vols * cz))

        return prices


class Level4_DynamicCorrelationVol(BaseLadderEnv):
    """Level 4: Dynamic Regime-Switching Correlation and Volatility."""
    def _generate_raw_prices(self):
        T = self.episode_len + self.history_len + 10
        prices = np.zeros((T, self.num_assets), np.float64)

        regimes = [
            {'drift': (0.15, 0.40), 'vol': (0.10, 0.18), 'corr': 0.3}, # Bull
            {'drift': (-0.40, -0.10), 'vol': (0.25, 0.50), 'corr': 0.7}, # Crash / High Corr
            {'drift': (-0.02, 0.08), 'vol': (0.12, 0.22), 'corr': 0.2}, # Sideways
        ]

        init_p = self.np_random.uniform(20., 300., self.num_assets)
        prices[0] = init_p
        curr_reg = 0
        reg_len = 0

        for t in range(1, T):
            if reg_len <= 0:
                curr_reg = self.np_random.integers(0, len(regimes))
                reg_len = self.np_random.integers(30, 120)
            reg_len -= 1

            reg = regimes[curr_reg]
            base_corr = reg['corr']

            C = np.eye(self.num_assets)
            for i in range(self.num_assets):
                for j in range(i+1, self.num_assets):
                    C[i, j] = C[j, i] = base_corr * self.np_random.uniform(0.8, 1.0)
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 0.01:
                C += (0.02 - eigvals.min()) * np.eye(self.num_assets)
            L = np.linalg.cholesky(C)

            drift = self.np_random.uniform(*reg['drift']) / 252.
            vol = self.np_random.uniform(*reg['vol']) / np.sqrt(252.)

            z = self.np_random.standard_normal(self.num_assets)
            cz = L @ z
            prices[t] = np.maximum(0.01, prices[t-1] * np.exp((drift - 0.5*vol**2) + vol * cz))

        return prices


class Level5_JumpDiffusion(BaseLadderEnv):
    """Level 5: Merton Jump-Diffusion (Poisson sudden market shocks)."""
    def _generate_raw_prices(self):
        T = self.episode_len + self.history_len + 10
        prices = np.zeros((T, self.num_assets), np.float64)

        jump_lambda = 0.02  # ~5 jumps per year
        jump_mean = -0.05
        jump_std = 0.08

        for a in range(self.num_assets):
            p = self.np_random.uniform(20., 300.)
            mu = self.np_random.uniform(-0.10, 0.25) / 252.
            sigma = self.np_random.uniform(0.10, 0.30) / np.sqrt(252.)
            series = [p]

            for _ in range(T-1):
                z = self.np_random.standard_normal()
                # Poisson jump occurrence
                n_jumps = self.np_random.poisson(jump_lambda)
                jump_factor = 0.0
                if n_jumps > 0:
                    jump_factor = np.sum(self.np_random.normal(jump_mean, jump_std, size=n_jumps))

                p = max(0.01, p * np.exp((mu - 0.5*sigma**2) + sigma * z + jump_factor))
                series.append(p)
            prices[:, a] = series[:T]
        return prices


class Level6_CombinedRealistic(BaseLadderEnv):
    """Level 6: Full Combined Simulator (Fat tails + GARCH vol clustering + Dynamic correlation + Jumps)."""
    def _generate_raw_prices(self):
        T = self.episode_len + self.history_len + 10
        prices = np.zeros((T, self.num_assets), np.float64)

        n_eq = 6
        df = 4.0
        scale_t = np.sqrt((df - 2) / df)
        jump_lambda = 0.015

        # GARCH states per asset
        base_vols = self.np_random.uniform(0.10, 0.30, self.num_assets) / np.sqrt(252.)
        current_vars = base_vols**2
        last_eps_sq = base_vols**2
        omegas = 0.05 * current_vars
        alphas = self.np_random.uniform(0.05, 0.15, self.num_assets)
        betas = self.np_random.uniform(0.75, 0.88, self.num_assets)

        drifts = self.np_random.uniform(-0.05, 0.25, self.num_assets) / 252.
        prices[0] = self.np_random.uniform(20., 300., self.num_assets)

        for t in range(1, T):
            # Dynamic correlation matrix
            base_corr = self.np_random.uniform(0.2, 0.6)
            C = np.eye(self.num_assets)
            for i in range(n_eq):
                for j in range(i+1, n_eq):
                    C[i, j] = C[j, i] = base_corr * self.np_random.uniform(0.7, 1.0)
            for i in range(n_eq, self.num_assets):
                for j in range(i+1, self.num_assets):
                    C[i, j] = C[j, i] = base_corr * self.np_random.uniform(0.2, 0.5)
            for i in range(n_eq):
                for j in range(n_eq, self.num_assets):
                    C[i, j] = C[j, i] = -base_corr * self.np_random.uniform(0.1, 0.4)

            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 0.01:
                C += (0.02 - eigvals.min()) * np.eye(self.num_assets)
            L = np.linalg.cholesky(C)

            # GARCH update
            current_vars = omegas + alphas * last_eps_sq + betas * current_vars
            sigmas = np.sqrt(np.maximum(1e-8, current_vars))

            # Fat-tailed correlated noise
            z_t = self.np_random.standard_t(df, size=self.num_assets) * scale_t
            cz = L @ z_t
            eps = sigmas * cz
            last_eps_sq = eps**2

            # Poisson jump
            jumps = np.zeros(self.num_assets)
            if self.np_random.random() < jump_lambda:
                jumps = self.np_random.normal(-0.06, 0.08, size=self.num_assets)

            prices[t] = np.maximum(0.01, prices[t-1] * np.exp((drifts - 0.5*sigmas**2) + eps + jumps))

        return prices


LADDER_REGISTRY = {
    "Level_0_GBM": Level0_GBM,
    "Level_1_FatTails": Level1_FatTails,
    "Level_2_VolClustering": Level2_VolClustering,
    "Level_3_StaticCorr": Level3_StaticCorrelation,
    "Level_4_DynamicCorrVol": Level4_DynamicCorrelationVol,
    "Level_5_JumpDiffusion": Level5_JumpDiffusion,
    "Level_6_CombinedRealistic": Level6_CombinedRealistic,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO TRAINING WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def train_ppo(model, env, seed, total_steps=100_000):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs, _ = env.reset(seed=seed); step = 0
    while step < total_steps:
        obs_b, act_b, rew_b, val_b, logp_b = [], [], [], [], []
        for _ in range(1024):
            ot = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                m, v = model(ot); d = Normal(m, torch.exp(model.log_std))
                a = d.sample(); lp = d.log_prob(a).sum(-1)
            an = a.squeeze(0).numpy(); no, r, dn, _, _ = env.step(an)
            obs_b.append(obs); act_b.append(an); rew_b.append(r)
            val_b.append(v.item()); logp_b.append(lp.item())
            obs = no; step += 1
            if dn: obs, _ = env.reset()
        with torch.no_grad():
            _, nv = model(torch.FloatTensor(obs).unsqueeze(0)); nv = nv.item()
        r_a = np.array(rew_b); v_a = np.array(val_b+[nv])
        delta = r_a + 0.99*v_a[1:] - v_a[:-1]
        adv = np.zeros_like(r_a); gae = 0.
        for t in reversed(range(len(r_a))): gae = delta[t]+0.99*0.95*gae; adv[t] = gae
        ret = adv + v_a[:-1]
        ot = torch.FloatTensor(np.array(obs_b)); at = torch.FloatTensor(np.array(act_b))
        advt = torch.FloatTensor(adv); rett = torch.FloatTensor(ret); oldt = torch.FloatTensor(np.array(logp_b))
        advt = (advt - advt.mean())/(advt.std()+1e-8)
        for _ in range(4):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), 64):
                bi = idx[s:s+64]; m2, v2 = model(ot[bi])
                d2 = Normal(m2, torch.exp(model.log_std)); nlp = d2.log_prob(at[bi]).sum(-1)
                ratio = torch.exp(nlp - oldt[bi])
                loss = -torch.min(ratio*advt[bi], torch.clamp(ratio, .8, 1.2)*advt[bi]).mean() + .5*F.mse_loss(v2.squeeze(-1), rett[bi])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), .5); opt.step()
    return model

def _worker_train_ladder(args):
    level_name, seed, path, total_steps = args
    if os.path.exists(path):
        return (level_name, seed, path, "exists", 0.)
    t0 = time.time()
    model = DeepEndToEndTradingNet()
    env = LADDER_REGISTRY[level_name]()
    model = train_ppo(model, env, seed=seed, total_steps=total_steps)
    torch.save(model.state_dict(), path)
    return (level_name, seed, path, "trained", time.time()-t0)


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(eq_list):
    eq = np.array(eq_list, np.float64)
    if len(eq) < 2: return {"final": eq[-1], "return_pct": 0, "vol_pct": 0, "sharpe": 0, "max_dd_pct": 0}
    r = (eq[1:]-eq[:-1])/np.maximum(1e-8, eq[:-1])
    pk = np.maximum.accumulate(eq)
    return {"final": float(eq[-1]), "return_pct": float((eq[-1]/eq[0]-1)*100),
            "vol_pct": float(np.std(r)*np.sqrt(252)*100),
            "sharpe": float(np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r) > 1e-8 else 0.,
            "max_dd_pct": float(np.min((eq-pk)/pk)*100)}

def eval_model_on_prices(model, prices_raw, fee_bps=5, slippage_pct=0.02):
    T, N = prices_raw.shape
    if T < 35: return compute_metrics([10000.])
    fee = fee_bps/10000.; cash = 500.; init_p = prices_raw[30]
    shares = (9500./N)/init_p; peak = 10000.; eq = [10000.]
    obs_h = []
    for t in range(30):
        p, pp = prices_raw[t], prices_raw[max(0, t-1)]
        np_ = np.pad(p/prices_raw[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.append(np.concatenate([np_, lr, [.05, 0.]]).astype(np.float32))
    for t in range(30, T):
        act = model.get_action(np.concatenate(obs_h).astype(np.float32), deterministic=True)
        cl = np.clip(act[0]-2.5, -8., 3.); tc = 1./(1.+np.exp(-cl)); ts = 1.-tc
        n = min(N, 10); ea = np.exp(act[1:1+n]-np.max(act[1:1+n])); taw = (ea/ea.sum())*ts
        p = prices_raw[t].copy()
        w = max(1e-4, cash+np.sum(shares*p)); caw = (shares*p)/w; ccf = cash/w
        if abs(ccf-tc)+np.sum(np.abs(caw-taw)) > 0.03:
            tv = abs(cash-w*tc)+np.sum(np.abs(shares*p-w*taw))
            net = max(1e-4, w-tv*fee); cash = net*tc; shares = (net*taw)/np.maximum(1e-4, p)
        nw = cash+np.sum(shares*p); peak = max(peak, nw); eq.append(nw)
        pp = prices_raw[t-1]
        np_ = np.pad(p/prices_raw[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.pop(0); obs_h.append(np.concatenate([np_, lr, [cash/max(1e-4,nw), np.clip((nw-peak)/max(1e-4,peak),-1,0)]]).astype(np.float32))
    return compute_metrics(eq)

def comp_stats(vals):
    a = np.array(vals); n = len(a); mu = np.mean(a)
    sd = np.std(a, ddof=1) if n > 1 else 0.; se = sd/np.sqrt(n) if n > 1 else 0.
    ci = stats.t.interval(0.95, df=max(1,n-1), loc=mu, scale=max(1e-12,se)) if n > 1 else (mu, mu)
    return {"mean": float(mu), "std": float(sd), "ci95": [float(ci[0]), float(ci[1])],
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": n}

def welch_test(a, b):
    a, b = np.array(a), np.array(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    ps = np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    d = (a.mean()-b.mean())/ps if ps > 1e-8 else 0.
    return {"t": float(t), "p": float(p), "d": float(d)}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()
    W = 120
    print("="*W)
    print("  SYNTHETIC GENERATOR ABLATION LADDER PROTOCOL")
    print("  Testing 7 levels of generator complexity across 5 seeds each (35 models total)")
    print("="*W, flush=True)

    # 1. Dispatch Parallel Training
    jobs = []
    for level_name in LADDER_REGISTRY:
        for seed in range(1, N_SEEDS + 1):
            path = os.path.join(RESULTS_DIR, "models", f"{level_name}_seed_{seed:02d}.pt")
            jobs.append((level_name, seed, path, TOTAL_STEPS))

    print(f"\n  Dispatching {len(jobs)} training jobs across {NUM_WORKERS} workers...", flush=True)
    t1 = time.time()
    with mp.Pool(processes=NUM_WORKERS) as pool:
        outcomes = pool.map(_worker_train_ladder, jobs)

    trained = sum(1 for o in outcomes if o[4] == "trained")
    cached = sum(1 for o in outcomes if o[4] == "exists")
    print(f"  ✓ Training complete: {trained} trained, {cached} cached in {time.time()-t1:.0f}s", flush=True)

    # 2. Load Evaluation Datasets
    TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]

    local_test_path = os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "test_prices.csv")
    eval_data = {}

    if os.path.exists(local_test_path):
        df_test = pd.read_csv(local_test_path, index_col=0, parse_dates=True)
        eval_data["2020-2024_OOS"] = df_test.values
        print(f"    ✓ 2020-2024_OOS (Local CSV): {len(df_test)} days ({df_test.index[0].date()} → {df_test.index[-1].date()})", flush=True)

    import yfinance as yf
    extra_periods = {
        "2024-2026_Holdout": ("2024-06-01", "2026-08-08"),
        "Full_2020-2026": ("2020-01-01", "2026-08-08")
    }

    for p_name, (start_d, end_d) in extra_periods.items():
        try:
            df = yf.download(TICKERS, start=start_d, end=end_d, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            df = df[TICKERS].dropna()
            if len(df) >= 35:
                eval_data[p_name] = df.values
                print(f"    ✓ {p_name}: {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})", flush=True)
        except Exception as e:
            print(f"    ⚠ Could not download {p_name}: {e}", flush=True)

    # 3. Evaluate All Models Across All Test Datasets
    ladder_results = {level_name: {p_name: [] for p_name in eval_data} for level_name in LADDER_REGISTRY}

    for level_name in LADDER_REGISTRY:
        for seed in range(1, N_SEEDS + 1):
            path = os.path.join(RESULTS_DIR, "models", f"{level_name}_seed_{seed:02d}.pt")
            model = DeepEndToEndTradingNet()
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()

            for p_name, p_vals in eval_data.items():
                m = eval_model_on_prices(model, p_vals)
                ladder_results[level_name][p_name].append(m)

    # 4. Print & Format Results Tables
    print(f"\n{'═'*W}")
    print(f"  RESULTS SUMMARY: ABLATION LADDER TRANSFER PERFORMANCE")
    print(f"{'═'*W}")

    summary_json = {}

    for p_name in eval_data:
        print(f"\n  ► TEST PERIOD: {p_name}")
        print(f"  {'Level / Generator Description':<32} | {'Return (%)':<24} | {'Sharpe':<20} | {'Max DD (%)':<22}", flush=True)
        print(f"  {'-'*110}", flush=True)

        lvl0_rets = [m["return_pct"] for m in ladder_results["Level_0_GBM"][p_name]]
        lvl0_sh = [m["sharpe"] for m in ladder_results["Level_0_GBM"][p_name]]

        summary_json[p_name] = {}

        for level_name in LADDER_REGISTRY:
            ms = ladder_results[level_name][p_name]
            rets = [m["return_pct"] for m in ms]
            sharpes = [m["sharpe"] for m in ms]
            dds = [m["max_dd_pct"] for m in ms]

            r_stat = comp_stats(rets)
            s_stat = comp_stats(sharpes)
            d_stat = comp_stats(dds)

            # Welch's t-test vs Level 0 (GBM)
            sig_str = ""
            if level_name != "Level_0_GBM":
                tt = welch_test(rets, lvl0_rets)
                p_val = tt['p']
                sig_flag = "***" if p_val < .001 else "**" if p_val < .01 else "*" if p_val < .05 else "ns"
                sig_str = f" (p={p_val:.3f} {sig_flag}, d={tt['d']:+.2f})"

            print(f"  {level_name:<32} | {r_stat['mean']:>+6.2f}±{r_stat['std']:<4.2f}%{sig_str:<12} | "
                  f"{s_stat['mean']:>+5.2f}±{s_stat['std']:<4.2f} | {d_stat['mean']:>+6.2f}±{d_stat['std']:<4.2f}%", flush=True)

            summary_json[p_name][level_name] = {
                "return": r_stat,
                "sharpe": s_stat,
                "max_dd": d_stat,
            }

    # Save Master JSON Output
    out_path = os.path.join(RESULTS_DIR, "ladder_results.json")
    with open(out_path, 'w') as f:
        json.dump(summary_json, f, indent=2, default=str)

    print(f"\n{'═'*W}")
    print(f"  ✅ ABLATION LADDER EXPERIMENT COMPLETE — {time.time()-T0:.1f} seconds")
    print(f"  Results saved to: {out_path}")
    print(f"{'═'*W}\n", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
