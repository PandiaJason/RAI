"""
═══════════════════════════════════════════════════════════════════════════════
  RAI v7 — BENCHMARK-BEATING AGENT
  ════════════════════════════════
  Key changes from v6:
    1. Alpha reward: beat SPY/equal-weight, not just grow wealth
    2. Richer observations: momentum, volatility, trend signals
    3. Correlated synthetic worlds: stocks cluster, bonds inverse
    4. Concentration-capable: can go all-in on best assets
    5. Larger model: handles richer input
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, time, json, hashlib, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "v7_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  V7 SYNTHETIC ENVIRONMENT — Correlated Assets + Alpha Reward
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaEnvV7(gym.Env):
    """
    Key innovations:
    - Assets have realistic correlations (equity cluster, bond inverse)
    - Observation includes momentum + volatility + trend signals
    - Reward = alpha over equal-weight benchmark + risk penalty
    """

    REGIMES = {
        'strong_bull':  {'drift': (0.25, 0.60),  'vol': (0.10, 0.18), 'corr': (0.3, 0.6)},
        'mild_bull':    {'drift': (0.10, 0.25),  'vol': (0.12, 0.20), 'corr': (0.2, 0.5)},
        'sideways':     {'drift': (-0.05, 0.08), 'vol': (0.14, 0.22), 'corr': (0.1, 0.4)},
        'bear_crash':   {'drift': (-0.50, -0.15),'vol': (0.30, 0.55), 'corr': (0.5, 0.85)},
        'recovery':     {'drift': (0.30, 0.80),  'vol': (0.20, 0.35), 'corr': (0.3, 0.5)},
    }

    def __init__(self, num_assets=10, history_len=30, episode_len=504,
                 initial_cash=10000.0, fee=0.001):
        super().__init__()
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee

        # Expanded features: price + return + momentum_20d + vol_20d + cash + dd + trend + vol_regime
        # Per asset: 4 features (norm_price, log_return, momentum_20d, vol_20d)
        # Global: 4 features (cash_frac, drawdown, market_trend, vol_regime)
        self.features_per_step = 4 * num_assets + 4
        self.observation_space = spaces.Box(-np.inf, np.inf,
            (history_len * self.features_per_step,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (num_assets + 1,), np.float32)
        self.reset()

    def _generate_correlated_prices(self):
        """Generate prices with realistic cross-asset correlations."""
        T = self.episode_len + self.history_len + 50
        n_seg = self.np_random.integers(3, 8)
        keys = list(self.REGIMES.keys())
        seq = [keys[self.np_random.integers(len(keys))] for _ in range(n_seg)]
        dur = self.np_random.dirichlet(np.ones(n_seg) * 2.0) * T
        dur = np.maximum(dur.astype(int), 30)
        dur[-1] = T - sum(dur[:-1])

        # Asset types: first 6 = "equities" (correlated), last 4 = "defensives" (inverse)
        n_eq = 6
        n_def = self.num_assets - n_eq

        prices = np.zeros((T, self.num_assets), np.float64)
        init_prices = self.np_random.uniform(20., 300., self.num_assets)

        day = 0
        for reg_name, d in zip(seq, dur):
            reg = self.REGIMES[reg_name]
            drift_range = reg['drift']
            vol_range = reg['vol']
            corr_range = reg['corr']

            # Generate correlated noise using Cholesky
            base_corr = self.np_random.uniform(*corr_range)

            # Equity correlation matrix
            C = np.full((self.num_assets, self.num_assets), 0.0)
            for i in range(self.num_assets):
                C[i, i] = 1.0

            # Equities correlated with each other
            for i in range(n_eq):
                for j in range(i+1, n_eq):
                    c = base_corr * self.np_random.uniform(0.7, 1.0)
                    C[i, j] = C[j, i] = c

            # Defensives slightly correlated with each other
            for i in range(n_eq, self.num_assets):
                for j in range(i+1, self.num_assets):
                    c = base_corr * self.np_random.uniform(0.2, 0.5)
                    C[i, j] = C[j, i] = c

            # Equities vs defensives: negative correlation (flight to safety)
            for i in range(n_eq):
                for j in range(n_eq, self.num_assets):
                    c = -base_corr * self.np_random.uniform(0.1, 0.4)
                    C[i, j] = C[j, i] = c

            # Ensure positive definite
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 0.01:
                C += (0.02 - eigvals.min()) * np.eye(self.num_assets)

            L = np.linalg.cholesky(C)

            # Per-asset drift and vol
            drifts = np.zeros(self.num_assets)
            vols = np.zeros(self.num_assets)
            for i in range(n_eq):
                drifts[i] = self.np_random.uniform(*drift_range) + self.np_random.uniform(-0.05, 0.05)
                vols[i] = self.np_random.uniform(*vol_range) * self.np_random.uniform(0.9, 1.1)
            for i in range(n_eq, self.num_assets):
                # Defensive assets: lower drift, lower vol, inverse direction in crashes
                if 'bear' in reg_name or 'crash' in reg_name:
                    drifts[i] = self.np_random.uniform(0.02, 0.15)  # Defensives rally in crashes
                else:
                    drifts[i] = self.np_random.uniform(-0.02, 0.08)  # Low return normally
                vols[i] = self.np_random.uniform(0.06, 0.14)

            mu_daily = drifts / 252.
            sigma_daily = vols / np.sqrt(252.)

            for _ in range(max(0, min(d, T - day))):
                z = self.np_random.standard_normal(self.num_assets)
                corr_z = L @ z
                for a in range(self.num_assets):
                    if day == 0:
                        prices[day, a] = init_prices[a]
                    else:
                        prices[day, a] = max(0.01, prices[day-1, a] *
                            np.exp((mu_daily[a] - 0.5*sigma_daily[a]**2) +
                                   sigma_daily[a] * corr_z[a]))
                day += 1
                if day >= T:
                    break
            if day >= T:
                break

        # Fill remaining
        while day < T:
            prices[day] = prices[day-1]
            day += 1

        return prices

    def _compute_features(self, t):
        """Rich feature vector at time t."""
        p = self.prices[t]
        pp = self.prices[max(0, t-1)]

        # Normalized prices
        norm_p = p / self.ref_prices

        # Log returns
        log_r = np.log(p / np.maximum(1e-4, pp))

        # 20-day momentum (if we have enough history)
        if t >= 20:
            mom20 = np.log(p / np.maximum(1e-4, self.prices[t-20]))
        else:
            mom20 = np.zeros(self.num_assets)

        # 20-day rolling volatility
        if t >= 20:
            log_rets = np.log(self.prices[max(1,t-19):t+1] /
                             np.maximum(1e-4, self.prices[max(0,t-20):t]))
            vol20 = np.std(log_rets, axis=0) * np.sqrt(252)
        else:
            vol20 = np.ones(self.num_assets) * 0.2

        # Global features
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        cash_frac = self.cash / w
        dd = np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1, 0)

        # Market trend: equal-weight index momentum
        if t >= 20:
            idx_now = np.mean(p / self.ref_prices)
            idx_20 = np.mean(self.prices[t-20] / self.ref_prices)
            market_trend = np.clip(np.log(idx_now / max(1e-4, idx_20)), -0.5, 0.5)
        else:
            market_trend = 0.0

        # Vol regime: average vol level
        vol_regime = np.clip(np.mean(vol20) / 0.3, 0, 2) - 1.0  # centered around normal

        features = np.concatenate([
            norm_p,              # 10: normalized prices
            log_r,               # 10: daily log returns
            mom20,               # 10: 20-day momentum
            np.clip(vol20, 0, 1),# 10: 20-day vol (clipped)
            [cash_frac],         # 1: cash fraction
            [dd],                # 1: drawdown
            [market_trend],      # 1: market trend
            [vol_regime],        # 1: vol regime
        ]).astype(np.float32)

        return features

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prices = self._generate_correlated_prices()
        self.start = self.history_len + 20  # Need 20 extra for momentum
        self.current_step = self.start
        self.ref_prices = self.prices[self.start].copy()

        # Start with equal-weight allocation
        self.cash = self.initial_cash * 0.05
        self.shares = (self.initial_cash * 0.95 / self.num_assets) / self.prices[self.start]
        self.peak_wealth = self.last_wealth = self.initial_cash
        self.steps_done = 0

        # Benchmark tracking: equal-weight buy-and-hold from same start
        self.bench_shares = (self.initial_cash / self.num_assets) / self.prices[self.start]
        self.bench_start_wealth = self.initial_cash

        # Build observation history
        self.obs_history = []
        for i in range(self.history_len):
            t = self.start - self.history_len + i
            self.obs_history.append(self._compute_features(t))

        return self._flat_obs(), {}

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _bench_wealth(self):
        return np.sum(self.bench_shares * self.prices[self.current_step])

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        # Decode action: action[0] = cash level, action[1:] = asset weights
        cl = np.clip(action[0] - 2.5, -8., 3.)
        target_cash = 1.0 / (1.0 + np.exp(-cl))
        target_invested = 1.0 - target_cash

        ea = np.exp(action[1:] - np.max(action[1:]))
        target_weights = (ea / ea.sum()) * target_invested

        p = self.prices[self.current_step]
        w = max(1e-4, self._wealth())

        current_weights = (self.shares * p) / w
        current_cash_frac = self.cash / w

        # Rebalance if needed (2% threshold — more active than v6's 3%)
        if abs(current_cash_frac - target_cash) + np.sum(np.abs(current_weights - target_weights)) > 0.02:
            tv = abs(self.cash - w * target_cash) + np.sum(np.abs(self.shares * p - w * target_weights))
            net = max(1e-4, w - tv * self.fee)
            self.cash = net * target_cash
            self.shares = (net * target_weights) / np.maximum(1e-4, p)

        self.current_step += 1
        self.steps_done += 1

        new_wealth = self._wealth()
        bench_wealth = self._bench_wealth()
        self.peak_wealth = max(self.peak_wealth, new_wealth)

        # ════════════════════════════════════════
        # V7 ALPHA REWARD
        # ════════════════════════════════════════
        port_return = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)

        # Benchmark return (equal-weight buy-and-hold)
        prev_bench = np.sum(self.bench_shares * self.prices[self.current_step - 1])
        bench_return = (bench_wealth - prev_bench) / max(1e-4, prev_bench)

        # Alpha = excess return over benchmark
        alpha = port_return - bench_return

        # Drawdown penalty
        dd = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)

        # Reward components:
        # 1. Alpha (excess return) — primary driver
        # 2. Absolute return — secondary
        # 3. Drawdown penalty — risk control
        reward = (
            alpha * 40.0 +           # Strong alpha incentive
            port_return * 15.0 +     # Still reward absolute returns
            dd * 3.0                 # Penalize drawdowns (dd is negative)
        )

        done = (self.current_step >= self.prices.shape[0] - 1 or
                self.steps_done >= self.episode_len)

        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._compute_features(self.current_step))

        return self._flat_obs(), reward, done, False, {}


# ═══════════════════════════════════════════════════════════════════════════════
#  V7 MODEL — Larger to handle richer input
# ═══════════════════════════════════════════════════════════════════════════════

class RAIv7Net(nn.Module):
    """
    RAI v7: Conv1D + Transformer with expanded features.
    ~85k parameters (up from v6's ~52k).
    """
    def __init__(self, history_len=30, features_per_step=44, action_dim=11,
                 embed_dim=96, nhead=4):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step

        # Conv1D: local pattern extraction from richer features
        self.conv1d = nn.Sequential(
            nn.Conv1d(features_per_step, 48, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(48, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        # Transformer: global temporal attention
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead,
            dim_feedforward=192, dropout=0.05, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

        # Heads
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 192),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
        )
        self.actor_head = nn.Linear(192, action_dim)
        self.critic_head = nn.Linear(192, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step)
        x = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.transformer(x)
        feat = self.fc(x.mean(dim=1))
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_ppo_v7(model, env, seed, total_steps=200_000):
    """PPO with higher step count for harder task."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)  # Slightly lower LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps//1024)

    obs, _ = env.reset(seed=seed)
    step = 0
    best_reward = -float('inf')

    while step < total_steps:
        obs_b, act_b, rew_b, val_b, logp_b = [], [], [], [], []

        for _ in range(1024):
            ot = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                m, v = model(ot)
                d = Normal(m, torch.exp(model.log_std))
                a = d.sample()
                lp = d.log_prob(a).sum(-1)

            an = a.squeeze(0).numpy()
            no, r, dn, _, _ = env.step(an)
            obs_b.append(obs); act_b.append(an); rew_b.append(r)
            val_b.append(v.item()); logp_b.append(lp.item())
            obs = no; step += 1
            if dn:
                obs, _ = env.reset()

        # GAE
        with torch.no_grad():
            _, nv = model(torch.FloatTensor(obs).unsqueeze(0))
            nv = nv.item()

        r_a = np.array(rew_b)
        v_a = np.array(val_b + [nv])
        delta = r_a + 0.99 * v_a[1:] - v_a[:-1]
        adv = np.zeros_like(r_a)
        gae = 0.
        for t in reversed(range(len(r_a))):
            gae = delta[t] + 0.99 * 0.95 * gae
            adv[t] = gae
        ret = adv + v_a[:-1]

        ot = torch.FloatTensor(np.array(obs_b))
        at = torch.FloatTensor(np.array(act_b))
        advt = torch.FloatTensor(adv)
        rett = torch.FloatTensor(ret)
        oldt = torch.FloatTensor(np.array(logp_b))
        advt = (advt - advt.mean()) / (advt.std() + 1e-8)

        # PPO epochs
        for _ in range(6):  # More epochs
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), 64):
                bi = idx[s:s+64]
                m2, v2 = model(ot[bi])
                d2 = Normal(m2, torch.exp(model.log_std))
                nlp = d2.log_prob(at[bi]).sum(-1)
                ratio = torch.exp(nlp - oldt[bi])

                # Clipped PPO loss
                loss_pi = -torch.min(
                    ratio * advt[bi],
                    torch.clamp(ratio, 0.8, 1.2) * advt[bi]
                ).mean()

                # Value loss
                loss_v = 0.5 * F.mse_loss(v2.squeeze(-1), rett[bi])

                # Entropy bonus (encourage exploration)
                entropy = d2.entropy().mean()

                loss = loss_pi + loss_v - 0.01 * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        scheduler.step()

        ep_reward = np.sum(r_a)
        if step % 10240 == 0:
            print(f"    Step {step:>7d}/{total_steps} | Reward: {ep_reward:>8.1f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION ON REAL DATA
# ═══════════════════════════════════════════════════════════════════════════════

TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]

def eval_on_real(model, prices_df, fee_bps=5):
    """Evaluate v7 model on real prices."""
    prices = prices_df.values
    T, N = prices.shape
    if T < 55:
        return None

    cash = 500.0
    shares = (9500.0 / N) / prices[50]
    ref_p = prices[50].copy()
    peak = 10000.0
    eq = [10000.0]

    # Benchmark: buy-and-hold equal weight
    bench_sh = (10000.0 / N) / prices[50]
    bench_eq = [10000.0]

    # SPY benchmark
    spy_idx = list(prices_df.columns).index('SPY') if 'SPY' in prices_df.columns else 5
    spy_eq = [10000.0]

    obs_h = []
    for i in range(30):
        t = 20 + i  # Start from day 20 (need momentum lookback)
        p = prices[t]
        pp = prices[max(0, t-1)]
        norm_p = p / ref_p
        log_r = np.log(p / np.maximum(1e-4, pp))

        if t >= 20:
            mom20 = np.log(p / np.maximum(1e-4, prices[t-20]))
            log_rets = np.log(prices[max(1,t-19):t+1] / np.maximum(1e-4, prices[max(0,t-20):t]))
            vol20 = np.std(log_rets, axis=0) * np.sqrt(252)
        else:
            mom20 = np.zeros(N)
            vol20 = np.ones(N) * 0.2

        w = max(1e-4, cash + np.sum(shares * p))
        np_ = np.pad(norm_p, (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(log_r, (0, max(0, 10-N)), constant_values=0.)[:10]
        m20 = np.pad(mom20, (0, max(0, 10-N)), constant_values=0.)[:10]
        v20 = np.pad(np.clip(vol20, 0, 1), (0, max(0, 10-N)), constant_values=0.2)[:10]

        idx_mom = np.log(np.mean(p/ref_p) / max(1e-4, np.mean(prices[max(0,t-20)]/ref_p))) if t >= 20 else 0.
        vol_reg = np.clip(np.mean(vol20)/0.3, 0, 2) - 1.0

        feat = np.concatenate([np_, lr, m20, v20, [0.05, 0., np.clip(idx_mom, -0.5, 0.5), vol_reg]]).astype(np.float32)
        obs_h.append(feat)

    for t in range(50, T):
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

        if abs(ccf - tc) + np.sum(np.abs(caw - taw)) > 0.02:
            tv = abs(cash - w*tc) + np.sum(np.abs(shares*p - w*taw))
            fee = fee_bps / 10000.0
            net = max(1e-4, w - tv*fee)
            cash = net * tc
            shares = (net * taw) / np.maximum(1e-4, p)

        nw = cash + np.sum(shares * p)
        peak = max(peak, nw)
        eq.append(nw)

        # Benchmarks
        bench_eq.append(np.sum(bench_sh * p))
        spy_eq.append(10000.0 * p[spy_idx] / prices[50, spy_idx])

        # Update obs
        pp = prices[t-1]
        norm_p = p / ref_p
        log_r = np.log(p / np.maximum(1e-4, pp))
        mom20 = np.log(p / np.maximum(1e-4, prices[t-20])) if t >= 20 else np.zeros(N)
        if t >= 20:
            log_rets = np.log(prices[max(1,t-19):t+1] / np.maximum(1e-4, prices[max(0,t-20):t]))
            vol20 = np.std(log_rets, axis=0) * np.sqrt(252)
        else:
            vol20 = np.ones(N) * 0.2

        np_ = np.pad(norm_p, (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(log_r, (0, max(0, 10-N)), constant_values=0.)[:10]
        m20 = np.pad(mom20, (0, max(0, 10-N)), constant_values=0.)[:10]
        v20 = np.pad(np.clip(vol20, 0, 1), (0, max(0, 10-N)), constant_values=0.2)[:10]

        idx_mom = np.log(np.mean(p/ref_p) / max(1e-4, np.mean(prices[max(0,t-20)]/ref_p))) if t >= 20 else 0.
        vol_reg = np.clip(np.mean(vol20)/0.3, 0, 2) - 1.0
        dd = np.clip((nw - peak) / max(1e-4, peak), -1, 0)

        feat = np.concatenate([np_, lr, m20, v20, [cash/max(1e-4,nw), dd,
                np.clip(idx_mom, -0.5, 0.5), vol_reg]]).astype(np.float32)
        obs_h.pop(0)
        obs_h.append(feat)

    # Compute metrics
    eq_a = np.array(eq)
    bench_a = np.array(bench_eq)
    spy_a = np.array(spy_eq)

    def metrics(e):
        r = np.diff(e) / np.maximum(1e-8, e[:-1])
        pk = np.maximum.accumulate(e)
        return {
            'return_pct': float((e[-1]/e[0]-1)*100),
            'sharpe': float(np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r) > 1e-8 else 0.,
            'max_dd_pct': float(np.min((e-pk)/pk)*100),
            'final': float(e[-1]),
            'vol_pct': float(np.std(r)*np.sqrt(252)*100),
        }

    return {
        'rai_v7': metrics(eq_a),
        'equal_weight': metrics(bench_a),
        'spy': metrics(spy_a),
        'alpha_vs_ew': float((eq_a[-1]/eq_a[0]-1)*100 - (bench_a[-1]/bench_a[0]-1)*100),
        'alpha_vs_spy': float((eq_a[-1]/eq_a[0]-1)*100 - (spy_a[-1]/spy_a[0]-1)*100),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    W = 80
    print("="*W)
    print("  RAI v7 — BENCHMARK-BEATING AGENT")
    print("  Alpha reward + correlated assets + momentum signals")
    print("="*W, flush=True)

    N_SEEDS = 5
    TOTAL_STEPS = 200_000

    # ── Train 5 seeds in parallel ──
    print(f"\n  Training {N_SEEDS} seeds × 200k steps...", flush=True)

    def train_worker(seed):
        t0 = time.time()
        model = RAIv7Net()
        env = AlphaEnvV7()
        print(f"  [Seed {seed}] Starting training ({sum(p.numel() for p in model.parameters()):,} params)...", flush=True)
        model = train_ppo_v7(model, env, seed=seed, total_steps=TOTAL_STEPS)
        path = os.path.join(RESULTS_DIR, f"rai_v7_seed_{seed:02d}.pt")
        torch.save(model.state_dict(), path)
        print(f"  [Seed {seed}] Done in {time.time()-t0:.0f}s → {path}", flush=True)
        return seed, path, time.time()-t0

    t0 = time.time()
    with mp.Pool(processes=min(N_SEEDS, 8)) as pool:
        results = pool.map(train_worker, range(1, N_SEEDS+1))
    print(f"\n  ✓ Training complete: {time.time()-t0:.0f}s total", flush=True)

    # ── Download real data ──
    print(f"\n  Downloading real market data...", flush=True)
    import yfinance as yf

    test_periods = {
        "2020-2024 OOS": ("2020-01-01", "2024-06-01"),
        "2024-2026 Holdout": ("2024-06-01", "2026-08-08"),
        "Full 2020-2026": ("2020-01-01", "2026-08-08"),
    }

    # ── Evaluate ──
    for period_name, (start, end) in test_periods.items():
        print(f"\n{'═'*W}")
        print(f"  TESTING: {period_name}")
        print(f"{'═'*W}", flush=True)

        df = yf.download(TICKERS, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        df = df[TICKERS].dropna()
        print(f"  Downloaded: {len(df)} days ({df.index[0].date()} → {df.index[-1].date()})", flush=True)

        if len(df) < 55:
            print("  ⚠ Too few days, skipping"); continue

        print(f"\n  {'Seed':<8} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'α vs EW':>10} {'α vs SPY':>10}", flush=True)
        print(f"  {'-'*60}", flush=True)

        all_results = []
        for seed in range(1, N_SEEDS+1):
            path = os.path.join(RESULTS_DIR, f"rai_v7_seed_{seed:02d}.pt")
            if not os.path.exists(path): continue
            model = RAIv7Net()
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()

            r = eval_on_real(model, df)
            if r is None: continue
            all_results.append(r)

            v7 = r['rai_v7']
            print(f"  {seed:<8} {v7['return_pct']:>+9.2f}% {v7['sharpe']:>7.3f} "
                  f"{v7['max_dd_pct']:>+9.2f}% {r['alpha_vs_ew']:>+9.2f}% "
                  f"{r['alpha_vs_spy']:>+9.2f}%", flush=True)

        if all_results:
            print(f"\n  {'─'*60}")
            rets = [r['rai_v7']['return_pct'] for r in all_results]
            sharpes = [r['rai_v7']['sharpe'] for r in all_results]
            alphas_ew = [r['alpha_vs_ew'] for r in all_results]
            alphas_spy = [r['alpha_vs_spy'] for r in all_results]
            print(f"  {'Mean':<8} {np.mean(rets):>+9.2f}% {np.mean(sharpes):>7.3f} "
                  f"{'':>10} {np.mean(alphas_ew):>+9.2f}% {np.mean(alphas_spy):>+9.2f}%")
            print(f"  {'Std':<8} {np.std(rets):>9.2f}% {np.std(sharpes):>7.3f}")

            print(f"\n  Benchmarks:")
            print(f"  {'EqualWt':<8} {all_results[0]['equal_weight']['return_pct']:>+9.2f}% "
                  f"{all_results[0]['equal_weight']['sharpe']:>7.3f} "
                  f"{all_results[0]['equal_weight']['max_dd_pct']:>+9.2f}%")
            print(f"  {'SPY':<8} {all_results[0]['spy']['return_pct']:>+9.2f}% "
                  f"{all_results[0]['spy']['sharpe']:>7.3f} "
                  f"{all_results[0]['spy']['max_dd_pct']:>+9.2f}%")

            beats_ew = sum(1 for a in alphas_ew if a > 0)
            beats_spy = sum(1 for a in alphas_spy if a > 0)
            print(f"\n  Beats Equal Weight: {beats_ew}/{len(all_results)}")
            print(f"  Beats SPY:          {beats_spy}/{len(all_results)}")

    print(f"\n{'═'*W}")
    print(f"  ✅ RAI v7 EXPERIMENT COMPLETE")
    print(f"{'═'*W}", flush=True)


if __name__ == "__main__":
    main()
