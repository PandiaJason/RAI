"""
====================================================================================================
⚡ RAI v6 GPU-ACCELERATED KAGGLE ENGINE (CUDA / PyTorch GPU Enabled)
====================================================================================================
Overview:
  This script automatically detects and utilizes Kaggle's GPU Accelerators (NVIDIA T4 / P100).
  Tensors, batching, and Neural Network forward/backward passes are executed directly on CUDA.
====================================================================================================
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import yfinance as yf

warnings.filterwarnings('ignore')

# Automatic GPU Device Selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ==================================================================================================
# SECTION 1: SYNTHETIC MARKET WORLD GENERATOR (0% Real Historical Data Intake)
# ==================================================================================================
class RawPriceSyntheticEnv:
    """Procedurally generates virtual synthetic market worlds with multi-regime dynamics."""
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


# ==================================================================================================
# SECTION 2: GPU-OPTIMIZED NEURAL ARCHITECTURE
# ==================================================================================================
class DeepEndToEndTradingNet(nn.Module):
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64, nhead=2):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step

        self.conv1d = nn.Sequential(
            nn.Conv1d(features_per_step, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=128,
            dropout=0.05, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * history_len, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
        )
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step)
        x_trans = x.transpose(1, 2)
        conv_out = self.conv1d(x_trans).transpose(1, 2)
        trans_out = self.transformer(conv_out)
        flat = trans_out.reshape(b, -1)
        feat = self.fc(flat)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).to(DEVICE).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs).to(DEVICE)
            mean, _ = self.forward(flat_obs)
            if deterministic:
                return mean.cpu().numpy().squeeze(0)
            dist = Normal(mean, torch.exp(self.log_std))
            return dist.sample().cpu().numpy().squeeze(0)


# ==================================================================================================
# SECTION 3: PPO AGENT TRAINING ON CUDA/GPU
# ==================================================================================================
def train_rai_v6_gpu(total_steps=50_000, seed=42):
    print("=" * 100)
    print(f"  ⚡ TRAINING RAI v6 AGENT ON GPU / ACCELERATOR ({DEVICE})")
    print("=" * 100)
    t0 = time.time()

    env = RawPriceSyntheticEnv(num_assets=10, episode_len=504)
    model = DeepEndToEndTradingNet().to(DEVICE)
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

        if step % 10000 == 0:
            print(f"    Step {step:5d} / {total_steps} | Elapsed: {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"  ✅ GPU Training complete in {elapsed:.1f} seconds!\n")
    model.eval()
    return model


if __name__ == "__main__":
    trained_model = train_rai_v6_gpu(total_steps=50_000, seed=SEED)
