"""
RAI v6 Fast: End-to-End Deep Neural AI on Raw Prices
=====================================================
Features:
1. ZERO hand-crafted indicators (NO SMAs, NO pre-computed volatility).
2. Input: 30 timesteps of raw price ratios & daily log-returns.
3. Architecture: 1D Conv1D + Transformer Encoder + Deep Actor/Critic heads.
4. Optimized for fast execution on CPU (~60-90s).
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces


class RawPriceSyntheticEnv(gym.Env):
    """Synthetic environment with 100% raw prices & zero human indicators."""
    REGIMES = {
        'bull':     {'drift': (0.15, 0.45),  'vol': (0.10, 0.20)},
        'bear':     {'drift': (-0.50, -0.15), 'vol': (0.25, 0.60)},
        'sideways': {'drift': (-0.05, 0.05),  'vol': (0.10, 0.20)},
    }

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001):
        super().__init__()
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee
        self.features_per_step = 2 * num_assets + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(history_len * self.features_per_step,), dtype=np.float32)
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(num_assets + 1,), dtype=np.float32)
        self.reset()

    def _generate_raw_prices(self):
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
                mu = drift / 252.0; sigma = vol / np.sqrt(252.0)
                for _ in range(max(0, min(d, total_T - day - 1))):
                    p = max(0.01, p * np.exp((mu - 0.5*sigma**2) + sigma * self.np_random.standard_normal()))
                    series.append(p)
                    day += 1
                    if day >= total_T: break
                if day >= total_T: break
            while len(series) < total_T: series.append(series[-1])
            prices[:, asset] = series[:total_T]

        return prices

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
        return self._flat_obs(), {}

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

        return self._flat_obs(), reward, done, False, {"portfolio_value": new_wealth, "cash_frac": target_cash_frac}


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

        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=128, dropout=0.05, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)

        self.fc_features = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LeakyReLU(0.1),
        )

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
            return mean.cpu().numpy().squeeze(0) if deterministic else Normal(mean, torch.exp(self.log_std)).sample().cpu().numpy().squeeze(0)


def train_and_eval():
    print("=" * 80, flush=True)
    print("  RAI v6: End-to-End Deep AI on Raw Prices (100% No Indicators)", flush=True)
    print("=" * 80, flush=True)

    env = RawPriceSyntheticEnv(num_assets=10, history_len=30, episode_len=504)
    model = DeepEndToEndTradingNet(history_len=30, features_per_step=22, action_dim=11)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"  Obs Dim: {30*22} (30 timesteps x 22 raw price/return features)", flush=True)
    print(f"  Training 100,000 steps with PPO...\n", flush=True)

    BATCH = 64; ROLLOUT = 1024; STEPS = 100_000; EPOCHS = 4
    obs, _ = env.reset(seed=42)
    step = 0; t0 = time.time()

    while step < STEPS:
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
            obs = nobs; step += 1
            if done: obs, _ = env.reset()

        with torch.no_grad():
            _, nval = model(torch.FloatTensor(obs).unsqueeze(0))
            nval = nval.item()

        r = np.array(rew_b); v = np.array(val_b + [nval])
        d = r + 0.99 * v[1:] - v[:-1]
        adv = np.zeros_like(r); gae = 0.0
        for t in reversed(range(len(r))):
            gae = d[t] + 0.99 * 0.95 * gae
            adv[t] = gae
        ret = adv + v[:-1]

        o_t, a_t, adv_t, ret_t, old_t = torch.FloatTensor(np.array(obs_b)), torch.FloatTensor(np.array(act_b)), torch.FloatTensor(adv), torch.FloatTensor(ret), torch.FloatTensor(np.array(logp_b))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(EPOCHS):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), BATCH):
                b_idx = idx[s:s+BATCH]
                mean, val = model(o_t[b_idx])
                dist = Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_t[b_idx])
                surr1 = ratio * adv_t[b_idx]; surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b_idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(val.squeeze(-1), ret_t[b_idx])
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step()

        if step % 20000 < ROLLOUT:
            c_frac = 1.0 / (1.0 + torch.exp(-a_t[:, 0]))
            print(f"  Step {step:>6d} | Cash Frac: min={c_frac.min().item():.3f} mean={c_frac.mean().item():.3f} max={c_frac.max().item():.3f}", flush=True)

    el = time.time() - t0
    print(f"\n  Trained 100k steps in {el:.0f}s ({STEPS/el:.0f} FPS)", flush=True)

    os.makedirs("./data/v0.6_rl_checkpoints/", exist_ok=True)
    torch.save(model.state_dict(), "./data/v0.6_rl_checkpoints/rai_v6_fast.pt")

    # ═══════════════════════════════════════════════
    #  EVALUATION ON REAL MARKET (RAW PRICES ONLY)
    # ═══════════════════════════════════════════════
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)

    model.eval()

    def eval_real(df):
        p_raw = df.values[:, :10]; T, N = p_raw.shape
        cash = 5000.0; init_p = p_raw[30]; shares = (5000.0 / N) / init_p
        peak = 10000.0; eq = [10000.0]; cf = []

        obs_h = []
        for t in range(30):
            p = p_raw[t]; p_prev = p_raw[max(0, t-1)]
            obs_h.append(np.concatenate([p / p_raw[30], np.log(p / np.maximum(1e-4, p_prev)), [0.5, 0.0]]).astype(np.float32))

        for t in range(30, T):
            flat_obs = np.concatenate(obs_h).astype(np.float32)
            act = model.get_action(flat_obs, deterministic=True)
            cl = np.clip(act[0], -5, 5)
            target_cash = 1.0 / (1.0 + np.exp(-cl))
            target_stock = 1.0 - target_cash
            ea = np.exp(act[1:] - np.max(act[1:])); target_aw = (ea / np.sum(ea)) * target_stock

            p = p_raw[t]; w = max(1e-4, cash + np.sum(shares * p))
            caw = (shares * p) / w; ccf = cash / w
            drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))
            if drift > 0.03:
                tv = abs(cash - w*target_cash) + np.sum(np.abs(shares*p - w*target_aw))
                net = max(1e-4, w - tv * 0.001)
                cash = net * target_cash; shares = (net * target_aw) / np.maximum(1e-4, p)

            nw = cash + np.sum(shares * p)
            peak = max(peak, nw)
            eq.append(nw); cf.append(target_cash)

            p_prev = p_raw[t-1]
            obs_h.pop(0)
            obs_h.append(np.concatenate([p / p_raw[30], np.log(p / np.maximum(1e-4, p_prev)), [cash/nw, np.clip((nw-peak)/peak, -1, 0)]]).astype(np.float32))

        return eq, cf

    for label, df in [("2020-2024 (Out-of-Sample)", test_df), ("2010-2019 (Historical)", train_df)]:
        print(f"\n{'='*80}", flush=True)
        print(f"  RAI v6 EVALUATION: {label}", flush=True)
        print(f"{'='*80}", flush=True)

        eq, cf = eval_real(df)
        eq_arr = np.array(eq)
        r = (eq_arr[1:] - eq_arr[:-1]) / np.maximum(1e-8, eq_arr[:-1])
        tot_ret = (eq_arr[-1] / eq_arr[0] - 1) * 100
        vol = np.std(r) * np.sqrt(252) * 100
        sh = np.mean(r) / np.std(r) * np.sqrt(252) if np.std(r) > 1e-8 else 0
        pk = np.maximum.accumulate(eq_arr)
        mdd = np.min((eq_arr - pk) / pk) * 100
        cf_arr = np.array(cf)

        print(f"  Final Wealth:     ${eq_arr[-1]:,.2f}", flush=True)
        print(f"  Total Return:     {tot_ret:+.2f}%", flush=True)
        print(f"  Volatility:       {vol:.2f}%", flush=True)
        print(f"  Sharpe Ratio:     {sh:.2f}", flush=True)
        print(f"  Max Drawdown:     {mdd:.2f}%", flush=True)
        print(f"  Cash Min / Max:   {np.min(cf_arr)*100:.1f}% / {np.max(cf_arr)*100:.1f}% (Range: {(np.max(cf_arr)-np.min(cf_arr))*100:.1f}%)", flush=True)

        spy = df['SPY'].values
        spy_ret = (spy[-1] / spy[0] - 1) * 100
        print(f"  vs SPY Buy & Hold: {spy_ret:+.2f}% return", flush=True)

if __name__ == "__main__":
    train_and_eval()
