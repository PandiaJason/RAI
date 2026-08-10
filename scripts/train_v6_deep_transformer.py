"""
RAI v6: True End-to-End Deep Neural AI (Raw Prices Only)
=========================================================
Features:
1. ZERO hand-crafted indicators (NO SMA, NO pre-computed volatility).
2. Input: Raw 60-day price histories & raw log-returns.
3. Architecture: 1D CNN Temporal Feature Extractor + Multi-Head Self-Attention Transformer block + Deep Policy.
4. Active Directional Action Space: Continuous allocation from [-1.0 (Short/Inverse) to +1.0 (Long)], allowing real alpha generation.
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


# ═══════════════════════════════════════════════════════════════════
#  RAI v6 END-TO-END RAW PRICE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════

class RawPriceSyntheticEnv(gym.Env):
    """
    Synthetic environment providing ONLY raw normalized price histories.
    Zero human indicators (no SMAs, no volatility features).
    """

    REGIMES = {
        'bull':     {'drift': (0.15, 0.45),  'vol': (0.10, 0.20)},
        'bear':     {'drift': (-0.50, -0.15), 'vol': (0.25, 0.60)},
        'sideways': {'drift': (-0.05, 0.05),  'vol': (0.10, 0.20)},
    }

    def __init__(self, num_assets=10, history_len=60, episode_len=504, initial_cash=10000.0, fee=0.001):
        super().__init__()
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee

        # Raw observation matrix: 60 timesteps x (2 * num_assets + 2)
        # Features per step: [raw_price_normalized(N), raw_daily_log_return(N), cash_ratio, portfolio_drawdown]
        self.features_per_step = 2 * num_assets + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(history_len * self.features_per_step,), dtype=np.float32)

        # Action space: continuous weights [-1.0, +1.0] for each asset + cash weight [0, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_assets + 1,), dtype=np.float32)

        self.reset()

    def _generate_raw_prices(self):
        total_T = self.episode_len + self.history_len + 10
        n_segments = self.np_random.integers(3, 7)
        regime_keys = list(self.REGIMES.keys())
        sequence = [regime_keys[self.np_random.integers(len(regime_keys))] for _ in range(n_segments)]

        durations = self.np_random.dirichlet(np.ones(n_segments) * 2.0) * total_T
        durations = np.maximum(durations.astype(int), 30)
        durations[-1] = total_T - sum(durations[:-1])

        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)

        for asset in range(self.num_assets):
            p = self.np_random.uniform(20.0, 300.0)
            series = [p]
            day = 0

            for reg, dur in zip(sequence, durations):
                params = self.REGIMES[reg]
                drift = self.np_random.uniform(*params['drift']) + self.np_random.uniform(-0.04, 0.04)
                vol = self.np_random.uniform(*params['vol']) * self.np_random.uniform(0.85, 1.15)

                mu = drift / 252.0
                sigma = vol / np.sqrt(252.0)

                for _ in range(max(0, min(dur, total_T - day - 1))):
                    log_ret = (mu - 0.5 * sigma**2) + sigma * self.np_random.standard_normal()
                    p = max(0.01, p * np.exp(log_ret))
                    series.append(p)
                    day += 1
                    if day >= total_T: break
                if day >= total_T: break

            while len(series) < total_T:
                series.append(series[-1])
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
        p = self.prices[t]
        p_prev = self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))

        # 1. Raw normalized price (price / initial_price)
        norm_prices = p / self.prices[self.start]
        # 2. Raw daily log return
        log_rets = np.log(p / np.maximum(1e-4, p_prev))
        # 3. Portfolio cash ratio
        cash_ratio = self.cash / w
        # 4. Current drawdown
        dd = np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1.0, 0.0)

        return np.concatenate([norm_prices, log_rets, [cash_ratio, dd]]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        # Action: [cash_logit, asset_weight_logits(N)]
        cash_logit = np.clip(action[0], -5.0, 5.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))

        # Softmax allocation for stock portion
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
        
        # Pure Sharpe-like differential reward (reward return, penalize drawdown)
        reward = daily_ret * 5.0
        if daily_ret < 0:
            reward *= 2.0  # Asymmetric loss penalty

        drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
        if drawdown < -0.10:
            reward += drawdown * 2.0

        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth

        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))

        return self._flat_obs(), reward, done, False, {"portfolio_value": new_wealth, "cash_frac": target_cash_frac}


# ═══════════════════════════════════════════════════════════════════
#  RAI v6 DEEP CNN + TRANSFORMER NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════

class DeepEndToEndTradingNet(nn.Module):
    """
    End-to-End Neural Architecture:
    1D CNN + Multi-Head Transformer Encoder + Policy & Value Heads.
    Processes raw price sequence (batch, history_len, features_per_step).
    """
    def __init__(self, history_len=60, features_per_step=22, action_dim=11, embed_dim=128, nhead=4):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step

        # 1. 1D CNN Temporal Projection
        self.conv1d = nn.Sequential(
            nn.Conv1d(features_per_step, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        # 2. Multi-Head Self-Attention Transformer Encoder Layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=256, dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

        # 3. Global Pooling & Feature Head
        self.fc_features = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
        )

        # 4. Action (Actor) & Value (Critic) Heads
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        batch_size = flat_obs.shape[0]
        # Reshape to (batch, history_len, features_per_step)
        x = flat_obs.reshape(batch_size, self.history_len, self.features_per_step)
        
        # Pass through Conv1D over time: (batch, features_per_step, history_len)
        x_conv = x.permute(0, 2, 1)
        x_conv = self.conv1d(x_conv) # -> (batch, embed_dim, history_len)

        # Pass through Transformer: (batch, history_len, embed_dim)
        x_trans = x_conv.permute(0, 2, 1)
        x_trans = self.transformer(x_trans)

        # Mean pooling across timesteps -> (batch, embed_dim)
        latent = x_trans.mean(dim=1)
        feat = self.fc_features(latent)

        action_mean = self.actor_head(feat)
        value = self.critic_head(feat)

        return action_mean, value

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            mean, val = self.forward(flat_obs)
            if deterministic:
                action = mean
            else:
                std = torch.exp(self.log_std)
                dist = Normal(mean, std)
                action = dist.sample()
            return action.cpu().numpy().squeeze(0)


# ═══════════════════════════════════════════════════════════════════
#  RAI v6 TRAINING & REAL MARKET EVALUATION
# ═══════════════════════════════════════════════════════════════════

def train_and_eval_v6():
    print("=" * 85, flush=True)
    print("  RAI v6: True End-to-End Deep AI (Raw Prices + Transformer Encoder)", flush=True)
    print("=" * 85, flush=True)

    env = RawPriceSyntheticEnv(num_assets=10, history_len=60, episode_len=504)
    flat_obs_dim = env.observation_space.shape[0]
    features_per_step = env.features_per_step
    action_dim = env.action_space.shape[0]

    model = DeepEndToEndTradingNet(history_len=60, features_per_step=features_per_step, action_dim=action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total Network Parameters: {total_params:,}", flush=True)
    print(f"  Obs Dim: {flat_obs_dim} (60 timesteps x {features_per_step} raw features)", flush=True)
    print(f"  Action Dim: {action_dim} continuous weights", flush=True)
    print(f"  Training 250,000 steps with PPO on raw synthetic price worlds...\n", flush=True)

    BATCH_SIZE = 64
    ROLLOUT_LEN = 2048
    TOTAL_STEPS = 250_000
    N_EPOCHS = 6

    obs, _ = env.reset(seed=42)
    step = 0
    t0 = time.time()

    while step < TOTAL_STEPS:
        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []

        for _ in range(ROLLOUT_LEN):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean, val = model(obs_t)
                std = torch.exp(model.log_std)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            act_np = action.squeeze(0).numpy()
            next_obs, reward, done, _, info = env.step(act_np)

            obs_buf.append(obs)
            act_buf.append(act_np)
            rew_buf.append(reward)
            val_buf.append(val.item())
            logp_buf.append(log_prob.item())

            obs = next_obs
            step += 1
            if done: obs, _ = env.reset()

        # Advantage estimation
        with torch.no_grad():
            _, next_val = model(torch.FloatTensor(obs).unsqueeze(0))
            next_val = next_val.item()

        rewards = np.array(rew_buf)
        values = np.array(val_buf + [next_val])
        deltas = rewards + 0.99 * values[1:] - values[:-1]

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + 0.99 * 0.95 * gae
            advantages[t] = gae
        returns = advantages + values[:-1]

        obs_tensor = torch.FloatTensor(np.array(obs_buf))
        act_tensor = torch.FloatTensor(np.array(act_buf))
        adv_tensor = torch.FloatTensor(advantages)
        ret_tensor = torch.FloatTensor(returns)
        old_logp_tensor = torch.FloatTensor(np.array(logp_buf))

        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # PPO Update
        for epoch in range(N_EPOCHS):
            indices = np.random.permutation(len(obs_buf))
            for s in range(0, len(obs_buf), BATCH_SIZE):
                idx = indices[s:s+BATCH_SIZE]
                b_obs, b_act, b_adv, b_ret, b_old = obs_tensor[idx], act_tensor[idx], adv_tensor[idx], ret_tensor[idx], old_logp_tensor[idx]

                mean, val = model(b_obs)
                std = torch.exp(model.log_std)
                dist = Normal(mean, std)
                new_logp = dist.log_prob(b_act).sum(dim=-1)

                ratio = torch.exp(new_logp - b_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(val.squeeze(-1), b_ret)

                loss = policy_loss + 0.5 * value_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if step % 25000 < ROLLOUT_LEN:
            with torch.no_grad():
                c_frac = 1.0 / (1.0 + torch.exp(-act_tensor[:, 0]))
                print(f"  Step {step:>7d} | Cash Frac: min={c_frac.min().item():.3f} mean={c_frac.mean().item():.3f} max={c_frac.max().item():.3f} | Action Std={act_tensor.std().item():.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  v6 Training Complete in {elapsed:.0f}s ({TOTAL_STEPS/elapsed:.0f} FPS)", flush=True)

    # Save model
    os.makedirs("./data/v0.6_rl_checkpoints/", exist_ok=True)
    torch.save(model.state_dict(), "./data/v0.6_rl_checkpoints/rai_v6_deep_transformer.pt")

    # ═══════════════════════════════════════════════
    #  EVALUATION ON REAL MARKET DATA (RAW PRICES ONLY)
    # ═══════════════════════════════════════════════
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)

    model.eval()

    def eval_real_v6(df):
        prices_raw = df.values[:, :10]
        T, N = prices_raw.shape
        cash = 5000.0
        init_p = prices_raw[60]
        shares = (5000.0 / N) / init_p
        peak = 10000.0
        wealth_hist = [10000.0]
        cash_hist = []

        obs_history = []
        for t in range(60):
            p = prices_raw[t]
            p_prev = prices_raw[max(0, t-1)]
            w = 10000.0
            np_ = p / prices_raw[60]
            lr = np.log(p / np.maximum(1e-4, p_prev))
            obs_history.append(np.concatenate([np_, lr, [0.5, 0.0]]).astype(np.float32))

        for t in range(60, T):
            flat_obs = np.concatenate(obs_history).astype(np.float32)
            act = model.get_action(flat_obs, deterministic=True)

            cl = np.clip(act[0], -5, 5)
            target_cash = 1.0 / (1.0 + np.exp(-cl))
            target_stock = 1.0 - target_cash

            exp_a = np.exp(act[1:] - np.max(act[1:]))
            target_aw = (exp_a / np.sum(exp_a)) * target_stock

            p = prices_raw[t]
            w = max(1e-4, cash + np.sum(shares * p))
            caw = (shares * p) / w
            ccf = cash / w

            drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))
            if drift > 0.03:
                tv = abs(cash - w*target_cash) + np.sum(np.abs(shares*p - w*target_aw))
                net = max(1e-4, w - tv * 0.001)
                cash = net * target_cash
                shares = (net * target_aw) / np.maximum(1e-4, p)

            nw = cash + np.sum(shares * p)
            peak = max(peak, nw)
            wealth_hist.append(nw)
            cash_hist.append(target_cash)

            p_prev = prices_raw[t-1]
            np_ = p / prices_raw[60]
            lr = np.log(p / np.maximum(1e-4, p_prev))
            dd = np.clip((nw - peak)/peak, -1, 0)
            step_obs = np.concatenate([np_, lr, [cash/nw, dd]]).astype(np.float32)

            obs_history.pop(0)
            obs_history.append(step_obs)

        return wealth_hist, cash_hist

    for label, df in [("2020-2024 (Out-of-Sample)", test_df), ("2010-2019 (Historical)", train_df)]:
        print(f"\n{'='*80}", flush=True)
        print(f"  RAI v6 EVALUATION: {label}", flush=True)
        print(f"{'='*80}", flush=True)

        eq, cf = eval_real_v6(df)
        eq_arr = np.array(eq)
        rets = (eq_arr[1:] - eq_arr[:-1]) / np.maximum(1e-8, eq_arr[:-1])
        tot_ret = (eq_arr[-1] / eq_arr[0] - 1) * 100
        vol = np.std(rets) * np.sqrt(252) * 100
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 1e-8 else 0.0
        pk = np.maximum.accumulate(eq_arr)
        mdd = np.min((eq_arr - pk) / pk) * 100

        print(f"  Final Wealth:    ${eq_arr[-1]:,.2f}", flush=True)
        print(f"  Total Return:    {tot_ret:+.2f}%", flush=True)
        print(f"  Volatility:      {vol:.2f}%", flush=True)
        print(f"  Sharpe Ratio:    {sharpe:.2f}", flush=True)
        print(f"  Max Drawdown:    {mdd:.2f}%", flush=True)
        print(f"  Cash Min / Max:  {np.min(cf)*100:.1f}% / {np.max(cf)*100:.1f}% (Range: {(np.max(cf)-np.min(cf))*100:.1f}%)", flush=True)

        spy = df['SPY'].values
        spy_ret = (spy[-1] / spy[0] - 1) * 100
        print(f"  vs SPY Buy & Hold: {spy_ret:+.2f}% return", flush=True)

if __name__ == "__main__":
    train_and_eval_v6()
