"""
RAI v6 Synthetic World Environment
===================================
100% Raw Price Synthetic Market Generator (0% Real Historical Data)
"""

import numpy as np
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
