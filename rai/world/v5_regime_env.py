"""
RAI v5: Regime-Supervised Multi-Asset Environment
===================================================
Solves Greed vs Fear collapse by returning both:
1. Market observations (prices, SMA trends, volatility, momentum)
2. Ground-truth regime labels:
   - 0: BULL (positive drift, low/moderate vol)
   - 1: BEAR / CRASH (negative drift, high vol)
   - 2: SIDEWAYS (zero drift, moderate vol)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SyntheticRegimeSupervisedEnv(gym.Env):
    """
    Synthetic environment where each step belongs to an explicit regime.
    Ground-truth regime labels are provided for auxiliary supervision.
    """

    REGIMES = {
        0: {'name': 'BULL',     'drift': (0.15, 0.45),  'vol': (0.08, 0.18)},
        1: {'name': 'BEAR',     'drift': (-0.60, -0.20), 'vol': (0.30, 0.70)},
        2: {'name': 'SIDEWAYS', 'drift': (-0.05, 0.05),  'vol': (0.12, 0.22)},
    }

    def __init__(self, num_assets=10, episode_len=504, history_len=16,
                 initial_cash=10000.0, transaction_fee=0.001):
        super().__init__()
        self.num_assets = num_assets
        self.episode_len = episode_len
        self.history_len = history_len
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee

        # Single step obs: [cash_w(1), drawdown(1), ret5_avg(1), vol10_avg(1), asset_w(N), trend_SMA(N)]
        self.single_obs_dim = 4 + 2 * num_assets
        self.obs_dim = history_len * self.single_obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Action: [cash_logit(1), asset_logits(N)]
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(num_assets + 1,), dtype=np.float32)

        self.reset()

    def _generate_regime_prices_and_labels(self):
        """Generate price series with explicit regime segment labels."""
        total_T = self.episode_len + self.history_len + 30

        # Create regime sequence: Bull -> Bear -> Sideways -> Bull ...
        n_segments = self.np_random.integers(3, 7)
        # Alternate or randomly sample regimes
        segment_regimes = [self.np_random.integers(0, 3) for _ in range(n_segments)]

        # Ensure at least one Bull and one Bear in every episode
        if 0 not in segment_regimes:
            segment_regimes[0] = 0
        if 1 not in segment_regimes:
            segment_regimes[min(1, n_segments - 1)] = 1

        durations = self.np_random.dirichlet(np.ones(n_segments) * 2.0) * total_T
        durations = np.maximum(durations.astype(int), 40)
        durations[-1] = total_T - sum(durations[:-1])

        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)
        labels = np.zeros(total_T, dtype=np.int64)

        for asset in range(self.num_assets):
            p = self.np_random.uniform(20.0, 250.0)
            series = [p]
            day = 0

            for seg_idx, (regime_id, dur) in enumerate(zip(segment_regimes, durations)):
                params = self.REGIMES[regime_id]
                drift = self.np_random.uniform(*params['drift'])
                vol = self.np_random.uniform(*params['vol'])

                # Per-asset noise
                drift += self.np_random.uniform(-0.05, 0.05)
                vol *= self.np_random.uniform(0.85, 1.15)

                mu = drift / 252.0
                sigma = vol / np.sqrt(252.0)

                actual_dur = max(0, min(dur, total_T - day - 1))
                for _ in range(actual_dur):
                    log_ret = (mu - 0.5 * sigma**2) + sigma * self.np_random.standard_normal()
                    p = max(0.01, p * np.exp(log_ret))
                    series.append(p)
                    if asset == 0:
                        labels[day] = regime_id
                    day += 1
                    if day >= total_T:
                        break
                if day >= total_T:
                    break

            while len(series) < total_T:
                series.append(series[-1])
            prices[:, asset] = series[:total_T]

        return prices, labels

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prices, self.regime_labels = self._generate_regime_prices_and_labels()
        self.start = self.history_len + 20
        self.current_step = self.start

        self.cash = self.initial_cash * 0.5
        init_p = self.prices[self.current_step]
        self.shares = (self.initial_cash * 0.5 / self.num_assets) / init_p

        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0

        self.obs_history = [self._obs_at(self.start - self.history_len + i)
                            for i in range(self.history_len)]
        return self._flat_obs(), {"regime_label": self.regime_labels[self.current_step]}

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p = self.prices[t]
        w = max(1e-4, self.cash + np.sum(self.shares * p))

        cash_w = self.cash / w
        dd = np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1.0, 0.0)

        ret5 = np.mean((p - self.prices[max(0, t-5)]) / np.maximum(1e-4, self.prices[max(0, t-5)])) if t >= 5 else 0.0

        if t >= 10:
            sub = self.prices[t-10:t+1]
            r = (sub[1:] - sub[:-1]) / np.maximum(1e-4, sub[:-1])
            avg_vol = np.mean(np.std(r, axis=0))
        else:
            avg_vol = 0.0

        asset_w = (self.shares * p) / w

        if t >= 50:
            s20 = np.mean(self.prices[t-20:t], axis=0)
            s50 = np.mean(self.prices[t-50:t], axis=0)
            trend = s20 / np.maximum(1e-4, s50) - 1.0
        else:
            trend = np.zeros(self.num_assets, dtype=np.float32)

        return np.concatenate([
            [cash_w, dd, ret5, avg_vol],
            asset_w,
            trend,
        ]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        # Sigmoid cash allocation
        cash_logit = np.clip(action[0], -10, 10)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))

        # Relative stock weights
        exp_a = np.exp(action[1:] - np.max(action[1:]))
        rel_w = exp_a / np.sum(exp_a)
        target_asset_w = rel_w * (1.0 - target_cash_frac)

        prices = self.prices[self.current_step]
        wealth = max(1e-4, self._wealth())
        cur_asset_w = (self.shares * prices) / wealth
        cur_cash_frac = self.cash / wealth

        drift = abs(cur_cash_frac - target_cash_frac) + np.sum(np.abs(cur_asset_w - target_asset_w))

        if drift > 0.03:
            t_vol = abs(self.cash - wealth * target_cash_frac) + \
                    np.sum(np.abs(self.shares * prices - wealth * target_asset_w))
            net = max(1e-4, wealth - t_vol * self.transaction_fee)
            self.cash = net * target_cash_frac
            self.shares = (net * target_asset_w) / np.maximum(1e-4, prices)

        self.current_step += 1
        self.steps_done += 1
        new_wealth = self._wealth()
        self.peak_wealth = max(self.peak_wealth, new_wealth)

        daily_ret = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)

        # Regime-aware reward:
        current_regime = self.regime_labels[self.current_step]
        
        if current_regime == 0:     # BULL: reward stock gains, penalize high cash
            reward = daily_ret * 2.0 - 0.1 * target_cash_frac
        elif current_regime == 1:   # BEAR: reward cash preservation, heavy penalty for drawdowns
            reward = (1.0 - target_cash_frac) * daily_ret * 4.0 - (1.0 - target_cash_frac) * 0.5
        else:                       # SIDEWAYS: reward low volatility / low turnover
            reward = daily_ret - 0.05 * drift

        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth

        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))

        info = {
            "portfolio_value": new_wealth,
            "cash_frac": target_cash_frac,
            "regime_label": current_regime,
            "drawdown": drawdown,
        }

        return self._flat_obs(), reward, done, False, info
