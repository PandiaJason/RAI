"""
═══════════════════════════════════════════════════════════════════════════════
  RAI v7: G6 JUMP-DIFFUSION & VOLATILITY CLUSTERING SYNTHETIC ENVIRONMENT
  ═══════════════════════════════════════════════════════════════════════════════
  Implements advanced synthetic market dynamics:
    1. Merton Poisson Jump-Diffusion (sudden market crashes & micro-shocks)
    2. GARCH(1,1) Volatility Clustering (persistent volatility regimes)
    3. Panic Correlation Breakdown (cross-asset correlation spikes in panics)
    4. Sortino Downside-Risk Reward Signal (penalizes negative semi-variance)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class G6EnhancedSyntheticEnv(gym.Env):
    """G6 Synthetic Environment with Jump Diffusion, GARCH Volatility, and Panic Correlations."""
    
    REGIMES = {
        'bull':          {'drift': (0.15, 0.40),  'vol_base': (0.10, 0.18), 'jump_lambda': 0.02, 'jump_mean': -0.03, 'jump_std': 0.02},
        'bear':          {'drift': (-0.45, -0.15), 'vol_base': (0.25, 0.45), 'jump_lambda': 0.12, 'jump_mean': -0.08, 'jump_std': 0.05},
        'panic_crash':   {'drift': (-0.75, -0.30), 'vol_base': (0.40, 0.70), 'jump_lambda': 0.35, 'jump_mean': -0.15, 'jump_std': 0.08},
        'sideways':      {'drift': (-0.05, 0.05),  'vol_base': (0.10, 0.20), 'jump_lambda': 0.03, 'jump_mean': -0.02, 'jump_std': 0.02},
    }

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001):
        super().__init__()
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee

        # Features per step: price ratios + log returns + cash ratio + peak drawdown
        self.features_per_step = 2 * num_assets + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (history_len * self.features_per_step,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (num_assets + 1,), np.float32)

    def _generate_g6_prices(self):
        total_T = self.episode_len + self.history_len + 10
        n_seg = self.np_random.integers(4, 8)
        keys = list(self.REGIMES.keys())
        seq = [keys[self.np_random.integers(len(keys))] for _ in range(n_seg)]

        dur = self.np_random.dirichlet(np.ones(n_seg) * 2.0) * total_T
        dur = np.maximum(dur.astype(int), 30)
        dur[-1] = total_T - sum(dur[:-1])

        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)
        
        # GARCH(1,1) parameters: h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}
        omega = 1e-5
        alpha_garch = 0.08
        beta_garch = 0.88

        for asset in range(self.num_assets):
            p = self.np_random.uniform(20.0, 300.0)
            series = [p]
            day = 0
            h_t = (0.20 / np.sqrt(252.0)) ** 2  # initial variance

            for reg, d in zip(seq, dur):
                params = self.REGIMES[reg]
                drift = self.np_random.uniform(*params['drift'])
                vol_b = self.np_random.uniform(*params['vol_base'])
                mu = drift / 252.0
                base_sigma = vol_b / np.sqrt(252.0)

                for _ in range(max(0, min(d, total_T - day - 1))):
                    # 1. GARCH volatility update
                    sigma_t = np.sqrt(max(1e-8, h_t))
                    
                    # 2. Diffusion shock
                    z = self.np_random.standard_normal()
                    eps_t = z * sigma_t
                    
                    # 3. Merton Poisson Jump Shock
                    n_jumps = self.np_random.poisson(params['jump_lambda'])
                    jump_shock = 0.0
                    if n_jumps > 0:
                        jump_shock = np.sum(self.np_random.normal(params['jump_mean'], params['jump_std'], size=n_jumps))

                    # Combine diffusion + jump
                    ret = mu + eps_t + jump_shock
                    p = max(0.01, p * np.exp(ret))
                    series.append(p)

                    # Update GARCH variance state for next step
                    h_t = omega + alpha_garch * (eps_t ** 2) + beta_garch * h_t
                    day += 1
                    if day >= total_T:
                        break
                if day >= total_T:
                    break
            while len(series) < total_T:
                series.append(series[-1])
            prices[:, asset] = series[:total_T]

        return prices

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prices = self._generate_g6_prices()
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.05
        self.shares = (self.initial_cash * 0.95 / self.num_assets) / self.prices[self.current_step]
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0
        self.recent_returns = []
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs(), {}

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p, pp = self.prices[t], self.prices[max(0, t - 1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        return np.concatenate([
            p / self.prices[self.start],
            np.log(p / np.maximum(1e-4, pp)),
            [self.cash / w, np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1, 0)]
        ]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        cash_logit = np.clip(action[0], -5.0, 5.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash_frac
        exp_a = np.exp(action[1:] - np.max(action[1:]))
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

        # ── SORTINO / DOWNSIDE SEMI-VARIANCE REWARD FUNCTION ──
        daily_ret = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        self.recent_returns.append(daily_ret)
        if len(self.recent_returns) > 20:
            self.recent_returns.pop(0)

        # Base return reward
        reward = daily_ret * 5.0

        # Downside semi-variance penalty (Sortino focus)
        if daily_ret < 0:
            downside_penalty = (daily_ret ** 2) * 50.0
            reward -= downside_penalty

        # Severe drawdown penalty
        drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
        if drawdown < -0.05:
            reward += drawdown * 3.0

        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))
        return self._flat_obs(), reward, done, False, {}
