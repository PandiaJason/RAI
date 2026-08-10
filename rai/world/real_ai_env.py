"""
RAI v4.1: REAL AI — Fixes the fundamental design flaws.

Root cause of v3/v4 failure: Softmax couples cash and stock allocation.
Moving cash from 6% → 50% requires enormous logit shifts that the network
can't learn in reasonable training time.

FIXES:
1. SIGMOID cash allocation — independent control, easy to swing 0-100%
2. EPISODE DEATH on 25% drawdown — forces risk management or die
3. 50% of episodes contain severe crashes — constant allocation = death
4. Clean observation with explicit danger signals
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RealAIEnv(gym.Env):
    """
    Training environment that FORCES the agent to learn regime switching.
    If it doesn't adapt, episodes terminate early with huge penalty.
    """

    REGIMES = {
        'strong_bull': {'drift': (0.30, 0.60), 'vol': (0.08, 0.15)},
        'mild_bull':   {'drift': (0.08, 0.20), 'vol': (0.12, 0.22)},
        'sideways':    {'drift': (-0.05, 0.05), 'vol': (0.15, 0.25)},
        'bear':        {'drift': (-0.30, -0.10), 'vol': (0.25, 0.40)},
        'crash':       {'drift': (-0.80, -0.40), 'vol': (0.50, 1.00)},
    }

    def __init__(self, num_assets=10, episode_len=504, history_len=16,
                 initial_cash=10000.0, transaction_fee=0.001,
                 max_drawdown=-0.25, death_penalty=-10.0):
        super().__init__()
        self.num_assets = num_assets
        self.episode_len = episode_len
        self.history_len = history_len
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.max_drawdown = max_drawdown
        self.death_penalty = death_penalty

        # Observation per step:
        # [cash_weight, drawdown, avg_5d_return, avg_volatility,
        #  per_asset_weight(N), per_asset_trend(N)]
        # = 4 + 2*N
        self.single_obs_dim = 4 + 2 * num_assets
        self.obs_dim = history_len * self.single_obs_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Action: [cash_logit(1), asset_logits(N)]
        # cash_fraction = sigmoid(cash_logit) → [0, 1]
        # asset_weights = softmax(asset_logits) → relative within stock portion
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(num_assets + 1,), dtype=np.float32)

        self.reset()

    def _generate_prices(self):
        """Generate multi-regime episode. 50% include a crash."""
        total_T = self.episode_len + self.history_len + 25
        n_regimes = self.np_random.integers(2, 5)

        # Sample regimes
        regime_names = list(self.REGIMES.keys())
        sequence = [regime_names[self.np_random.integers(len(regime_names))]
                    for _ in range(n_regimes)]

        # 50% chance of a crash somewhere in the episode
        if self.np_random.random() < 0.5:
            idx = self.np_random.integers(n_regimes)
            sequence[idx] = 'crash'

        # Ensure at least one non-crash to learn from
        if all(r == 'crash' for r in sequence):
            sequence[0] = 'mild_bull'

        durations = self.np_random.dirichlet(np.ones(n_regimes) * 2.0) * total_T
        durations = np.maximum(durations.astype(int), 30)
        durations[-1] = total_T - sum(durations[:-1])

        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)

        for asset in range(self.num_assets):
            p = self.np_random.uniform(30.0, 300.0)
            series = [p]
            day = 0

            for regime, dur in zip(sequence, durations):
                params = self.REGIMES[regime]
                drift = self.np_random.uniform(*params['drift'])
                vol = self.np_random.uniform(*params['vol'])
                # Per-asset variation
                drift += self.np_random.uniform(-0.08, 0.08)
                vol *= self.np_random.uniform(0.7, 1.3)

                mu = drift / 252.0
                sigma = vol / np.sqrt(252.0)

                for _ in range(max(0, min(dur, total_T - day - 1))):
                    log_ret = (mu - 0.5*sigma**2) + sigma * self.np_random.standard_normal()
                    p = p * np.exp(log_ret)
                    p = max(0.01, p)  # Floor
                    series.append(p)
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
        self.prices = self._generate_prices()
        T, N = self.prices.shape
        self.start = self.history_len + 20
        self.current_step = self.start

        # Start 50/50 cash/stocks
        self.cash = self.initial_cash * 0.5
        init_prices = self.prices[self.current_step]
        per_asset = (self.initial_cash * 0.5) / N
        self.shares = per_asset / init_prices

        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0

        # Build history
        self.obs_history = [self._obs_at(self.start) for _ in range(self.history_len)]
        return self._flat_obs(), {}

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p = self.prices[t]
        w = max(1e-4, self.cash + np.sum(self.shares * p))

        cash_w = self.cash / w
        dd = (w - self.peak_wealth) / max(1e-4, self.peak_wealth)
        dd = np.clip(dd, -1.0, 0.0)

        # Average 5-day return across assets
        if t >= 5:
            ret5 = np.mean((p - self.prices[t-5]) / np.maximum(1e-4, self.prices[t-5]))
        else:
            ret5 = 0.0

        # Average 10-day volatility
        if t >= 10:
            sub = self.prices[t-10:t+1]
            r = (sub[1:] - sub[:-1]) / np.maximum(1e-4, sub[:-1])
            avg_vol = np.mean(np.std(r, axis=0))
        else:
            avg_vol = 0.0

        # Per-asset weights
        asset_w = np.zeros(self.num_assets, dtype=np.float32)
        asset_w[:] = (self.shares * p) / w

        # Per-asset trend (SMA20/SMA50)
        if t >= 50:
            sma20 = np.mean(self.prices[t-20:t], axis=0)
            sma50 = np.mean(self.prices[t-50:t], axis=0)
            trend = sma20 / np.maximum(1e-4, sma50) - 1.0  # Centered at 0
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
        # ═══ SIGMOID CASH ALLOCATION ═══
        # cash_fraction = sigmoid(action[0]) → [0, 1]
        cash_logit = np.clip(action[0], -10, 10)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))

        # Asset weights within stock portion = softmax(action[1:])
        asset_logits = action[1:]
        exp_a = np.exp(asset_logits - np.max(asset_logits))
        relative_weights = exp_a / np.sum(exp_a)

        # Target allocation
        target_stock_frac = 1.0 - target_cash_frac
        target_asset_w = relative_weights * target_stock_frac

        # Current state
        prices = self.prices[self.current_step]
        wealth = max(1e-4, self._wealth())
        current_asset_w = (self.shares * prices) / wealth
        current_cash_frac = self.cash / wealth

        # Rebalance if allocation drifts > 3%
        total_drift = abs(current_cash_frac - target_cash_frac) + \
                      np.sum(np.abs(current_asset_w - target_asset_w))

        if total_drift > 0.03:
            trade_vol = abs(self.cash - wealth * target_cash_frac) + \
                       np.sum(np.abs(self.shares * prices - wealth * target_asset_w))
            fee = trade_vol * self.transaction_fee
            net = max(1e-4, wealth - fee)
            self.cash = net * target_cash_frac
            self.shares = (net * target_asset_w) / np.maximum(1e-4, prices)

        # Advance
        self.current_step += 1
        self.steps_done += 1
        new_wealth = self._wealth()
        self.peak_wealth = max(self.peak_wealth, new_wealth)

        # ═══ REWARD ═══
        daily_ret = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)

        # Asymmetric: 3x penalty for losses
        if daily_ret >= 0:
            reward = daily_ret
        else:
            reward = daily_ret * 3.0

        # Quadratic drawdown penalty
        if drawdown < -0.05:
            reward -= 2.0 * (drawdown + 0.05) ** 2

        # ═══ DEATH: Episode ends on severe drawdown ═══
        done = False
        if drawdown < self.max_drawdown:
            done = True
            reward = self.death_penalty

        if self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len:
            done = True

        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))

        return self._flat_obs(), reward, done, False, {
            "portfolio_value": new_wealth,
            "cash_frac": target_cash_frac,
            "drawdown": drawdown,
        }
