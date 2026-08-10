"""
RAI v4: Zero-Shot Regime-Switching Environment
================================================
Fixes v3's root cause: policy collapsed to constant output because
the reward didn't punish static behavior and episodes had no regime switches.

Key changes from v3:
1. REGIME SWITCHES within each episode (2-4 transitions)
2. ASYMMETRIC REWARD: 3x penalty for losses + drawdown punishment
3. Guaranteed crash episodes so constant allocation FAILS
4. Drawdown-from-peak as observation feature
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SyntheticRegimeSwitchEnv(gym.Env):
    """
    Each episode contains 2-4 regime switches (bull→crash→recovery etc).
    The agent MUST change its allocation to avoid drawdown penalties.
    """
    
    REGIMES = {
        'bull':     {'drift': (0.15, 0.40), 'vol': (0.08, 0.18)},
        'bear':     {'drift': (-0.35, -0.10), 'vol': (0.20, 0.40)},
        'crash':    {'drift': (-0.70, -0.40), 'vol': (0.40, 0.80)},
        'sideways': {'drift': (-0.05, 0.05), 'vol': (0.10, 0.20)},
        'recovery': {'drift': (0.30, 0.60), 'vol': (0.15, 0.30)},
    }
    
    def __init__(self, num_assets=10, episode_len=504, history_len=16,
                 initial_cash=10000.0, transaction_fee=0.001,
                 rebalance_threshold=0.03, loss_penalty_mult=3.0,
                 drawdown_threshold=0.05, drawdown_penalty=1.0):
        super().__init__()
        self.num_assets = num_assets
        self.episode_len = episode_len
        self.history_len = history_len
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.rebalance_threshold = rebalance_threshold
        self.loss_penalty_mult = loss_penalty_mult
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_penalty = drawdown_penalty
        
        # Observation: per timestep = cash_w(1) + asset_w(N) + returns_5d(N) + trend(N) + vol(N) + drawdown(1)
        self.single_obs_dim = 1 + num_assets * 4 + 1
        self.obs_dim = history_len * self.single_obs_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        # Action: cash_weight + N asset weights (softmaxed)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_assets + 1,), dtype=np.float32
        )
        
        self.reset()
    
    def _generate_regime_switching_prices(self):
        """Generate price series with 2-4 regime transitions."""
        total_T = self.episode_len + self.history_len + 25  # +25 for 5d returns warmup
        
        # Number of regime switches
        n_regimes = self.np_random.integers(2, 5)
        
        # Sample regime types - guarantee at least one crash/bear
        regime_names = list(self.REGIMES.keys())
        regime_sequence = [regime_names[self.np_random.integers(len(regime_names))] 
                          for _ in range(n_regimes)]
        
        # Force at least one crash or bear
        has_negative = any(r in ['crash', 'bear'] for r in regime_sequence)
        if not has_negative:
            idx = self.np_random.integers(n_regimes)
            regime_sequence[idx] = self.np_random.choice(['crash', 'bear'])
        
        # Generate regime durations
        raw_durations = self.np_random.dirichlet(np.ones(n_regimes) * 2.0)
        regime_durations = (raw_durations * total_T).astype(int)
        regime_durations[-1] = total_T - np.sum(regime_durations[:-1])
        regime_durations = np.maximum(regime_durations, 20)  # Min 20 days per regime
        
        # Generate prices per asset
        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)
        
        for asset_i in range(self.num_assets):
            initial_price = self.np_random.uniform(20.0, 300.0)
            log_price = np.log(initial_price)
            price_series = [initial_price]
            
            day = 0
            for reg_idx, (regime_name, duration) in enumerate(zip(regime_sequence, regime_durations)):
                params = self.REGIMES[regime_name]
                drift_annual = self.np_random.uniform(*params['drift'])
                vol_annual = self.np_random.uniform(*params['vol'])
                
                # Add per-asset variation
                drift_annual += self.np_random.uniform(-0.05, 0.05)
                vol_annual *= self.np_random.uniform(0.8, 1.2)
                
                mu_daily = drift_annual / 252.0
                sigma_daily = vol_annual / np.sqrt(252.0)
                
                actual_duration = min(duration, total_T - day - 1)
                if actual_duration <= 0:
                    break
                
                for _ in range(actual_duration):
                    log_ret = (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * self.np_random.standard_normal()
                    log_price += log_ret
                    price_series.append(np.exp(log_price))
                    day += 1
                    if day >= total_T:
                        break
                
                if day >= total_T:
                    break
            
            # Pad if needed
            while len(price_series) < total_T:
                price_series.append(price_series[-1])
            
            prices[:, asset_i] = price_series[:total_T]
        
        return prices
    
    def _precompute_indicators(self, prices):
        """Precompute SMA and volatility indicators."""
        T, N = prices.shape
        self.sma20 = np.zeros_like(prices)
        self.sma50 = np.zeros_like(prices)
        self.vol10 = np.zeros_like(prices)
        self.returns_5d = np.zeros_like(prices)
        
        for t in range(T):
            s20 = max(0, t - 20)
            s50 = max(0, t - 50)
            self.sma20[t] = np.mean(prices[s20:t+1], axis=0)
            self.sma50[t] = np.mean(prices[s50:t+1], axis=0)
            
            if t >= 5:
                self.returns_5d[t] = (prices[t] - prices[t-5]) / np.maximum(1e-4, prices[t-5])
            
            if t >= 10:
                sub_p = prices[t-10:t+1]
                r = (sub_p[1:] - sub_p[:-1]) / np.maximum(1e-4, sub_p[:-1])
                self.vol10[t] = np.std(r, axis=0)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate regime-switching prices
        self.prices_matrix = self._generate_regime_switching_prices()
        self._precompute_indicators(self.prices_matrix)
        
        total_T = self.prices_matrix.shape[0]
        self.start_step = self.history_len + 20  # warmup for indicators
        self.current_step = self.start_step
        
        # Initialize portfolio: 50% cash, 50% equal-weight stocks
        self.cash = self.initial_cash * 0.50
        prices = self.prices_matrix[self.current_step]
        per_asset = (self.initial_cash * 0.50) / self.num_assets
        self.shares = per_asset / prices
        
        # Peak tracking for drawdown
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        
        # Build initial observation history
        self.obs_history = []
        for i in range(self.history_len):
            step_idx = self.start_step - self.history_len + i
            self.obs_history.append(self._get_single_obs_at(step_idx))
        
        self.rebalance_count = 0
        self.total_steps_in_episode = 0
        
        return self._get_obs(), {}
    
    def _get_portfolio_value(self):
        return self.cash + np.sum(self.shares * self.prices_matrix[self.current_step])
    
    def _get_single_obs_at(self, step_idx):
        """Get observation at a specific step (for history building)."""
        prices = self.prices_matrix[step_idx]
        wealth = max(1e-4, self.cash + np.sum(self.shares * prices))
        
        w_cash = self.cash / wealth
        w_assets = (self.shares * prices) / wealth
        
        # 5-day returns per asset (captures recent momentum)
        ret_5d = self.returns_5d[step_idx]
        
        # Trend: SMA20/SMA50 ratio (faster signal than SMA50/200)
        trend = self.sma20[step_idx] / np.maximum(1e-4, self.sma50[step_idx])
        
        # 10-day volatility
        vol = self.vol10[step_idx]
        
        # Drawdown from peak
        dd = (wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
        dd = np.clip(dd, -1.0, 0.0)
        
        obs = np.concatenate([
            [w_cash],           # 1
            w_assets,           # N
            ret_5d,             # N  (recent momentum)
            trend,              # N  (trend direction)
            vol,                # N  (current volatility)
            [dd],               # 1  (drawdown from peak)
        ]).astype(np.float32)
        
        return obs
    
    def _get_single_obs(self):
        return self._get_single_obs_at(self.current_step)
    
    def _get_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)
    
    def step(self, action):
        # Convert action to target weights via softmax
        exp_act = np.exp(action - np.max(action))
        target_weights = exp_act / np.sum(exp_act)
        
        target_cash_w = target_weights[0]
        target_asset_w = target_weights[1:]
        
        # Normalize
        total_w = target_cash_w + np.sum(target_asset_w)
        target_cash_w /= total_w
        target_asset_w /= total_w
        
        prices = self.prices_matrix[self.current_step]
        current_wealth = max(1e-4, self._get_portfolio_value())
        current_asset_w = (self.shares * prices) / current_wealth
        
        # Rebalance only if drift exceeds threshold
        weight_drift = np.sum(np.abs(current_asset_w - target_asset_w))
        if weight_drift > self.rebalance_threshold:
            self.rebalance_count += 1
            target_vals = current_wealth * target_asset_w
            rebalance_vol = np.sum(np.abs((self.shares * prices) - target_vals))
            fee = rebalance_vol * self.transaction_fee
            net_wealth = max(1e-4, current_wealth - fee)
            self.cash = net_wealth * target_cash_w
            self.shares = (net_wealth * target_asset_w) / np.maximum(1e-4, prices)
        
        # Advance
        self.current_step += 1
        self.total_steps_in_episode += 1
        
        max_step = self.prices_matrix.shape[0] - 1
        done = (self.current_step >= max_step) or (self.total_steps_in_episode >= self.episode_len)
        
        new_wealth = self._get_portfolio_value()
        
        # ═══════════════════════════════════════════
        #  ASYMMETRIC REWARD (v4 key innovation)
        # ═══════════════════════════════════════════
        daily_return = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        
        if daily_return >= 0:
            reward = daily_return  # Normal reward for gains
        else:
            reward = daily_return * self.loss_penalty_mult  # 3x penalty for losses
        
        # Drawdown penalty
        self.peak_wealth = max(self.peak_wealth, new_wealth)
        current_drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
        
        if current_drawdown < -self.drawdown_threshold:
            reward -= self.drawdown_penalty * abs(current_drawdown)
        
        # Update state
        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._get_single_obs())
        
        return self._get_obs(), reward, done, False, {
            "portfolio_value": new_wealth,
            "rebalances": self.rebalance_count,
            "drawdown": current_drawdown,
        }
