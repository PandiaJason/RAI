"""
Zero-Shot RAI v3: Multi-Regime Synthetic World Environment
===========================================================
Upgrades over v2:
1. CONTINUOUS ACTION SPACE: Softmax portfolio target weights [w_cash, w_1..w_20].
   Rebalances to target weights, avoiding fixed $1,000 discrete trade churn.
2. MULTI-REGIME GENERATOR: Procedurally generates Bull, Bear, and Volatile Sideways markets.
   Teaches the agent to Hold & Compound in Bull runs, and De-risk in Crashes.
3. TREND & VOLATILITY PERCEPTION: Includes 50-day/200-day trend ratios and 20-day volatility.
4. LONGER EPISODES: 504 steps (2 synthetic years) for multi-year memory.

100% Synthetic Data (0% Real Data).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SyntheticMultiRegimeEnv(gym.Env):
    def __init__(self, num_assets=20, episode_len=504, history_len=32,
                 initial_cash=10000.0, transaction_fee=0.001, rebalance_threshold=0.03):
        super().__init__()
        
        self.num_assets = num_assets
        self.episode_len = episode_len
        self.history_len = history_len
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        self.rebalance_threshold = rebalance_threshold  # minimum weight shift to rebalance
        
        # Action space: Continuous target weights for [Cash, Asset_0, ..., Asset_19] (21 values)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_assets + 1,), dtype=np.float32)
        
        # Single timestep feature count:
        # 1 (cash_weight) + 20 (asset_weights) + 20 (prices/100) + 20 (SMA50/SMA200) + 20 (Vol20) = 81
        self.single_obs_dim = 1 + num_assets + num_assets + num_assets + num_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(history_len * self.single_obs_dim,), dtype=np.float32
        )
        
        self.prices_matrix = None
        self.sma50 = None
        self.sma200 = None
        self.vol20 = None
        self.reset()

    def _generate_multi_regime_prices(self):
        """Generate 20 asset prices across 3 synthetic regimes: Bull, Bear, Sideways."""
        T = self.episode_len + self.history_len + 210  # extra buffer for 200-day SMA
        prices = np.zeros((T, self.num_assets), dtype=np.float64)
        
        # Decide regime for this episode
        regime_roll = np.random.rand()
        if regime_roll < 0.45:
            # REGIME 0: STEADY BULL MARKET (Low vol, strong positive drift)
            mu_range = (0.12, 0.40)
            sigma_range = (0.08, 0.22)
        elif regime_roll < 0.75:
            # REGIME 1: BEAR / CRASH MARKET (High vol, negative drift)
            mu_range = (-0.35, -0.05)
            sigma_range = (0.25, 0.60)
        else:
            # REGIME 2: VOLATILE SIDEWAYS MARKET (High vol, flat drift)
            mu_range = (-0.05, 0.08)
            sigma_range = (0.20, 0.50)
            
        for i in range(self.num_assets):
            mu_annual = np.random.uniform(*mu_range)
            sigma_annual = np.random.uniform(*sigma_range)
            initial_price = np.random.uniform(20.0, 300.0)
            
            mu_daily = mu_annual / 252.0
            sigma_daily = sigma_annual / np.sqrt(252.0)
            
            log_returns = (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * np.random.randn(T - 1)
            log_prices = np.log(initial_price) + np.concatenate([[0.0], np.cumsum(log_returns)])
            prices[:, i] = np.exp(log_prices)
            
        # Add market factor correlation
        market_factor = np.random.randn(T) * 0.008
        for i in range(self.num_assets):
            beta = np.random.uniform(0.3, 1.4)
            prices[:, i] *= np.exp(np.cumsum(beta * market_factor))
            
        # Precompute indicators (SMA 50, SMA 200, Volatility 20)
        sma50 = np.zeros_like(prices)
        sma200 = np.zeros_like(prices)
        vol20 = np.zeros_like(prices)
        
        for t in range(200, T):
            sma50[t] = np.mean(prices[t-50:t], axis=0)
            sma200[t] = np.mean(prices[t-200:t], axis=0)
            returns = (prices[t-20:t] - prices[t-21:t-1]) / prices[t-21:t-1]
            vol20[t] = np.std(returns, axis=0)
            
        return prices, sma50, sma200, vol20

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.prices_matrix, self.sma50, self.sma200, self.vol20 = self._generate_multi_regime_prices()
        self.current_step = self.history_len + 200  # start after SMA200 buffer
        
        self.cash = self.initial_cash * 0.50
        prices = self.prices_matrix[self.current_step]
        per_asset_cash = (self.initial_cash * 0.50) / self.num_assets
        self.shares = per_asset_cash / prices
        
        self.obs_history = []
        for _ in range(self.history_len):
            self.obs_history.append(self._get_single_obs())
            
        self.last_wealth = self.initial_cash
        self.step_count = 0
        return self._get_obs(), {}

    def _get_portfolio_value(self):
        prices = self.prices_matrix[self.current_step]
        return self.cash + np.sum(self.shares * prices)

    def _get_single_obs(self):
        prices = self.prices_matrix[self.current_step]
        wealth = max(1e-4, self._get_portfolio_value())
        
        w_cash = np.array([self.cash / wealth], dtype=np.float32)
        w_assets = ((self.shares * prices) / wealth).astype(np.float32)
        
        norm_prices = (prices / 100.0).astype(np.float32)
        
        # Trend ratio SMA50 / SMA200 (1.0 = neutral, >1.0 = uptrend, <1.0 = downtrend)
        s50 = self.sma50[self.current_step]
        s200 = self.sma200[self.current_step]
        trend_ratio = (s50 / np.maximum(1e-4, s200)).astype(np.float32)
        
        v20 = self.vol20[self.current_step].astype(np.float32)
        
        single_obs = np.concatenate([
            w_cash, w_assets, norm_prices, trend_ratio, v20
        ]).astype(np.float32)
        
        return single_obs

    def _get_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        # Convert continuous action logits to softmax target weights
        exp_act = np.exp(action - np.max(action))
        target_weights = exp_act / np.sum(exp_act)  # Length 21: [cash_w, asset_0_w, ..., asset_19_w]
        
        target_cash_w = target_weights[0]
        target_asset_w = target_weights[1:]
        
        prices = self.prices_matrix[self.current_step]
        current_wealth = max(1e-4, self._get_portfolio_value())
        
        current_asset_w = (self.shares * prices) / current_wealth
        weight_diff = np.abs(current_asset_w - target_asset_w)
        
        # Only rebalance if total allocation shift exceeds threshold (prevents fee churn!)
        if np.sum(weight_diff) > self.rebalance_threshold:
            # Rebalance portfolio to target weights
            target_cash_val = current_wealth * target_cash_w
            target_asset_vals = current_wealth * target_asset_w
            
            # Fee is charged on volume of rebalanced capital
            rebalance_volume = np.sum(np.abs((self.shares * prices) - target_asset_vals))
            fee = rebalance_volume * self.transaction_fee
            
            net_wealth = max(1e-4, current_wealth - fee)
            self.cash = net_wealth * target_cash_w
            self.shares = (net_wealth * target_asset_w) / np.maximum(1e-4, prices)
            
        self.current_step += 1
        self.step_count += 1
        done = self.step_count >= self.episode_len
        
        self.obs_history.pop(0)
        self.obs_history.append(self._get_single_obs())
        
        new_wealth = self._get_portfolio_value()
        
        # Reward: log return + Sharpe-like volatility penalty
        delta_log_w = np.log(max(1e-4, new_wealth)) - np.log(max(1e-4, self.last_wealth))
        reward = float(np.clip(delta_log_w, -1.0, 1.0))
        
        self.last_wealth = new_wealth
        
        return self._get_obs(), reward, done, False, {"portfolio_value": new_wealth}
