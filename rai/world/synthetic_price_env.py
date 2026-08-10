"""
Zero-Shot RAI v2: Train on Synthetic Random Price Worlds
========================================================
Generates random GBM (Geometric Brownian Motion) price series for 20 assets.
Uses the EXACT same observation/action structure as ZeroShotRealMarketEnv.
This ensures zero observation mismatch at test time.

Still 0% real data — all prices are procedurally generated.
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SyntheticPriceWorldEnv(gym.Env):
    """
    Generates a random synthetic market with 20 assets each episode.
    Each asset follows GBM with random drift & volatility.
    Observation structure is identical to ZeroShotRealMarketEnv.
    """
    
    def __init__(self, num_assets=20, episode_len=252, history_len=32, 
                 initial_cash=10000.0, transaction_fee=0.001):
        super().__init__()
        
        self.num_assets = num_assets
        self.max_resources = num_assets
        self.episode_len = episode_len
        self.history_len = history_len
        self.initial_cash = initial_cash
        self.transaction_fee = transaction_fee
        
        # Action: [action_type (0=Hold,1=Buy,2=Sell), asset_index (0..19)]
        self.action_space = spaces.MultiDiscrete([3, num_assets])  # removed Produce (useless here)
        self.single_obs_dim = 2 + 5 * num_assets  # 102
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(history_len * self.single_obs_dim,), dtype=np.float32
        )
        
        self.last_action_type = 0  # for compatibility with callbacks
        self.prices_matrix = None
        self.reset()
        
    def _generate_random_prices(self):
        """Generate 20 random GBM price series for one episode."""
        T = self.episode_len + self.history_len + 10  # extra buffer
        prices = np.zeros((T, self.num_assets), dtype=np.float64)
        
        for i in range(self.num_assets):
            # Random parameters for each asset each episode
            mu_annual = np.random.uniform(-0.15, 0.40)       # drift: -15% to +40% annually
            sigma_annual = np.random.uniform(0.10, 0.60)     # vol: 10% to 60% annually
            initial_price = np.random.uniform(10.0, 500.0)   # starting price $10-$500
            
            mu_daily = mu_annual / 252.0
            sigma_daily = sigma_annual / np.sqrt(252.0)
            
            # GBM: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            log_returns = (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * np.random.randn(T - 1)
            log_prices = np.log(initial_price) + np.concatenate([[0.0], np.cumsum(log_returns)])
            prices[:, i] = np.exp(log_prices)
            
        # Add random correlations: inject shared market factor
        market_factor = np.random.randn(T) * 0.01
        for i in range(self.num_assets):
            beta = np.random.uniform(0.2, 1.5)  # market sensitivity
            prices[:, i] *= np.exp(np.cumsum(beta * market_factor))
            
        return prices
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate fresh random market each episode
        self.prices_matrix = self._generate_random_prices()
        self.current_step = self.history_len
        
        # Start with 50% cash, 50% split across assets (same as real eval)
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
        """Observation structure identical to ZeroShotRealMarketEnv."""
        prices = self.prices_matrix[self.current_step]
        wealth = max(1e-4, self._get_portfolio_value())
        
        q_agent = np.array([self.cash / wealth], dtype=np.float32)
        cap_agent = np.ones(1, dtype=np.float32)
        
        # Portfolio weights per asset
        x_pad = np.zeros(self.max_resources, dtype=np.float32)
        x_pad[:self.num_assets] = (self.shares * prices) / wealth
        
        sub_pad = np.full(self.max_resources, 0.01, dtype=np.float32)
        
        # Normalized prices (same as real env)
        prices_pad = np.ones(self.max_resources, dtype=np.float32)
        prices_pad[:self.num_assets] = prices / 100.0
        
        inputs_pad = np.zeros(self.max_resources, dtype=np.float32)
        output_pad = np.zeros(self.max_resources, dtype=np.float32)
        
        single_obs = np.concatenate([
            q_agent, cap_agent, x_pad, sub_pad, prices_pad, inputs_pad, output_pad
        ]).astype(np.float32)
        
        return single_obs
    
    def _get_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)
    
    def step(self, action):
        act_type = int(action[0])
        res_idx = int(action[1]) % self.num_assets
        self.last_action_type = act_type
        
        prices = self.prices_matrix[self.current_step]
        trade_chunk = 1000.0
        
        if act_type == 1:  # BUY
            if self.cash >= 50.0:
                spend = min(trade_chunk, self.cash)
                fee = spend * self.transaction_fee
                net_spend = spend - fee
                bought_shares = net_spend / max(1e-4, prices[res_idx])
                self.cash -= spend
                self.shares[res_idx] += bought_shares
                
        elif act_type == 2:  # SELL
            current_val = self.shares[res_idx] * prices[res_idx]
            if current_val >= 50.0:
                sell_val = min(trade_chunk, current_val)
                sold_shares = sell_val / max(1e-4, prices[res_idx])
                fee = sell_val * self.transaction_fee
                net_proceeds = sell_val - fee
                self.shares[res_idx] -= sold_shares
                self.cash += net_proceeds
        
        self.current_step += 1
        self.step_count += 1
        done = self.step_count >= self.episode_len
        
        self.obs_history.pop(0)
        self.obs_history.append(self._get_single_obs())
        
        current_wealth = self._get_portfolio_value()
        
        # Reward: log wealth change (encourages growth) + drawdown penalty
        delta_log_w = np.log(max(1e-4, current_wealth)) - np.log(max(1e-4, self.last_wealth))
        reward = float(np.clip(delta_log_w, -1.0, 1.0))
        
        self.last_wealth = current_wealth
        
        return self._get_obs(), reward, done, False, {"portfolio_value": current_wealth}
