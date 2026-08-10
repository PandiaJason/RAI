import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

class RealMarketEnv(gym.Env):
    """
    RAI Real Market Trading Environment
    Allows an RL agent to trade anonymous real-world financial assets (ETFs, stocks, commodities).
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, price_df, initial_cash=10000.0, history_len=10, transaction_fee=0.001):
        super().__init__()
        
        self.price_df = price_df.copy()
        self.prices_matrix = self.price_df.values # Shape: (T, N)
        self.timestamps = self.price_df.index
        self.num_steps, self.num_assets = self.prices_matrix.shape
        
        self.initial_cash = initial_cash
        self.history_len = history_len
        self.transaction_fee = transaction_fee
        
        # Calculate daily log returns for observation features
        prices_clean = np.where(self.prices_matrix <= 0, 1e-4, self.prices_matrix)
        log_returns = np.zeros_like(prices_clean)
        log_returns[1:] = np.log(prices_clean[1:] / prices_clean[:-1])
        self.log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Action space: [ActionType (0:Hold, 1:Buy, 2:Sell), TargetAsset (0..N-1)]
        self.action_space = spaces.MultiDiscrete([3, self.num_assets])
        
        # Observation space:
        # 1. Historical log returns: (history_len * num_assets)
        # 2. Portfolio asset values / total wealth: (num_assets)
        # 3. Cash fraction: (1)
        # 4. Normalized wealth (W_t / W_0): (1)
        self.single_obs_dim = (self.history_len * self.num_assets) + self.num_assets + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.single_obs_dim,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.history_len
        self.cash = self.initial_cash
        self.shares = np.zeros(self.num_assets, dtype=np.float32)
        
        self.last_wealth = self.initial_cash
        self.last_action_type = 0
        
        return self._get_obs(), {}
        
    def _get_portfolio_value(self):
        prices = self.prices_matrix[self.current_step]
        return self.cash + np.sum(self.shares * prices)
        
    def _get_obs(self):
        # 1. Rolling window of historical log returns
        ret_window = self.log_returns[self.current_step - self.history_len : self.current_step]
        ret_flat = ret_window.flatten()
        
        # 2. Current portfolio state
        prices = self.prices_matrix[self.current_step]
        wealth = self._get_portfolio_value()
        
        asset_values = self.shares * prices
        asset_weights = asset_values / max(1e-4, wealth)
        cash_fraction = np.array([self.cash / max(1e-4, wealth)], dtype=np.float32)
        norm_wealth = np.array([wealth / self.initial_cash], dtype=np.float32)
        
        obs = np.concatenate([ret_flat, asset_weights, cash_fraction, norm_wealth]).astype(np.float32)
        return obs
        
    def step(self, action):
        act_type = int(action[0])
        asset_idx = int(action[1])
        self.last_action_type = act_type
        
        prices = self.prices_matrix[self.current_step]
        trade_chunk = 1000.0 # Trade unit in USD
        
        if act_type == 1: # BUY
            if self.cash >= 50.0: # Minimum trade threshold
                spend = min(trade_chunk, self.cash)
                fee = spend * self.transaction_fee
                net_spend = spend - fee
                bought_shares = net_spend / max(1e-4, prices[asset_idx])
                
                self.cash -= spend
                self.shares[asset_idx] += bought_shares
                
        elif act_type == 2: # SELL
            current_asset_val = self.shares[asset_idx] * prices[asset_idx]
            if current_asset_val >= 50.0:
                sell_val = min(trade_chunk, current_asset_val)
                sold_shares = sell_val / max(1e-4, prices[asset_idx])
                fee = sell_val * self.transaction_fee
                net_proceeds = sell_val - fee
                
                self.shares[asset_idx] -= sold_shares
                self.cash += net_proceeds
                
        # Advance step
        self.current_step += 1
        done = False
        truncated = False
        
        if self.current_step >= self.num_steps - 1:
            done = True
            
        current_wealth = self._get_portfolio_value()
        
        # Check bankruptcy (loss of 90% capital)
        if current_wealth < 0.10 * self.initial_cash:
            done = True
            reward = -5.0
        else:
            # Reward: Dense Log Return + small survival bonus
            eps = 1e-4
            delta_log_w = np.log(current_wealth + eps) - np.log(self.last_wealth + eps)
            reward = float(0.20 * np.clip(delta_log_w, -1.0, 1.0) + 0.001)
            
        self.last_wealth = current_wealth
        
        return self._get_obs(), reward, done, truncated, {"portfolio_value": current_wealth}
