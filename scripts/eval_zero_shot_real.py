import os
import yfinance as yf
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
import gymnasium as gym
from gymnasium import spaces

class ZeroShotRealMarketEnv(gym.Env):
    """
    Adapter Environment that maps 20 Real-World Market Assets to the exact 
    observation vector space (510-dim) and action space (MultiDiscrete[4, 20])
    of the synthetic 2,000-Year RAI World Model.
    """
    def __init__(self, price_df, initial_cash=10000.0, history_len=32, transaction_fee=0.001):
        super().__init__()
        
        self.price_df = price_df.copy()
        self.prices_matrix = self.price_df.values # Shape (T, 20)
        self.timestamps = self.price_df.index
        self.num_steps, self.num_resources = self.prices_matrix.shape # (T, 20)
        
        self.initial_cash = initial_cash
        self.history_len = history_len
        self.transaction_fee = transaction_fee
        
        # Action space matching RAI World: [ActionType (0..3), ResourceIndex (0..19)]
        self.action_space = spaces.MultiDiscrete([4, self.num_resources])
        
        # Single obs dim matching RAI World: history_len * (2 + 5 * num_resources) = 5 * 102 = 510
        self.single_obs_dim = self.history_len * (2 + 5 * self.num_resources)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.single_obs_dim,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.history_len
        self.cash = self.initial_cash
        self.shares = np.zeros(self.num_resources, dtype=np.float32)
        
        self.obs_history = []
        for _ in range(self.history_len):
            self.obs_history.append(self._get_single_obs())
            
        self.last_wealth = self.initial_cash
        return self._get_obs(), {}
        
    def _get_portfolio_value(self):
        prices = self.prices_matrix[self.current_step]
        return self.cash + np.sum(self.shares * prices)
        
    def _get_single_obs(self):
        prices = self.prices_matrix[self.current_step]
        wealth = self._get_portfolio_value()
        
        # Exact RAIWorldEnv vector: 2 + 5N = 102
        q_agent = np.array([self.cash / max(1e-4, wealth)], dtype=np.float32)
        cap_agent = np.ones(1, dtype=np.float32)
        x_agent = (self.shares * prices) / max(1e-4, wealth)
        sub_agent = np.full(self.num_resources, 0.01, dtype=np.float32)
        norm_prices = prices / 100.0
        inputs_agent = np.zeros(self.num_resources, dtype=np.float32)
        output_agent = np.zeros(self.num_resources, dtype=np.float32)
        
        single_obs = np.concatenate([
            q_agent, cap_agent, x_agent, sub_agent, norm_prices, inputs_agent, output_agent
        ]).astype(np.float32)
        
        return single_obs
        
    def _get_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)
        
    def step(self, action):
        act_type = int(action[0])
        res_idx = int(action[1])
        
        prices = self.prices_matrix[self.current_step]
        trade_chunk = 1000.0 # Trade chunk in USD
        
        if act_type == 1: # BUY
            if self.cash >= 50.0:
                spend = min(trade_chunk, self.cash)
                fee = spend * self.transaction_fee
                net_spend = spend - fee
                bought_shares = net_spend / max(1e-4, prices[res_idx])
                
                self.cash -= spend
                self.shares[res_idx] += bought_shares
                
        elif act_type == 2: # SELL
            current_val = self.shares[res_idx] * prices[res_idx]
            if current_val >= 50.0:
                sell_val = min(trade_chunk, current_val)
                sold_shares = sell_val / max(1e-4, prices[res_idx])
                fee = sell_val * self.transaction_fee
                net_proceeds = sell_val - fee
                
                self.shares[res_idx] -= sold_shares
                self.cash += net_proceeds
                
        # Advance step
        self.current_step += 1
        done = False
        if self.current_step >= self.num_steps - 1:
            done = True
            
        # Update history
        self.obs_history.pop(0)
        self.obs_history.append(self._get_single_obs())
        
        current_wealth = self._get_portfolio_value()
        delta_log_w = np.log(current_wealth + 1e-4) - np.log(self.last_wealth + 1e-4)
        reward = float(0.20 * np.clip(delta_log_w, -1.0, 1.0) + 0.001)
        self.last_wealth = current_wealth
        
        return self._get_obs(), reward, done, False, {"portfolio_value": current_wealth}

def compute_metrics(equity_curve, risk_free_rate=0.02):
    eq = np.array(equity_curve)
    tot_ret = (eq[-1] - eq[0]) / eq[0]
    daily_ret = (eq[1:] - eq[:-1]) / eq[:-1]
    ann_vol = np.std(daily_ret) * np.sqrt(252)
    ann_ret = np.mean(daily_ret) * 252
    sharpe = (ann_ret - risk_free_rate) / max(1e-4, ann_vol)
    peak = np.maximum.accumulate(eq)
    max_dd = np.min((eq - peak) / peak)
    return {
        "final": eq[-1],
        "return_pct": tot_ret * 100,
        "vol_pct": ann_vol * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100
    }

def main():
    print("=== TRUE ZERO-SHOT EVALUATION ON REAL MARKET DATA (2020 - 2024) ===")
    print("Model source: Synthetic 2,000-Year Civilization (data/v0.2_rl_checkpoints/rai_world_ppo_final)")
    print("Real Data: 20 Real-World Asset ETFs (0% training on real market data)\n")
    
    # 20 diverse real asset ETFs
    tickers_20 = [
        'SPY', 'QQQ', 'IWM', 'EEM', 'VGK', 'EWJ', 'GLD', 'SLV', 'USO', 'UNG',
        'TLT', 'IEF', 'LQD', 'HYG', 'VNQ', 'UUP', 'DBC', 'XLF', 'XLK', 'XLE'
    ]
    
    raw = yf.download(tickers_20, start='2020-01-01', end='2023-12-31')['Close'].dropna()
    print(f"Loaded 20 real assets for test period ({len(raw)} trading days).")
    
    env = ZeroShotRealMarketEnv(price_df=raw, initial_cash=10000.0)
    
    model_path = "data/v0.2_rl_checkpoints/rai_world_ppo_final.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    model = RecurrentPPO.load(model_path)
    
    obs, _ = env.reset()
    equity_curve = [10000.0]
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        equity_curve.append(info["portfolio_value"])
        
    metrics = compute_metrics(equity_curve)
    
    # Compare with SPY Buy & Hold
    spy_prices = raw['SPY'].values
    spy_eq = 10000.0 * (spy_prices / spy_prices[0])
    spy_metrics = compute_metrics(spy_eq)
    
    print("\n" + "="*80)
    print(f"{'Strategy':<36} | {'Final ($)':<10} | {'Return (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<6} | {'Max DD (%)':<10}")
    print("="*80)
    print(f"{'ZERO-SHOT Synthetic RAI Agent':<36} | ${metrics['final']:<9.2f} | {metrics['return_pct']:<9.2f}% | {metrics['vol_pct']:<7.2f}% | {metrics['sharpe']:<6.2f} | {metrics['max_dd_pct']:<9.2f}%")
    print(f"{'SPY Buy & Hold (S&P 500 Index)':<36} | ${spy_metrics['final']:<9.2f} | {spy_metrics['return_pct']:<9.2f}% | {spy_metrics['vol_pct']:<7.2f}% | {spy_metrics['sharpe']:<6.2f} | {spy_metrics['max_dd_pct']:<9.2f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
