"""
Evaluate Zero-Shot RAI v3 Continuous Allocator on Real Market Data
==================================================================
Evaluates Zero-Shot RAI v3 against SPY, 60/40, Equal-Weight, and RAI v2 across:
1. 2020-2024 Out-of-Sample Period (High Volatility, Crashes, Shocks)
2. 2010-2019 Historical Period (10-Year Continuous Bull Run)
"""
import os
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


class ZeroShotRealMarketEnvV3(gym.Env):
    """
    Real Market Evaluation Environment matching SyntheticMultiRegimeEnv (v3) specs.
    Computes exact SMA50, SMA200, Volatility20 on real market price series.
    Uses continuous softmax weight rebalancing with thresholding.
    """
    def __init__(self, price_df, initial_cash=10000.0, history_len=32,
                 max_resources=20, transaction_fee=0.001, rebalance_threshold=0.03):
        super().__init__()
        
        self.price_df = price_df.copy()
        self.prices_matrix = self.price_df.values  # (T, 20)
        self.num_steps, self.num_resources = self.prices_matrix.shape
        self.max_resources = max_resources
        
        self.initial_cash = initial_cash
        self.history_len = history_len
        self.transaction_fee = transaction_fee
        self.rebalance_threshold = rebalance_threshold
        
        # Precompute SMA 50, SMA 200, Volatility 20 on real data
        T, N = self.prices_matrix.shape
        self.sma50 = np.zeros_like(self.prices_matrix)
        self.sma200 = np.zeros_like(self.prices_matrix)
        self.vol20 = np.zeros_like(self.prices_matrix)
        
        for t in range(T):
            t50_start = max(0, t - 50)
            t200_start = max(0, t - 200)
            t20_start = max(0, t - 20)
            
            self.sma50[t] = np.mean(self.prices_matrix[t50_start:t+1], axis=0)
            self.sma200[t] = np.mean(self.prices_matrix[t200_start:t+1], axis=0)
            
            if t > 1:
                sub_p = self.prices_matrix[t20_start:t+1]
                rets = (sub_p[1:] - sub_p[:-1]) / np.maximum(1e-4, sub_p[:-1])
                self.vol20[t] = np.std(rets, axis=0)
            else:
                self.vol20[t] = np.zeros(N)
                
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(max_resources + 1,), dtype=np.float32)
        self.single_obs_dim = 1 + max_resources * 4  # 81
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.history_len * self.single_obs_dim,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.history_len
        
        self.cash = self.initial_cash * 0.50
        prices = self.prices_matrix[self.current_step]
        per_asset_cash = (self.initial_cash * 0.50) / self.num_resources
        self.shares = per_asset_cash / prices
        
        self.obs_history = []
        for _ in range(self.history_len):
            self.obs_history.append(self._get_single_obs())
            
        self.last_wealth = self.initial_cash
        self.rebalance_count = 0
        return self._get_obs(), {}

    def _get_portfolio_value(self):
        prices = self.prices_matrix[self.current_step]
        return self.cash + np.sum(self.shares * prices)

    def _get_single_obs(self):
        prices = self.prices_matrix[self.current_step]
        wealth = max(1e-4, self._get_portfolio_value())
        
        w_cash = np.array([self.cash / wealth], dtype=np.float32)
        
        w_assets = np.zeros(self.max_resources, dtype=np.float32)
        w_assets[:self.num_resources] = (self.shares * prices) / wealth
        
        norm_prices = np.ones(self.max_resources, dtype=np.float32)
        norm_prices[:self.num_resources] = prices / 100.0
        
        s50 = self.sma50[self.current_step]
        s200 = self.sma200[self.current_step]
        trend_ratio = np.ones(self.max_resources, dtype=np.float32)
        trend_ratio[:self.num_resources] = s50 / np.maximum(1e-4, s200)
        
        v20 = np.zeros(self.max_resources, dtype=np.float32)
        v20[:self.num_resources] = self.vol20[self.current_step]
        
        return np.concatenate([w_cash, w_assets, norm_prices, trend_ratio, v20]).astype(np.float32)

    def _get_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        exp_act = np.exp(action - np.max(action))
        target_weights = exp_act / np.sum(exp_act)
        
        target_cash_w = target_weights[0]
        raw_asset_w = target_weights[1:1 + self.num_resources]
        
        total_active_w = target_cash_w + np.sum(raw_asset_w)
        target_cash_w = target_cash_w / total_active_w
        target_asset_w = raw_asset_w / total_active_w
        
        prices = self.prices_matrix[self.current_step]
        current_wealth = max(1e-4, self._get_portfolio_value())
        
        current_asset_w = (self.shares * prices) / current_wealth
        weight_diff = np.abs(current_asset_w - target_asset_w)
        
        if np.sum(weight_diff) > self.rebalance_threshold:
            self.rebalance_count += 1
            target_asset_vals = current_wealth * target_asset_w
            
            rebalance_vol = np.sum(np.abs((self.shares * prices) - target_asset_vals))
            fee = rebalance_vol * self.transaction_fee
            
            net_wealth = max(1e-4, current_wealth - fee)
            self.cash = net_wealth * target_cash_w
            self.shares = (net_wealth * target_asset_w) / np.maximum(1e-4, prices)
            
        self.current_step += 1
        done = self.current_step >= self.num_steps - 1
        
        self.obs_history.pop(0)
        self.obs_history.append(self._get_single_obs())
        
        new_wealth = self._get_portfolio_value()
        return self._get_obs(), 0.0, done, False, {"portfolio_value": new_wealth, "rebalances": self.rebalance_count}


def compute_metrics(equity_curve, rf_daily=0.0):
    eq = np.array(equity_curve, dtype=np.float64)
    returns = (eq[1:] - eq[:-1]) / eq[:-1]
    
    total_return_pct = ((eq[-1] - eq[0]) / eq[0]) * 100.0
    vol_annual_pct = np.std(returns) * np.sqrt(252) * 100.0
    
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (mean_ret - rf_daily) / std_ret * np.sqrt(252) if std_ret > 1e-8 else 0.0
    
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    max_dd_pct = np.min(drawdown) * 100.0
    
    return {
        "final": eq[-1],
        "return_pct": total_return_pct,
        "vol_pct": vol_annual_pct,
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct
    }


def evaluate_v3_model(model, raw, initial_cash=10000.0):
    env = ZeroShotRealMarketEnvV3(price_df=raw, initial_cash=initial_cash)
    obs, _ = env.reset()
    equity_curve = [initial_cash]
    rebalance_count = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        equity_curve.append(info["portfolio_value"])
        rebalance_count = info["rebalances"]
    return equity_curve, rebalance_count


def main():
    print("=" * 90)
    print("  ZERO-SHOT RAI v3: Comprehensive Real Market Benchmark Evaluation")
    print("=" * 90)
    
    test_csv = "./data/real_market_checkpoints/test_prices.csv"
    train_csv = "./data/real_market_checkpoints/train_prices.csv"
    
    datasets = []
    if os.path.exists(test_csv):
        datasets.append(("OUT-OF-SAMPLE (2020-2024)", pd.read_csv(test_csv, index_col=0, parse_dates=True)))
    if os.path.exists(train_csv):
        datasets.append(("HISTORICAL (2010-2019)", pd.read_csv(train_csv, index_col=0, parse_dates=True)))
        
    v3_path = "./data/v0.3_rl_checkpoints/rai_v3_multiregime.zip"
    
    for dataset_name, raw in datasets:
        print(f"\n{'='*90}")
        print(f"  DATASET: {dataset_name}")
        print(f"  Period: {raw.index[0]} to {raw.index[-1]} | {len(raw)} trading days")
        print(f"{'='*90}")
        print(f"{'Model / Strategy':<42} | {'Final ($)':<10} | {'Return (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<6} | {'Max DD (%)':<10}")
        print("-" * 90)
        
        if os.path.exists(v3_path):
            model = PPO.load(v3_path)
            eq, reb_cnt = evaluate_v3_model(model, raw)
            m = compute_metrics(eq)
            print(f"{'Zero-Shot RAI v3 (Multi-Regime Allocator)':<42} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}%")
            print(f"  Total Rebalance Executions: {reb_cnt} / {len(raw)} days ({reb_cnt/len(raw)*100:.1f}% frequency)")
        else:
            print(f"v3 Checkpoint not found: {v3_path}")
            
        # Equal Weight Baseline
        prices_matrix = raw.values
        eq_ew = 10000.0 * np.mean(prices_matrix / prices_matrix[0], axis=1)
        m = compute_metrics(eq_ew)
        print(f"{'Equal-Weight (1/N 20-ETF)':<42} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}%")
        
        # 60/40 Baseline
        if 'SPY' in raw.columns and 'TLT' in raw.columns:
            spy = raw['SPY'].values; tlt = raw['TLT'].values
            eq_6040 = 10000.0 * (0.60 * (spy / spy[0]) + 0.40 * (tlt / tlt[0]))
            m = compute_metrics(eq_6040)
            print(f"{'60/40 Portfolio (SPY / TLT)':<42} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}%")
            
        # SPY Baseline
        spy_prices = raw['SPY'].values
        eq_spy = 10000.0 * (spy_prices / spy_prices[0])
        m = compute_metrics(eq_spy)
        print(f"{'SPY Buy & Hold (S&P 500)':<42} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}%")
        print("-" * 90)

if __name__ == "__main__":
    main()
