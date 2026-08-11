import os
import yfinance as yf
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
import gymnasium as gym
from gymnasium import spaces
from scripts.eval_curriculum_zero_shot import ZeroShotRealMarketEnv, compute_metrics

def main():
    print("=== ZERO-SHOT EVALUATION ON PREVIOUS HISTORICAL YEARS (2010 - 2019) ===")
    print("Model source: Synthetic Curriculum (0% real-data training)")
    print("Historical Dataset: 2010-01-04 to 2019-12-31 (2,516 trading days / 10 Years)\n")
    
    train_csv = "./data/real_market_checkpoints/train_prices.csv"
    if os.path.exists(train_csv):
        raw = pd.read_csv(train_csv, index_col=0, parse_dates=True)
    else:
        tickers_20 = [
            'SPY', 'QQQ', 'IWM', 'EEM', 'VGK', 'EWJ', 'GLD', 'SLV', 'USO', 'UNG',
            'TLT', 'IEF', 'LQD', 'HYG', 'VNQ', 'UUP', 'DBC', 'XLF', 'XLK', 'XLE'
        ]
        raw = yf.download(tickers_20, start='2010-01-01', end='2019-12-31')['Close'].dropna()
        
    checkpoints = [
        ("Stage 1 (Simple Survival)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_1.zip"),
        ("Stage 2 (Scarcity & Dynamic AMM)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_2.zip"),
        ("Stage 3 (Production Chains)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_3.zip"),
        ("Stage 4 (Market Competition)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_4.zip"),
        ("Stage 5 (Full XEconomics)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_5.zip")
    ]
    
    print("="*85)
    print(f"{'Curriculum Stage Model':<36} | {'Final ($)':<10} | {'Return (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<6} | {'Max DD (%)':<10}")
    print("="*85)
    
    for name, path in checkpoints:
        if os.path.exists(path):
            env = ZeroShotRealMarketEnv(price_df=raw, initial_cash=10000.0)
            model = RecurrentPPO.load(path)
            
            obs, _ = env.reset()
            equity_curve = [10000.0]
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                equity_curve.append(info["portfolio_value"])
                
            metrics = compute_metrics(equity_curve)
            print(f"{name:<36} | ${metrics['final']:<9.2f} | {metrics['return_pct']:<9.2f}% | {metrics['vol_pct']:<7.2f}% | {metrics['sharpe']:<6.2f} | {metrics['max_dd_pct']:<9.2f}%")
        else:
            print(f"{name:<36} | Not ready yet...")
            
    # SPY benchmark
    spy_prices = raw['SPY'].values
    spy_eq = 10000.0 * (spy_prices / spy_prices[0])
    spy_metrics = compute_metrics(spy_eq)
    print(f"{'SPY Buy & Hold (S&P 500 Index)':<36} | ${spy_metrics['final']:<9.2f} | {spy_metrics['return_pct']:<9.2f}% | {spy_metrics['vol_pct']:<7.2f}% | {spy_metrics['sharpe']:<6.2f} | {spy_metrics['max_dd_pct']:<9.2f}%")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
