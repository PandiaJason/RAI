import os
import yfinance as yf
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from scripts.eval_curriculum_zero_shot import ZeroShotRealMarketEnv, compute_metrics

def main():
    print("=== MULTI-MODEL BENCHMARK COMPARISON ON OUT-OF-SAMPLE TEST DATA (2020 - 2024) ===")
    
    test_csv = "./data/real_market_checkpoints/test_prices.csv"
    if os.path.exists(test_csv):
        raw = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    else:
        tickers_20 = [
            'SPY', 'QQQ', 'IWM', 'EEM', 'VGK', 'EWJ', 'GLD', 'SLV', 'USO', 'UNG',
            'TLT', 'IEF', 'LQD', 'HYG', 'VNQ', 'UUP', 'DBC', 'XLF', 'XLK', 'XLE'
        ]
        raw = yf.download(tickers_20, start='2020-01-01', end='2023-12-31')['Close'].dropna()
        
    prices_matrix = raw.values
    T, N = prices_matrix.shape
    
    results = []
    
    # 1. Zero-Shot RAI (Stage 5 Master Model)
    stage5_path = "./data/v0.2_rl_checkpoints/rai_curriculum_stage_5.zip"
    if os.path.exists(stage5_path):
        env = ZeroShotRealMarketEnv(price_df=raw, initial_cash=10000.0)
        model = RecurrentPPO.load(stage5_path)
        obs, _ = env.reset()
        eq = [10000.0]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            eq.append(info["portfolio_value"])
        m = compute_metrics(eq)
        results.append(("Zero-Shot RAI (Stage 5 Master)", m, "0% Real Data Training (Synthetic World Only)"))
        
    # 2. In-Domain Real-Market Trained PPO (if checkpoint exists)
    real_ppo_path = "./data/real_market_checkpoints/real_market_ppo_final.zip"
    if os.path.exists(real_ppo_path):
        env = ZeroShotRealMarketEnv(price_df=raw, initial_cash=10000.0)
        model = RecurrentPPO.load(real_ppo_path)
        obs, _ = env.reset()
        eq = [10000.0]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            eq.append(info["portfolio_value"])
        m = compute_metrics(eq)
        results.append(("In-Domain Trained PPO", m, "Trained directly on 2010-2019 real stock data"))
        
    # 3. 60/40 Stock/Bond Benchmark (60% SPY / 40% TLT)
    if 'SPY' in raw.columns and 'TLT' in raw.columns:
        spy = raw['SPY'].values
        tlt = raw['TLT'].values
        eq_6040 = 10000.0 * (0.60 * (spy / spy[0]) + 0.40 * (tlt / tlt[0]))
        m = compute_metrics(eq_6040)
        results.append(("60/40 Portfolio (SPY / TLT)", m, "Classic Asset Allocation Benchmark"))
        
    # 4. Equal-Weight Portfolio (1/N across all 20 ETFs)
    normalized_prices = prices_matrix / prices_matrix[0]
    eq_ew = 10000.0 * np.mean(normalized_prices, axis=1)
    m = compute_metrics(eq_ew)
    results.append(("Equal-Weight (1/N ETF Portfolio)", m, "Passive 20-ETF Allocation"))
    
    # 5. SPY Buy & Hold (S&P 500 Index)
    spy = raw['SPY'].values
    eq_spy = 10000.0 * (spy / spy[0])
    m = compute_metrics(eq_spy)
    results.append(("SPY Buy & Hold (S&P 500 Index)", m, "S&P 500 Market Benchmark"))
    
    # 6. Random Action Agent Baseline
    np.random.seed(42)
    env = ZeroShotRealMarketEnv(price_df=raw, initial_cash=10000.0)
    obs, _ = env.reset()
    eq_rand = [10000.0]
    done = False
    while not done:
        rand_action = env.action_space.sample()
        obs, reward, done, _, info = env.step(rand_action)
        eq_rand.append(info["portfolio_value"])
    m = compute_metrics(eq_rand)
    results.append(("Random Action Baseline", m, "Stochastic Buy/Sell/Hold"))
    
    # Print Table
    print("\n" + "="*110)
    print(f"{'Model / Strategy':<34} | {'Final ($)':<10} | {'Return (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<6} | {'Max DD (%)':<10} | {'Training Source'}")
    print("="*110)
    for name, m, desc in results:
        print(f"{name:<34} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}% | {desc}")
    print("="*110 + "\n")

if __name__ == "__main__":
    main()
