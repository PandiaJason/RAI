import os
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from scripts.eval_curriculum_zero_shot import ZeroShotRealMarketEnv, compute_metrics

def main():
    print("=== ZERO-SHOT EVALUATION ON NOVEL STRESS-TESTED MARKET DATASET (HIGH VOLATILITY REGIME) ===")
    print("Model source: Synthetic Curriculum (0% real-data training)")
    print("Scenario: Real 2020-2024 Market Data injected with +50% Volatility Shocks & Synthetic Inflation Panics\n")
    
    test_csv = "./data/real_market_checkpoints/test_prices.csv"
    raw = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    
    # Generate Stress-Tested High Volatility Novel Dataset
    np.random.seed(999)
    pct_changes = raw.pct_change().dropna()
    # Inject 1.5x volatility multiplier and random macro regime shifts
    shocked_returns = pct_changes * 1.5 + np.random.normal(0, 0.005, size=pct_changes.shape)
    shocked_prices = raw.iloc[0].values * (1 + shocked_returns).cumprod()
    shocked_df = pd.DataFrame(shocked_prices, index=pct_changes.index, columns=raw.columns)
    
    checkpoints = [
        ("Stage 1 (Simple Survival)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_1.zip"),
        ("Stage 2 (Scarcity & Dynamic AMM)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_2.zip"),
        ("Stage 3 (Production Chains)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_3.zip"),
        ("Stage 4 (Market Competition)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_4.zip"),
        ("Stage 5 (Full XEconomics)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_5.zip")
    ]
    
    print("="*105)
    print(f"{'Curriculum Stage Model':<36} | {'Final ($)':<10} | {'Return (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<6} | {'Max DD (%)':<10}")
    print("="*105)
    
    for name, path in checkpoints:
        if os.path.exists(path):
            env = ZeroShotRealMarketEnv(price_df=shocked_df, initial_cash=10000.0)
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
            
    # SPY Stress Benchmark
    spy_prices = shocked_df['SPY'].values
    spy_eq = 10000.0 * (spy_prices / spy_prices[0])
    spy_metrics = compute_metrics(spy_eq)
    print(f"{'SPY Buy & Hold (Stress-Tested)':<36} | ${spy_metrics['final']:<9.2f} | {spy_metrics['return_pct']:<9.2f}% | {spy_metrics['vol_pct']:<7.2f}% | {spy_metrics['sharpe']:<6.2f} | {spy_metrics['max_dd_pct']:<9.2f}%")
    
    # Equal Weight Benchmark
    prices_matrix = shocked_df.values
    eq_ew = 10000.0 * np.mean(prices_matrix / prices_matrix[0], axis=1)
    ew_metrics = compute_metrics(eq_ew)
    print(f"{'Equal-Weight 20-ETF Basket (Shocked)':<36} | ${ew_metrics['final']:<9.2f} | {ew_metrics['return_pct']:<9.2f}% | {ew_metrics['vol_pct']:<7.2f}% | {ew_metrics['sharpe']:<6.2f} | {ew_metrics['max_dd_pct']:<9.2f}%")
    print("="*105 + "\n")

if __name__ == "__main__":
    main()
