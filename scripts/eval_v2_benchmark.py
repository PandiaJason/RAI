"""
Evaluate Zero-Shot RAI v2 on real market data and compare against all baselines.
"""
import os
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from scripts.eval_curriculum_zero_shot import ZeroShotRealMarketEnv, compute_metrics


def evaluate_model(model, raw, initial_cash=10000.0):
    """Run a model through real market data and return equity curve."""
    env = ZeroShotRealMarketEnv(price_df=raw, initial_cash=initial_cash)
    obs, _ = env.reset()
    equity_curve = [initial_cash]
    actions_taken = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        act_type = int(np.array(action[0]).flat[0]) if hasattr(action[0], 'flat') else int(action[0])
        res_idx = int(np.array(action[1]).flat[0]) if hasattr(action[1], 'flat') else int(action[1])
        actions_taken.append(act_type)
        obs, reward, done, _, info = env.step(action)
        equity_curve.append(info["portfolio_value"])
    return equity_curve, actions_taken


def main():
    print("=" * 90)
    print("  ZERO-SHOT RAI v2: Complete Benchmark Comparison on Real Market Data")
    print("=" * 90)
    
    # Load real market data
    test_csv = "./data/real_market_checkpoints/test_prices.csv"
    train_csv = "./data/real_market_checkpoints/train_prices.csv"
    
    datasets = []
    if os.path.exists(test_csv):
        datasets.append(("OUT-OF-SAMPLE (2020-2024)", pd.read_csv(test_csv, index_col=0, parse_dates=True)))
    if os.path.exists(train_csv):
        datasets.append(("HISTORICAL (2010-2019)", pd.read_csv(train_csv, index_col=0, parse_dates=True)))
    
    # Models to compare
    models_to_test = [
        ("Zero-Shot RAI v2 (Synthetic Prices)", "./data/v0.2_rl_checkpoints/rai_v2_synthetic_price.zip", "0% Real Data (Random GBM Worlds)"),
        ("Zero-Shot RAI v1 Stage 5 (XEconomics)", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_5.zip", "0% Real Data (XEconomics World)"),
    ]
    
    for dataset_name, raw in datasets:
        print(f"\n{'='*90}")
        print(f"  DATASET: {dataset_name}")
        print(f"  Period: {raw.index[0]} to {raw.index[-1]} | {len(raw)} trading days")
        print(f"{'='*90}")
        print(f"{'Model / Strategy':<42} | {'Final ($)':<10} | {'Return (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<6} | {'Max DD (%)':<10}")
        print("-" * 90)
        
        for name, path, desc in models_to_test:
            if os.path.exists(path):
                model = RecurrentPPO.load(path)
                eq, acts = evaluate_model(model, raw)
                m = compute_metrics(eq)
                
                # Action breakdown
                act_counts = pd.Series(acts).value_counts()
                hold_pct = act_counts.get(0, 0) / len(acts) * 100
                buy_pct = act_counts.get(1, 0) / len(acts) * 100
                sell_pct = act_counts.get(2, 0) / len(acts) * 100
                
                print(f"{name:<42} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}%")
                print(f"  Actions: Hold={hold_pct:.0f}% Buy={buy_pct:.0f}% Sell={sell_pct:.0f}%")
            else:
                print(f"{name:<42} | Checkpoint not found: {path}")
        
        # Random Agent
        np.random.seed(42)
        env = ZeroShotRealMarketEnv(price_df=raw, initial_cash=10000.0)
        obs, _ = env.reset()
        eq_rand = [10000.0]
        done = False
        while not done:
            obs, _, done, _, info = env.step(env.action_space.sample())
            eq_rand.append(info["portfolio_value"])
        m = compute_metrics(eq_rand)
        print(f"{'Random Action Baseline':<42} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}%")
        
        # Equal Weight
        prices_matrix = raw.values
        eq_ew = 10000.0 * np.mean(prices_matrix / prices_matrix[0], axis=1)
        m = compute_metrics(eq_ew)
        print(f"{'Equal-Weight (1/N 20-ETF)':<42} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}%")
        
        # 60/40
        if 'SPY' in raw.columns and 'TLT' in raw.columns:
            spy = raw['SPY'].values; tlt = raw['TLT'].values
            eq_6040 = 10000.0 * (0.60 * (spy / spy[0]) + 0.40 * (tlt / tlt[0]))
            m = compute_metrics(eq_6040)
            print(f"{'60/40 Portfolio (SPY / TLT)':<42} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}%")
        
        # SPY
        spy_prices = raw['SPY'].values
        eq_spy = 10000.0 * (spy_prices / spy_prices[0])
        m = compute_metrics(eq_spy)
        print(f"{'SPY Buy & Hold (S&P 500)':<42} | ${m['final']:<9.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}%")
        
        print("-" * 90)

if __name__ == "__main__":
    main()
