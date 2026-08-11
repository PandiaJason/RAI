import os
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from rai.world.real_env import RealMarketEnv

def compute_financial_metrics(equity_curve, risk_free_rate=0.02):
    """Calculates Total Return, Annualized Volatility, Max Drawdown, and Sharpe Ratio."""
    equity_curve = np.array(equity_curve)
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    
    daily_returns = (equity_curve[1:] - equity_curve[:-1]) / equity_curve[:-1]
    ann_return = np.mean(daily_returns) * 252
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    
    sharpe = (ann_return - risk_free_rate) / max(1e-4, ann_vol)
    
    # Max Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = np.min(drawdown)
    
    return {
        "final_wealth": equity_curve[-1],
        "total_return_pct": total_return * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown * 100
    }

def run_backtest(agent_type, test_df, model_path=None):
    env = RealMarketEnv(price_df=test_df, initial_cash=10000.0, history_len=10)
    obs, _ = env.reset()
    
    if agent_type == "ppo":
        model = RecurrentPPO.load(model_path)
        
    equity_curve = [10000.0]
    done = False
    
    # For Equal-Weight baseline: allocation array
    eq_shares = None
    
    while not done:
        prices = env.prices_matrix[env.current_step]
        
        if agent_type == "ppo":
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            equity_curve.append(info["portfolio_value"])
            
        elif agent_type == "spy_buy_hold":
            # Buy 100% SPY at step 0 (SPY is at index 0 or 'SPY' col index)
            spy_idx = list(test_df.columns).index('SPY') if 'SPY' in test_df.columns else 0
            if env.current_step == env.history_len:
                # Spend all cash on SPY
                shares = env.cash / prices[spy_idx]
                env.cash = 0.0
                env.shares[spy_idx] = shares
                
            env.current_step += 1
            if env.current_step >= env.num_steps - 1:
                done = True
            val = env._get_portfolio_value()
            equity_curve.append(val)
            
        elif agent_type == "equal_weight":
            if env.current_step == env.history_len:
                # Buy 10% in each asset
                per_asset_cash = env.cash / env.num_assets
                shares = per_asset_cash / prices
                env.cash = 0.0
                env.shares = shares
                
            env.current_step += 1
            if env.current_step >= env.num_steps - 1:
                done = True
            val = env._get_portfolio_value()
            equity_curve.append(val)
            
        elif agent_type == "random":
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            equity_curve.append(info["portfolio_value"])
            
    return compute_financial_metrics(equity_curve), equity_curve

def main():
    print("=== Out-of-Sample Financial Backtest (2020 - 2024) ===")
    
    test_csv = "./data/real_market_checkpoints/test_prices.csv"
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found. Please run train_real_world.py first.")
        return
        
    test_df = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    model_path = "./data/real_market_checkpoints/rai_real_ppo_final"
    
    strategies = [
        ("RAI Agent (Recurrent PPO)", "ppo"),
        ("SPY Buy & Hold (S&P 500)", "spy_buy_hold"),
        ("Equal-Weight 10-Asset Portfolio", "equal_weight"),
        ("Random Trading Strategy", "random")
    ]
    
    results = []
    
    print("\n" + "="*80)
    print(f"{'Strategy':<32} | {'Final ($)':<10} | {'Return (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<6} | {'Max DD (%)':<10}")
    print("="*80)
    
    for name, stype in strategies:
        try:
            m_path = model_path if stype == "ppo" else None
            metrics, _ = run_backtest(stype, test_df, model_path=m_path)
            results.append((name, metrics))
            
            print(f"{name:<32} | ${metrics['final_wealth']:<9.2f} | {metrics['total_return_pct']:<9.2f}% | {metrics['ann_vol_pct']:<7.2f}% | {metrics['sharpe_ratio']:<6.2f} | {metrics['max_drawdown_pct']:<9.2f}%")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
