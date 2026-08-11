import os
import yfinance as yf
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO
from scripts.eval_curriculum_zero_shot import ZeroShotRealMarketEnv

def debug_actions(stage_name, model_path):
    print(f"\n--- DEBUGGING ACTIONS FOR: {stage_name} ---")
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return
        
    tickers_20 = [
        'SPY', 'QQQ', 'IWM', 'EEM', 'VGK', 'EWJ', 'GLD', 'SLV', 'USO', 'UNG',
        'TLT', 'IEF', 'LQD', 'HYG', 'VNQ', 'UUP', 'DBC', 'XLF', 'XLK', 'XLE'
    ]
    
    raw = yf.download(tickers_20, start='2020-01-01', end='2023-12-31')['Close'].dropna()
    env = ZeroShotRealMarketEnv(price_df=raw, initial_cash=10000.0)
    model = RecurrentPPO.load(model_path)
    
    obs, _ = env.reset()
    actions_taken = []
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(action)
        obs, reward, done, _, info = env.step(action)
        if done:
            break
            
    actions_arr = np.array(actions_taken)
    act_types = actions_arr[:, 0]
    
    hold_c = np.sum(act_types == 0)
    buy_c = np.sum(act_types == 1)
    sell_c = np.sum(act_types == 2)
    prod_c = np.sum(act_types == 3)
    
    print(f"First 100 Actions Breakdown:")
    print(f"  Hold    (0): {hold_c} ({hold_c}%)")
    print(f"  Buy     (1): {buy_c} ({buy_c}%)")
    print(f"  Sell    (2): {sell_c} ({sell_c}%)")
    print(f"  Produce (3): {prod_c} ({prod_c}%)")
    print(f"First 15 raw action pairs [type, target]:")
    print(actions_arr[:15])

def main():
    debug_actions("Stage 1 Model", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_1.zip")
    debug_actions("Stage 2 Model", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_2.zip")
    debug_actions("Stage 3 Model", "./data/v0.2_rl_checkpoints/rai_curriculum_stage_3.zip")

if __name__ == "__main__":
    main()
