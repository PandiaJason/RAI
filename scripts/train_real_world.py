import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO
from rai.world.real_env import RealMarketEnv

def main():
    print("=== Training RAI on Real Financial Market Data (2010 - 2019) ===")
    
    tickers = ['SPY', 'QQQ', 'GLD', 'USO', 'TLT', 'EEM', 'VNQ', 'UUP', 'DBC', 'HYG']
    print(f"Fetching historical data for {len(tickers)} asset ETFs...")
    
    raw_data = yf.download(tickers, start='2010-01-01', end='2024-01-01')['Close']
    raw_data = raw_data.dropna()
    
    # Split train vs out-of-sample test
    train_df = raw_data.loc['2010-01-01':'2019-12-31']
    test_df = raw_data.loc['2020-01-01':'2023-12-31']
    
    print(f"Train period: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} trading days)")
    print(f"Test period:  {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} trading days)")
    
    # Save dataset split locally
    os.makedirs("./data/real_market_checkpoints", exist_ok=True)
    train_df.to_csv("./data/real_market_checkpoints/train_prices.csv")
    test_df.to_csv("./data/real_market_checkpoints/test_prices.csv")
    
    def make_env():
        return RealMarketEnv(price_df=train_df, initial_cash=10000.0, history_len=10)
        
    vec_env = DummyVecEnv([make_env])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./data/real_market_checkpoints/",
        name_prefix="rai_real_ppo"
    )
    
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",
        tensorboard_log="./data/real_market_checkpoints/tb_log/"
    )
    
    total_timesteps = 100_000
    print(f"Starting training on CPU for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
    
    model.save("./data/real_market_checkpoints/rai_real_ppo_final")
    print("Training complete! Model saved to ./data/real_market_checkpoints/rai_real_ppo_final")

if __name__ == "__main__":
    main()
