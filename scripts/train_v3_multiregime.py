"""
Zero-Shot RAI v3: Multi-Regime Synthetic Training Script
=========================================================
- Algorithm: PPO (MlpPolicy for continuous target weights)
- Training: 150,000 timesteps on procedural multi-regime synthetic price worlds (0% real data)
- Expected wall-clock time: ~15-20 minutes
"""
import os
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from rai.world.synthetic_v3_env import SyntheticMultiRegimeEnv


class V3TrainingMonitor(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self):
        if self.num_timesteps % 10000 == 0:
            print(f"  [Step {self.num_timesteps:>7d}] Training in progress...")
        return True


def main():
    print("=" * 75)
    print("  ZERO-SHOT RAI v3: Multi-Regime Continuous Portfolio Allocator")
    print("  0% Real Data | Bull, Bear & Sideways Synthetic Regimes | ~15 min")
    print("=" * 75)
    
    os.makedirs("./data/v0.3_rl_checkpoints/", exist_ok=True)
    
    def make_env():
        return SyntheticMultiRegimeEnv(
            num_assets=20,
            episode_len=504,         # 2 synthetic years
            history_len=32,
            initial_cash=10000.0,
            transaction_fee=0.001,
            rebalance_threshold=0.03 # only trade if weight drift > 3%
        )
        
    vec_env = DummyVecEnv([make_env])
    total_steps = 150_000
    
    print(f"\n  Episode Length: 504 steps (2 synthetic years)")
    print(f"  Observation Dim: {make_env().observation_space.shape}")
    print(f"  Action Space: Continuous Box(21) -> Softmax Portfolio Target Weights")
    print(f"  Total Training Steps: {total_steps:,}")
    print()
    
    model = PPO(
        "MlpPolicy",
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
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        tensorboard_log="./data/v0.3_rl_checkpoints/tb_log_v3/"
    )
    
    print("  Starting v3 multi-regime training...\n")
    model.learn(total_timesteps=total_steps, callback=[V3TrainingMonitor()])
    
    save_path = "./data/v0.3_rl_checkpoints/rai_v3_multiregime"
    model.save(save_path)
    print(f"\n  ✅ Model saved to {save_path}.zip")
    print(f"  Training complete!")

if __name__ == "__main__":
    main()
