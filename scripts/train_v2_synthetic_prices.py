"""
Zero-Shot RAI v2: Train on Synthetic Random Price Worlds
=========================================================
- Trains on procedurally generated GBM price series (0% real data)
- Uses identical observation/action structure as real market eval
- Fast training: ~15 minutes for 500k steps
"""
import os
import json
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
from rai.world.synthetic_price_env import SyntheticPriceWorldEnv


class TrainingMonitor(BaseCallback):
    """Logs action diversity and portfolio performance during training."""
    
    def __init__(self, log_path="./data/v0.2_rl_checkpoints/v2_training_log.jsonl", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.total_actions = 0
        self.episode_returns = []
        
    def _on_training_start(self):
        self.log_file = open(self.log_path, "a")
        
    def _on_step(self):
        # Track action distribution
        actions = self.locals.get("actions", None)
        if actions is not None:
            for a in actions:
                act_type = int(a[0])
                if act_type in self.action_counts:
                    self.action_counts[act_type] += 1
                self.total_actions += 1
        
        # Log every 5000 steps
        if self.num_timesteps % 5000 == 0 and self.total_actions > 0:
            stats = {
                "step": self.num_timesteps,
                "hold_pct": round(100 * self.action_counts[0] / max(1, self.total_actions), 1),
                "buy_pct": round(100 * self.action_counts[1] / max(1, self.total_actions), 1),
                "sell_pct": round(100 * self.action_counts[2] / max(1, self.total_actions), 1),
            }
            self.log_file.write(json.dumps(stats) + "\n")
            self.log_file.flush()
            
            print(f"  [Step {self.num_timesteps:>7d}] "
                  f"Hold: {stats['hold_pct']:5.1f}%  Buy: {stats['buy_pct']:5.1f}%  Sell: {stats['sell_pct']:5.1f}%")
            
            # Reset counts
            self.action_counts = {0: 0, 1: 0, 2: 0}
            self.total_actions = 0
            
        return True
    
    def _on_training_end(self):
        if self.log_file:
            self.log_file.close()


def main():
    print("=" * 70)
    print("  ZERO-SHOT RAI v2: Training on Synthetic Random Price Worlds")
    print("  0% Real Data | Procedural GBM Markets | ~15 min training")
    print("=" * 70)
    
    os.makedirs("./data/v0.2_rl_checkpoints/", exist_ok=True)
    
    def make_env():
        return SyntheticPriceWorldEnv(
            num_assets=20,
            episode_len=252,    # 1 year per episode
            history_len=32,
            initial_cash=10000.0,
            transaction_fee=0.001
        )
    
    vec_env = DummyVecEnv([make_env])
    
    total_steps = 100_000
    print(f"\n  Parallel Environments: 1")
    print(f"  Episode Length: 252 steps (1 synthetic year)")
    print(f"  Observation Dim: {make_env().observation_space.shape}")
    print(f"  Action Space: {make_env().action_space}")
    print(f"  Total Training Steps: {total_steps:,}")
    print()
    
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
        ent_coef=0.05,        # HIGH entropy to encourage action diversity
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        tensorboard_log="./data/v0.2_rl_checkpoints/tb_log_v2/"
    )
    
    monitor = TrainingMonitor()
    
    print("  Starting training...\n")
    model.learn(total_timesteps=total_steps, callback=[monitor])
    
    save_path = "./data/v0.2_rl_checkpoints/rai_v2_synthetic_price"
    model.save(save_path)
    print(f"\n  ✅ Model saved to {save_path}.zip")
    print(f"  Total training complete!")

if __name__ == "__main__":
    main()
