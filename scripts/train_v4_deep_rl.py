"""
RAI v4: Deep RL Training
=========================
Custom DNN architecture:
  - 1D CNN temporal encoder (extracts patterns from price history)
  - Deep MLP policy/value heads (256 → 128)
  - PPO with asymmetric reward + regime-switching episodes

NOT using SB3's default MlpPolicy. Custom deep feature extractor.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rai.world.synthetic_v4_env import SyntheticRegimeSwitchEnv


# ═══════════════════════════════════════════════════════════════════
#  CUSTOM DEEP NEURAL NETWORK FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

class CNNTemporalExtractor(BaseFeaturesExtractor):
    """
    Deep feature extractor that processes temporal observations.
    
    Architecture:
      Input: (batch, history_len * single_obs_dim) flat vector
        ↓ reshape to (batch, features_per_step, history_len)
      Conv1D block 1: features → 64 channels, kernel=3
      Conv1D block 2: 64 → 128 channels, kernel=3
      Conv1D block 3: 128 → 64 channels, kernel=3
      Global Average Pool: (batch, 64, T) → (batch, 64)
      FC: 64 → 256 → features_dim(128)
    
    This CNN learns temporal patterns (trends, reversals, volatility shifts)
    directly from the observation sequence. NOT a simple flatten+linear.
    """
    def __init__(self, observation_space, features_dim=128,
                 history_len=16, single_obs_dim=41):
        super().__init__(observation_space, features_dim)
        self.history_len = history_len
        self.single_obs_dim = single_obs_dim
        
        # 1D CNN over temporal dimension
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(single_obs_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            
            nn.AdaptiveAvgPool1d(1),  # Pool across time → (batch, 64, 1)
        )
        
        # Deep FC to produce final features
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.LeakyReLU(0.1),
        )
        
        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        print(f"  CNN Feature Extractor: {total:,} parameters")
    
    def forward(self, observations):
        batch_size = observations.shape[0]
        # Reshape flat observation → (batch, timesteps, features_per_step)
        x = observations.reshape(batch_size, self.history_len, self.single_obs_dim)
        # → (batch, features_per_step, timesteps) for Conv1d
        x = x.permute(0, 2, 1)
        # CNN extracts temporal patterns
        x = self.temporal_cnn(x)  # → (batch, 64, 1)
        x = x.squeeze(-1)         # → (batch, 64)
        # Deep FC
        x = self.fc(x)            # → (batch, features_dim)
        return x


# ═══════════════════════════════════════════════════════════════════
#  TRAINING CALLBACK: Monitor Action Diversity
# ═══════════════════════════════════════════════════════════════════

class ActionDiversityCallback(BaseCallback):
    """
    Monitor action standard deviation during training.
    If actions collapse to constant, warn immediately.
    """
    def __init__(self, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.action_history = []
    
    def _on_step(self):
        # Collect actions
        if self.locals.get('actions') is not None:
            self.action_history.append(self.locals['actions'].copy())
        
        if self.n_calls % self.check_freq == 0 and len(self.action_history) > 100:
            all_actions = np.concatenate(self.action_history[-500:], axis=0)
            action_std = np.mean(np.std(all_actions, axis=0))
            action_range = np.mean(np.max(all_actions, axis=0) - np.min(all_actions, axis=0))
            
            # Check per-dimension std
            per_dim_std = np.std(all_actions, axis=0)
            cash_std = per_dim_std[0]
            asset_std_mean = np.mean(per_dim_std[1:])
            
            status = "✅ DIVERSE" if action_std > 0.01 else "⚠️ COLLAPSING" if action_std > 0.001 else "❌ COLLAPSED"
            
            if self.verbose:
                print(f"  Step {self.n_calls:>7d} | Action std: {action_std:.6f} | "
                      f"Range: {action_range:.4f} | Cash std: {cash_std:.6f} | "
                      f"Asset std: {asset_std_mean:.6f} | {status}")
        
        return True


# ═══════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  RAI v4: Deep RL Training")
    print("  Architecture: CNN Temporal Encoder + Deep PPO")
    print("  Environment: Regime-Switching with Asymmetric Reward")
    print("=" * 80)
    
    NUM_ASSETS = 10
    HISTORY_LEN = 16
    EPISODE_LEN = 504
    TOTAL_TIMESTEPS = 300_000
    
    # Environment config
    env_kwargs = dict(
        num_assets=NUM_ASSETS,
        episode_len=EPISODE_LEN,
        history_len=HISTORY_LEN,
        initial_cash=10000.0,
        transaction_fee=0.001,
        rebalance_threshold=0.03,
        loss_penalty_mult=3.0,
        drawdown_threshold=0.05,
        drawdown_penalty=1.0,
    )
    
    # Calculate observation dimensions
    single_obs_dim = 1 + NUM_ASSETS * 4 + 1  # cash_w + assets + returns + trend + vol + drawdown
    print(f"\n  Observation: {HISTORY_LEN} timesteps × {single_obs_dim} features = {HISTORY_LEN * single_obs_dim} dims")
    print(f"  Action: {NUM_ASSETS + 1} continuous (cash + {NUM_ASSETS} assets)")
    
    # Create vectorized environments
    def make_env(seed):
        def _init():
            env = SyntheticRegimeSwitchEnv(**env_kwargs)
            env.reset(seed=seed)
            return env
        return _init
    
    n_envs = 1
    vec_env = DummyVecEnv([make_env(0)])
    
    # Custom policy kwargs with CNN feature extractor
    policy_kwargs = dict(
        features_extractor_class=CNNTemporalExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            history_len=HISTORY_LEN,
            single_obs_dim=single_obs_dim,
        ),
        net_arch=dict(
            pi=[256, 128],   # Deep policy network
            vf=[256, 128],   # Deep value network
        ),
        activation_fn=nn.LeakyReLU,
    )
    
    print(f"\n  Policy Architecture:")
    print(f"    Feature Extractor: CNN Temporal (3-layer Conv1D + FC)")
    print(f"    Policy Head: 128 → 256 → 128 → {NUM_ASSETS + 1}")
    print(f"    Value Head:  128 → 256 → 128 → 1")
    print(f"    Activation: LeakyReLU")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",  # Will use custom feature extractor
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,       # Higher entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="cpu",
        seed=42,
    )
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\n  Total Parameters:     {total_params:>10,}")
    print(f"  Trainable Parameters: {trainable_params:>10,}")
    
    # Print full architecture
    print(f"\n  Full Architecture:")
    print(f"  {model.policy}")
    
    # Train with action diversity monitoring
    print(f"\n{'='*80}")
    print(f"  Training: {TOTAL_TIMESTEPS:,} steps on {n_envs} parallel envs")
    print(f"{'='*80}\n")
    
    callback = ActionDiversityCallback(check_freq=10000)
    
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    elapsed = time.time() - t0
    
    fps = TOTAL_TIMESTEPS / elapsed
    print(f"\n  Training complete in {elapsed:.0f}s ({fps:.0f} FPS)")
    
    # Save model
    save_dir = "./data/v0.4_rl_checkpoints/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "rai_v4_deep_rl")
    model.save(save_path)
    print(f"  Saved to: {save_path}")
    
    # ═══════════════════════════════════════════════
    #  QUICK DIAGNOSTIC: Is the policy diverse?
    # ═══════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  POST-TRAINING DIAGNOSTIC")
    print(f"{'='*80}")
    
    # Generate a few different observations and check if actions differ
    test_env = SyntheticRegimeSwitchEnv(**env_kwargs)
    
    actions_collected = []
    for episode in range(5):
        obs, _ = test_env.reset(seed=episode * 100)
        episode_actions = []
        done = False
        step = 0
        while not done and step < 200:
            action, _ = model.predict(obs, deterministic=True)
            episode_actions.append(action.copy())
            obs, _, done, _, _ = test_env.step(action)
            step += 1
        actions_collected.append(np.array(episode_actions))
    
    # Check within-episode diversity
    print(f"\n  Within-Episode Action Diversity (does it change during an episode?):")
    for i, ep_actions in enumerate(actions_collected):
        action_std = np.mean(np.std(ep_actions, axis=0))
        action_range = np.mean(np.max(ep_actions, axis=0) - np.min(ep_actions, axis=0))
        cash_range = np.max(ep_actions[:, 0]) - np.min(ep_actions[:, 0])
        print(f"    Episode {i}: action_std={action_std:.6f}  range={action_range:.4f}  cash_logit_range={cash_range:.4f}")
    
    # Check across-episode diversity
    all_first_actions = np.array([ep[0] for ep in actions_collected])
    across_std = np.mean(np.std(all_first_actions, axis=0))
    print(f"\n  Across-Episode Diversity (different obs → different actions?):")
    print(f"    Std of first action across 5 episodes: {across_std:.6f}")
    
    # Verdict
    within_std = np.mean([np.mean(np.std(ea, axis=0)) for ea in actions_collected])
    if within_std < 0.001 and across_std < 0.001:
        print(f"\n  ❌ POLICY COLLAPSED: Actions are constant. Model failed to learn.")
    elif within_std < 0.01:
        print(f"\n  ⚠️ POLICY BARELY ADAPTS: Very small action changes. Needs more training.")
    else:
        print(f"\n  ✅ POLICY IS ADAPTIVE: Actions change across time and episodes.")
    
    vec_env.close()
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
