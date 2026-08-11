"""
RAI v4 FAST: Deep RL Training (Optimized for Speed)
=====================================================
Keeps the v4 fixes (asymmetric reward, regime switching)
but uses a fast MLP (256×128) instead of slow CNN.
Target: ~2 minutes training.
"""
import os, sys, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rai.world.synthetic_v4_env import SyntheticRegimeSwitchEnv


class ActionMonitor(BaseCallback):
    def __init__(self, check_freq=5000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.actions = []
    def _on_step(self):
        if self.locals.get('actions') is not None:
            self.actions.append(self.locals['actions'].copy())
        if self.n_calls % self.check_freq == 0 and len(self.actions) > 50:
            a = np.concatenate(self.actions[-200:], axis=0)
            std = np.mean(np.std(a, axis=0))
            rng = np.mean(np.max(a, axis=0) - np.min(a, axis=0))
            tag = "✅ DIVERSE" if std > 0.05 else "⚠️ LOW" if std > 0.005 else "❌ DEAD"
            print(f"  Step {self.n_calls:>7d} | std={std:.4f} range={rng:.4f} | {tag}", flush=True)
        return True


def main():
    print("=" * 70, flush=True)
    print("  RAI v4 FAST: Regime-Switch + Asymmetric Reward", flush=True)
    print("=" * 70, flush=True)

    env_kwargs = dict(
        num_assets=10, episode_len=504, history_len=16,
        initial_cash=10000.0, transaction_fee=0.001,
        rebalance_threshold=0.03, loss_penalty_mult=3.0,
        drawdown_threshold=0.05, drawdown_penalty=1.0,
    )

    vec_env = DummyVecEnv([lambda: SyntheticRegimeSwitchEnv(**env_kwargs)])

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
        verbose=0, device="cpu", seed=42,
    )

    total_p = sum(p.numel() for p in model.policy.parameters())
    print(f"  Params: {total_p:,} | Arch: 672→256→128→11", flush=True)
    print(f"  Training 150k steps...\n", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=150_000, callback=ActionMonitor(check_freq=10000))
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s ({150000/elapsed:.0f} FPS)", flush=True)

    os.makedirs("./data/v0.4_rl_checkpoints/", exist_ok=True)
    model.save("./data/v0.4_rl_checkpoints/rai_v4_fast")
    print(f"  Saved.", flush=True)

    # ── Quick diagnostic ──
    print(f"\n  POST-TRAINING DIAGNOSTIC:", flush=True)
    test_env = SyntheticRegimeSwitchEnv(**env_kwargs)
    all_actions = []
    for ep in range(3):
        obs, _ = test_env.reset(seed=ep*100)
        done = False; step = 0
        while not done and step < 200:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(action.copy())
            obs, _, done, _, _ = test_env.step(action)
            step += 1

    a = np.array(all_actions)
    within_std = np.mean(np.std(a, axis=0))
    cash_range = np.max(a[:, 0]) - np.min(a[:, 0])
    print(f"  Action std: {within_std:.6f}  Cash logit range: {cash_range:.4f}", flush=True)

    if within_std < 0.001:
        print(f"  ❌ COLLAPSED — actions constant", flush=True)
    elif within_std < 0.01:
        print(f"  ⚠️ BARELY ADAPTS", flush=True)
    else:
        print(f"  ✅ ADAPTIVE — actions vary with conditions", flush=True)

    # Now evaluate on real market data
    import pandas as pd

    test_csv = "./data/real_market_checkpoints/test_prices.csv"
    train_csv = "./data/real_market_checkpoints/train_prices.csv"

    if os.path.exists(test_csv):
        print(f"\n  REAL MARKET EVALUATION:", flush=True)
        test_df = pd.read_csv(test_csv, index_col=0, parse_dates=True)
        train_df = pd.read_csv(train_csv, index_col=0, parse_dates=True)

        for label, df in [("2020-2024", test_df), ("2010-2019", train_df)]:
            from scripts.diagnostic_mlp import DiagnosticEnv
            env = DiagnosticEnv(price_df=df, history_len=16,
                                max_resources=10, rebalance_threshold=0.03)
            # Fix obs space to match v4
            # v4 has different obs structure, so we need a compatible eval env
            # For now, just report that we need the eval env
            print(f"  [{label}] Eval env needs v4-compatible bridge (TODO)", flush=True)

    print(f"\n  ✅ v4 training complete!", flush=True)


if __name__ == "__main__":
    main()
