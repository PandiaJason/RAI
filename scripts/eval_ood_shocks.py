import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from rai.world.env import RAIWorldEnv

def eval_shock(model_path, shock_type="depletion"):
    print(f"--- Running OOD Shock Test: {shock_type.upper()} ---")
    model = RecurrentPPO.load(model_path)
    
    env = RAIWorldEnv()
    obs, _ = env.reset(seed=42)
    
    survived = 0
    total_reward = 0
    
    # Run for 1000 steps
    for t in range(1000):
        # Trigger shock at t=500
        if t == 500:
            print(f"[{t}] Triggering {shock_type} shock silently...")
            if shock_type == "depletion":
                # Find the most abundant resource and wipe out AMM liquidity
                most_abundant = np.argmax(env.world.amm_X)
                env.world.amm_X[most_abundant] = 1.0 # Near zero
                env.world.amm_Q[most_abundant] = 10000.0 # Price skyrockets
            elif shock_type == "obsolescence":
                # Change all agents' subsistence vectors dramatically
                for a in env.world.agents:
                    a.subsistence = np.zeros(env.num_resources)
                    req_idx = np.random.choice(env.num_resources, size=3, replace=False)
                    a.subsistence[req_idx] = np.random.uniform(0.5, 1.5, size=3)
                    
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
            
    a = env.world.agents[0]
    print(f"Result -> Bankrupt: {a.bankrupt}, Final Wealth (Q): {a.Q:.1f}, Reward: {total_reward:.1f}")

def eval_permutation(model_path):
    print("--- Running Permutation Invariance Test ---")
    model = RecurrentPPO.load(model_path)
    
    # We will test if the agent behaves identically if we permute the observation
    # and un-permute the action.
    # Note: A true permutation test requires modifying the environment itself to scramble IDs.
    
    # Standard run
    env1 = RAIWorldEnv()
    obs1, _ = env1.reset(seed=101)
    
    # This is a placeholder for a deep environment modification
    print("Permutation test requires engine-level ID scrambling. Skipping full test for Gate 1.")

def main():
    model_path = "data/v0.1_rl_checkpoints/rai_world_ppo_final"
    try:
        eval_shock(model_path, "depletion")
        eval_shock(model_path, "obsolescence")
    except Exception as e:
        print("Model not ready or error:", e)

if __name__ == "__main__":
    main()
