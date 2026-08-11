import os
import torch
import numpy as np
import json
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sb3_contrib import RecurrentPPO
from rai.world.env import RAIWorldEnv

class CivilizationLoggerCallback(BaseCallback):
    def __init__(self, log_path="./data/v0.2_rl_checkpoints/civilization_history.jsonl", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.rai_lifespan = 0
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_actions = 0
        
    def _on_training_start(self) -> None:
        self.log_file = open(self.log_path, "a") # Append mode for staged training
        
    def _on_step(self) -> bool:
        env0 = self.training_env.envs[0]
        world = env0.unwrapped.world if hasattr(env0, 'unwrapped') else env0.world
        
        # Track RAI lifespan
        if world.agents[0].bankrupt:
            self.rai_lifespan = 0
        else:
            self.rai_lifespan += 1
            
        # Track action frequencies
        act_type = env0.env.env.last_action_type if hasattr(env0, 'env') and hasattr(env0.env, 'env') else env0.unwrapped.last_action_type
        if act_type in self.action_counts:
            self.action_counts[act_type] += 1
        self.total_actions += 1
            
        # Log annually
        if world.current_step > 0 and world.current_step % 1000 == 0:
            prices = world.get_prices()
            agent_wealths = [a.Q + np.sum(a.X * prices) for a in world.agents if not a.bankrupt]
            population = len(agent_wealths)
            
            # Compute wealth inequality (Gini)
            if population > 1:
                sorted_w = np.sort(agent_wealths)
                index = np.arange(1, population + 1)
                gini = (2 * np.sum(index * sorted_w)) / (population * np.sum(sorted_w)) - (population + 1) / population
            else:
                gini = 0.0
                
            stats = {
                "step": self.num_timesteps,
                "year": world.current_step // 1000,
                "population": population,
                "total_wealth": float(np.sum(agent_wealths) + np.sum(world.amm_Q)),
                "inequality_gini": float(gini),
                "avg_prices": float(np.mean(prices)),
                "max_price": float(np.max(prices)),
                "rai_wealth": float(world.agents[0].Q),
                "rai_lifespan": self.rai_lifespan,
                "rai_bankrupt": bool(world.agents[0].bankrupt),
                "hold_freq": self.action_counts[0] / max(1, self.total_actions),
                "buy_freq": self.action_counts[1] / max(1, self.total_actions),
                "sell_freq": self.action_counts[2] / max(1, self.total_actions),
                "produce_freq": self.action_counts[3] / max(1, self.total_actions)
            }
            self.log_file.write(json.dumps(stats) + "\n")
            self.log_file.flush()
            
            self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            self.total_actions = 0
            
        return True
        
    def _on_training_end(self) -> None:
        if self.log_file:
            self.log_file.close()

def main():
    print("=== XEconomics Staged Milestone Training (200,000 Step Block) ===")
    
    def make_env():
        return RAIWorldEnv(num_agents=50, num_resources=20, history_len=32)
        
    vec_env = DummyVecEnv([make_env])
    
    os.makedirs("./data/v0.2_rl_checkpoints/", exist_ok=True)
    latest_model_path = "./data/v0.2_rl_checkpoints/rai_world_ppo_latest.zip"
    
    if os.path.exists(latest_model_path):
        print(f"Resuming training from existing checkpoint: {latest_model_path}")
        model = RecurrentPPO.load(latest_model_path, env=vec_env, device="cpu")
    else:
        print("Initializing brand new Zero-Shot RAI model...")
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
            tensorboard_log="./data/v0.2_rl_checkpoints/tb_log/"
        )
        
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./data/v0.2_rl_checkpoints/",
        name_prefix="rai_world_ppo"
    )
    
    logger_callback = CivilizationLoggerCallback(log_path="./data/v0.2_rl_checkpoints/civilization_history.jsonl")
    
    # Run in 200,000 step blocks (~25 minutes / ~200 virtual years)
    stage_steps = 200_000 
    print(f"Executing Stage Block: {stage_steps} timesteps (~200 virtual years)...")
    
    model.learn(total_timesteps=stage_steps, callback=[checkpoint_callback, logger_callback], reset_num_timesteps=False)
    
    # Save checkpoint after stage completion
    model.save("./data/v0.2_rl_checkpoints/rai_world_ppo_latest")
    model.save(f"./data/v0.2_rl_checkpoints/rai_world_ppo_stage_{model.num_timesteps}")
    print(f"Stage complete! Total steps trained so far: {model.num_timesteps}. Model saved to ./data/v0.2_rl_checkpoints/rai_world_ppo_latest")

if __name__ == "__main__":
    main()
