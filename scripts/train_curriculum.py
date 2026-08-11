import os
import json
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sb3_contrib import RecurrentPPO
from rai.world.env import RAIWorldEnv, STAGE_CONFIGS

class CurriculumCallback(BaseCallback):
    def __init__(self, log_path="./data/v0.2_rl_checkpoints/curriculum_history.jsonl", verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.log_file = None
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_actions = 0
        
    def _on_training_start(self) -> None:
        self.log_file = open(self.log_path, "a")
        
    def _on_step(self) -> bool:
        env0 = self.training_env.envs[0]
        unwrapped = env0.unwrapped if hasattr(env0, 'unwrapped') else env0
        
        act_type = unwrapped.last_action_type
        if act_type in self.action_counts:
            self.action_counts[act_type] += 1
        self.total_actions += 1
        
        if unwrapped.world is not None and unwrapped.world.current_step > 0 and unwrapped.world.current_step % 1000 == 0:
            prices = unwrapped.world.get_prices()
            agent = unwrapped.world.agents[0]
            
            stats = {
                "step": self.num_timesteps,
                "stage": unwrapped.stage,
                "rai_wealth": float(agent.Q),
                "hold_freq": float(self.action_counts[0] / max(1, self.total_actions)),
                "buy_freq": float(self.action_counts[1] / max(1, self.total_actions)),
                "sell_freq": float(self.action_counts[2] / max(1, self.total_actions)),
                "produce_freq": float(self.action_counts[3] / max(1, self.total_actions))
            }
            self.log_file.write(json.dumps(stats) + "\n")
            self.log_file.flush()
            
            self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            self.total_actions = 0
            
        return True
        
    def _on_training_end(self) -> None:
        if self.log_file:
            self.log_file.close()

def run_curriculum_stage(stage_id, model=None):
    cfg = STAGE_CONFIGS[stage_id]
    print(f"\n================================================================================")
    print(f"  STARTING CURRICULUM STAGE {stage_id}: {cfg}")
    print(f"================================================================================\n")
    
    def make_env():
        return RAIWorldEnv(stage=stage_id, history_len=32)
        
    vec_env = DummyVecEnv([make_env])
    
    os.makedirs("./data/v0.2_rl_checkpoints/", exist_ok=True)
    
    if model is None:
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
    else:
        model.set_env(vec_env)
        
    logger_callback = CurriculumCallback()
    
    # Train each stage for 100,000 steps
    stage_timesteps = 100_000
    model.learn(total_timesteps=stage_timesteps, callback=[logger_callback], reset_num_timesteps=False)
    
    stage_save_path = f"./data/v0.2_rl_checkpoints/rai_curriculum_stage_{stage_id}"
    model.save(stage_save_path)
    print(f"Stage {stage_id} Complete! Saved model checkpoint to {stage_save_path}\n")
    
    return model

def main():
    print("=== Zero AI: 5-Stage Progressive World Curriculum Training ===")
    
    model = None
    for stage_id in range(1, 6):
        model = run_curriculum_stage(stage_id, model=model)
        
    print("\n🎉 ALL 5 CURRICULUM STAGES COMPLETED SUCCESSFULLY!")
    print("Final model saved to ./data/v0.2_rl_checkpoints/rai_curriculum_stage_5")

if __name__ == "__main__":
    main()
