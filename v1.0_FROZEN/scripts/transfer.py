import yaml
import argparse
import torch
from rai.generation.world_generator import WorldGenerator
from rai.learning.env import RAIEnv
from rai.learning.actor_critic import SharedActorCritic

def run_transfer_experiment(config_path: str, model_path: str, new_seed: int):
    print(f"Running Zero-Shot Synthetic Transfer. Model: {model_path} | Target Seed: {new_seed}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    env_cfg = config['env']
    ppo_cfg = config['ppo']
    
    # Generate entirely unseen world
    generator = WorldGenerator(seed=new_seed)
    world = generator.generate(
        num_agents=env_cfg['agents'],
        num_entities=env_cfg['entities'],
        num_relations=env_cfg['relations'],
        event_filepath="results/transfer_events.jsonl"
    )
    
    env = RAIEnv(world, max_entities=ppo_cfg['max_entities'], max_relations=ppo_cfg['max_relations'])
    
    # Load Frozen Policy
    policy = SharedActorCritic(obs_dim=env.get_obs_dim(), num_actions=env.num_actions, hidden_size=ppo_cfg['hidden_size'])
    policy.load_state_dict(torch.load(model_path))
    policy.eval() # Freeze weights
    
    obs, action_masks = env.get_observations()
    
    total_reward = 0.0
    for step in range(env_cfg['steps']):
        with torch.no_grad():
            dist, _ = policy(obs, action_mask=action_masks)
            # Use deterministic argmax for evaluation
            action = dist.probs.argmax(dim=-1)
            
        next_obs, next_masks, rewards = env.step(action)
        total_reward += rewards.mean().item()
        
        obs = next_obs
        action_masks = next_masks
        
    print(f"Transfer Evaluation Finished. Average Reward per step: {total_reward / env_cfg['steps']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to frozen .pt model")
    parser.add_argument("--seed", type=int, default=9999, help="Unseen world seed")
    args = parser.parse_args()
    
    run_transfer_experiment(args.config, args.model, args.seed)
