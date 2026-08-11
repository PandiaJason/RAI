import yaml
import argparse
import torch
from rai.generation.real_world_parser import RealWorldParser
from rai.learning.env import RAIEnv
from rai.learning.actor_critic import SharedActorCritic

def run_real_world_transfer(model_path: str, data_path: str, steps: int = 500, num_agents: int = 100):
    print(f"Running Real-World Zero-Shot Transfer.\nModel: {model_path}\nData: {data_path}")
    
    # Generate completely abstract world from real semantics
    parser = RealWorldParser()
    world = parser.parse_csv(data_path, num_agents=num_agents)
    
    print(f"Parsed World Structure: {len(parser.entity_map)} abstract entities, {parser.next_relation_id} relations.")
    
    # We must match the max dimensions the network was trained on
    # In base.yaml we used max_entities=100, max_relations=500
    env = RAIEnv(world, max_entities=100, max_relations=500)
    
    # Load Frozen Policy
    policy = SharedActorCritic(obs_dim=env.get_obs_dim(), num_actions=env.num_actions, hidden_size=128)
    policy.load_state_dict(torch.load(model_path))
    policy.eval() # Freeze weights
    
    obs, action_masks = env.get_observations()
    
    total_reward = 0.0
    for step in range(steps):
        with torch.no_grad():
            dist, _ = policy(obs, action_mask=action_masks)
            action = dist.probs.argmax(dim=-1)
            
        next_obs, next_masks, rewards = env.step(action)
        total_reward += rewards.mean().item()
        
        obs = next_obs
        action_masks = next_masks
        
    avg_reward = total_reward / steps
    print(f"Transfer Evaluation Finished. Average Reward per step: {avg_reward:.4f}")
    if avg_reward > 0:
        print("SUCCESS: The frozen agents successfully generated positive utility in a previously unseen real-world economic structure!")
    else:
        print("FAILURE: The agents could not adapt to the real-world structure.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to frozen .pt model")
    parser.add_argument("--data", type=str, required=True, help="Path to real world csv")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    
    run_real_world_transfer(args.model, args.data, args.steps)
