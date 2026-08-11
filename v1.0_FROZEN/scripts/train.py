import yaml
import argparse
import os
import torch
from rai.generation.world_generator import WorldGenerator
from rai.learning.env import RAIEnv
from rai.learning.actor_critic import SharedActorCritic
from rai.learning.ppo import PPOUpdate
from rai.emergence.specialization import calculate_specialization_entropy
from rai.emergence.exchange_network import build_exchange_network, calculate_network_centrality

def train(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    env_cfg = config['env']
    ppo_cfg = config['ppo']
    os.makedirs(config['logging']['results_dir'], exist_ok=True)
    
    print(f"Starting Training across {env_cfg['seeds']} seeds...")
    
    for seed in range(env_cfg['seeds']):
        print(f"\n--- Seed {seed} ---")
        log_file = f"{config['logging']['results_dir']}/events_seed_{seed}.jsonl"
        
        # 1. Generate World
        generator = WorldGenerator(seed=seed)
        world = generator.generate(
            num_agents=env_cfg['agents'],
            num_entities=env_cfg['entities'],
            num_relations=env_cfg['relations'],
            event_filepath=log_file
        )
        
        # 2. Setup Env and Agent
        env = RAIEnv(world, max_entities=ppo_cfg['max_entities'], max_relations=ppo_cfg['max_relations'])
        
        policy = SharedActorCritic(obs_dim=env.get_obs_dim(), num_actions=env.num_actions, hidden_size=ppo_cfg['hidden_size'])
        ppo = PPOUpdate(policy, lr=ppo_cfg['lr'], gamma=ppo_cfg['gamma'], clip_param=ppo_cfg['clip_param'], ppo_epochs=ppo_cfg['ppo_epochs'])
        
        # 3. Training Loop
        obs, action_masks = env.get_observations()
        
        total_rewards = []
        for step in range(env_cfg['steps']):
            # Act
            with torch.no_grad():
                dist, value = policy(obs, action_mask=action_masks)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
            # Step env
            next_obs, next_masks, rewards = env.step(action)
            
            total_rewards.append(rewards.mean().item())
            
            # PPO Update (simplified batched update over the agents as a single trajectory for this prototype)
            with torch.no_grad():
                _, next_value = policy(next_obs, action_mask=next_masks)
                
            returns = rewards + ppo_cfg['gamma'] * next_value.squeeze(-1)
            advantages = returns - value.squeeze(-1)
            
            rollouts = {
                'obs': obs,
                'actions': action,
                'log_probs_old': log_prob,
                'returns': returns,
                'advantages': advantages,
                'action_masks': action_masks
            }
            
            loss = ppo.update(rollouts)
            
            obs = next_obs
            action_masks = next_masks
            
            if (step + 1) % max(1, (env_cfg['steps'] // 5)) == 0:
                print(f"Step {step+1}/{env_cfg['steps']} | Loss: {loss:.4f} | Avg Reward: {rewards.mean().item():.4f}")
                
        # 4. Emergence Analytics
        print("Calculating Emergence Metrics...")
        spec_ent = calculate_specialization_entropy(log_file)
        G = build_exchange_network(log_file)
        cent = calculate_network_centrality(G)
        
        print(f"Specialization (Entropy): {spec_ent:.4f}")
        print(f"Max Exchange Centrality: {cent:.4f}")
        
        # Save model
        torch.save(policy.state_dict(), f"{config['logging']['results_dir']}/rai_seed_{seed}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    train(args.config)
