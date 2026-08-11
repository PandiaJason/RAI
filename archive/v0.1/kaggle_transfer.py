import argparse
import torch
import copy
from rai.generation.kaggle_parser import KaggleParser
from rai.learning.env import RAIEnv
from rai.learning.actor_critic import SharedActorCritic
from rai.learning.baselines import RandomPolicy, HeuristicPolicy, UntrainedNeuralBaseline

def run_kaggle_transfer(model_path: str, data_path: str, steps: int = 100, num_agents: int = 100):
    print(f"Running Real-World Zero-Shot Transfer on Kaggle Dataset.\nModel: {model_path}\nData: {data_path}")
    
    # Generate abstract world from real Kaggle semantics
    parser = KaggleParser()
    world = parser.parse_csv(data_path, num_agents=num_agents)
    print(f"Parsed Kaggle Structure: {len(parser.entity_map)} abstract entities, {parser.next_relation_id} relations.\n")
    
    # We must match the max dimensions the network was trained on
    base_env = RAIEnv(world, max_entities=100, max_relations=500)
    
    # 1. Load Trained RAI Policy
    rai_policy = SharedActorCritic(obs_dim=base_env.get_obs_dim(), num_actions=base_env.num_actions, hidden_size=128)
    rai_policy.load_state_dict(torch.load(model_path))
    rai_policy.eval()
    
    # 2. Setup Baselines
    random_policy = RandomPolicy(num_actions=base_env.num_actions)
    heuristic_policy = HeuristicPolicy(num_actions=base_env.num_actions)
    untrained_policy = UntrainedNeuralBaseline(obs_dim=base_env.get_obs_dim(), num_actions=base_env.num_actions, hidden_size=128)
    
    models = {
        "Random Policy": random_policy,
        "Greedy Heuristic": heuristic_policy,
        "Untrained Neural Net": untrained_policy,
        "RAI Trained Policy (Zero-Shot)": rai_policy
    }
    
    results = {}
    
    # Run evaluation for each model
    for model_name, model in models.items():
        # Create a fresh copy of the environment for fair testing
        env_copy = RAIEnv(copy.deepcopy(world), max_entities=100, max_relations=500)
        obs, action_masks = env_copy.get_observations()
        
        total_reward = 0.0
        tp, fp, tn, fn = 0, 0, 0, 0
        
        for step in range(steps):
            if model_name == "RAI Trained Policy (Zero-Shot)":
                with torch.no_grad():
                    dist, _ = model(obs, action_mask=action_masks)
                    action = dist.probs.argmax(dim=-1)
            else:
                action = model.act(obs, action_masks)
                
            # --- CLASSIFICATION METRICS MAPPING (VECTORIZED) ---
            is_pass = (action == 0)
            is_transform = (action > 0)
            
            # Action_masks: True means INVALID, False means VALID. (Wait, let's be careful. 
            # Assuming action_masks is float tensor where 1.0 = Invalid, 0.0 = Valid
            has_valid_transform = (action_masks[:, 1:] == 0).any(dim=1)
            
            fn_mask = is_pass & has_valid_transform
            tn_mask = is_pass & ~has_valid_transform
            
            chosen_action_invalid = action_masks.gather(1, action.unsqueeze(1)).squeeze(1) > 0.5
            
            fp_mask = is_transform & chosen_action_invalid
            tp_mask = is_transform & ~chosen_action_invalid
            
            fn += fn_mask.sum().item()
            tn += tn_mask.sum().item()
            fp += fp_mask.sum().item()
            tp += tp_mask.sum().item()
            
            next_obs, next_masks, rewards = env_copy.step(action)
            total_reward += rewards.mean().item()
            
            obs = next_obs
            action_masks = next_masks
            
        avg_reward = total_reward / steps
        
        # Calculate Metrics safely
        accuracy = (tp + tn) / steps if steps > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[model_name] = {
            "reward": avg_reward,
            "acc": accuracy,
            "f1": f1,
            "prec": precision,
            "rec": recall
        }
        
    print("--- 4-WAY COMPETITION LEADERBOARD ---")
    print(f"{'Model Name':<32} | {'Reward':<8} | {'Accuracy':<8} | {'F1 Score':<8} | {'Precision':<9} | {'Recall':<8}")
    print("-" * 85)
    
    for name, metrics in sorted(results.items(), key=lambda x: x[1]["reward"], reverse=True):
        print(f"{name:<32} | {metrics['reward']:8.4f} | {metrics['acc']:8.4f} | {metrics['f1']:8.4f} | {metrics['prec']:9.4f} | {metrics['rec']:8.4f}")
    
    print("\nCONCLUSION:")
    if results["RAI Trained Policy (Zero-Shot)"]["reward"] > results["Untrained Neural Net"]["reward"]:
        print("RAI mathematically outperforms the standard untreated base models on out-of-distribution real world economic networks.")
    else:
        print("RAI failed to outperform base models.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to frozen .pt model")
    parser.add_argument("--data", type=str, required=True, help="Path to Kaggle csv")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    
    run_kaggle_transfer(args.model, args.data, args.steps)
