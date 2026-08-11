import torch
import numpy as np
from rai.learning.rl_mini_ppo import RAIPolicy

def evaluate_ablated_relations():
    device = torch.device("cpu")
    tensors = torch.load("data/rl_macro_test/epi_tensors.pt")
    states = list(tensors.keys())
    
    print("--- RAI-RL v0.3: RELATIONAL ABLATION TEST ---")
    print("Destroying cross-variable relationships while preserving 1D time series.")
    
    window_size = 3
    
    top_seeds = torch.load("data/rl_macro_test/v03_top_seeds.pt")
    ensemble = []
    for seed in top_seeds:
        policy = RAIPolicy(window_size=3, hidden_dim=64)
        policy.load_state_dict(torch.load(f"data/rl_macro_test/v03_seeds/rai_policy_seed_{seed:02d}.pt", map_location=device))
        policy.eval()
        policy.to(device)
        ensemble.append(policy)
        
    total_true = []
    total_mom = []
    total_rai = []
    
    for state in states:
        data = tensors[state]
        num_vars = data.shape[1] # 10
        seq_len = data.shape[0] # 1142
        
        # We need a 500-day evaluation period.
        # To destroy relationships, we will extract a random 500-day continuous window for EACH variable independently.
        eval_len = 500
        
        test_start_idx = window_size
        c_true, c_mom, c_rai = [], [], []
        
        for target_idx in range(num_vars):
            # To ensure a perfect control, the TARGET variable must be exactly the last 500 days
            # just like in the original evaluation. We only decorrelate the OTHER 9 variables.
            start_indices = np.random.randint(0, seq_len - eval_len, size=num_vars)
            start_indices[target_idx] = seq_len - eval_len # Fix target to the last 500 days
            
            # Build the decorrelated dataset for this specific target
            decorrelated_data = torch.zeros((eval_len, num_vars))
            for j in range(num_vars):
                decorrelated_data[:, j] = data[start_indices[j] : start_indices[j] + eval_len, j]
                
            for t in range(test_start_idx, eval_len - 1):
                window = decorrelated_data[t - window_size:t].unsqueeze(0).transpose(1, 2)
                is_target = torch.zeros((1, num_vars, 1))
                is_target[0, target_idx, 0] = 1.0
                obs = torch.cat([window, is_target], dim=2)
                
                last_diff = decorrelated_data[t, target_idx] - decorrelated_data[t-1, target_idx]
                mom_action = 1 if last_diff > 0 else 0
                
                actual_diff = decorrelated_data[t+1, target_idx] - decorrelated_data[t, target_idx]
                true_action = 1 if actual_diff > 0 else 0
                
                votes = []
                with torch.no_grad():
                    for policy in ensemble:
                        logits, _ = policy(obs)
                        action = torch.argmax(logits, dim=1).item()
                        votes.append(action)
                        
                rai_action = 1 if sum(votes) > len(votes)/2 else 0
                
                c_true.append(true_action)
                c_mom.append(mom_action)
                c_rai.append(rai_action)
                
        total_true.extend(c_true)
        total_mom.extend(c_mom)
        total_rai.extend(c_rai)
        
    y_true = np.array(total_true)
    y_mom = np.array(total_mom)
    y_rai = np.array(total_rai)
    
    mom_acc = np.mean(y_true == y_mom)
    rai_acc = np.mean(y_true == y_rai)
    
    print(f"\nTotal Prediction Steps: {len(total_true)}")
    print(f"Random Baseline: 50.00%")
    print(f"Momentum Heuristic (Control): {mom_acc*100:.2f}%")
    print(f"Frozen RAI-RL v0.3 (Ablated Relations): {rai_acc*100:.2f}%")
    print(f"\nOriginal RAI-RL v0.3 (Intact Relations): 72.23%")
    
    drop = 72.23 - (rai_acc * 100)
    print(f"Accuracy Drop: -{drop:.2f}%")
    
    if drop > 5.0:
        print(">>> CONCLUSION: MASSIVE DROP. RAI relies heavily on cross-variable relational dynamics!")
    elif drop > 1.0:
        print(">>> CONCLUSION: MODERATE DROP. RAI uses some relations, but heavily relies on 1D temporal patterns.")
    else:
        print(">>> CONCLUSION: NO DROP. RAI is just a sophisticated 1D temporal pattern matcher.")

if __name__ == "__main__":
    evaluate_ablated_relations()
