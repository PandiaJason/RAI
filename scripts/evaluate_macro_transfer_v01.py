import torch
import numpy as np
import pandas as pd
from rai.learning.rl_mini_ppo import RAIPolicy

def test_on_real_data():
    device = torch.device("cpu")
    print("Loading completely sealed real-world macroeconomic data...")
    tensors = torch.load("data/rl_macro_test/macro_tensors.pt")
    
    countries = list(tensors.keys())
    val_countries = countries[:10]
    test_countries = countries[10:]
    
    print(f"\nValidation countries: {val_countries}")
    print(f"Test countries (Sealed): {test_countries}")
    
    # Load frozen policy
    # We train on 10 abstract variables, but evaluate on 7 real ones!
    # Because RAIPolicy is a GNN (MultiheadAttention over nodes), it handles dynamic sizes perfectly.
    policy = RAIPolicy(window_size=3, hidden_dim=32).to(device)
    policy.load_state_dict(torch.load("data/rl_macro_test/rai_policy.pt", map_location=device))
    policy.eval()
    
    # The testing loop
    window_size = 3
    # Our normalization left us with 432 months (1990 to 2025). 
    # 1990-01-01 to 2015-12-01 is 312 months (Train/Normalization window)
    # Test window is 2016-01-01 to 2025-12-01 (120 months)
    # The tensors start at 1990.
    test_start_idx = 312
    
    def evaluate_group(country_group, name):
        print(f"\n=== EVALUATING ON {name} ===")
        total_steps = 0
        rai_correct = 0
        momentum_correct = 0
        
        for country in country_group:
            data = tensors[country] # shape: (T, 4)
            if data.shape[0] <= test_start_idx:
                continue
                
            num_vars = data.shape[1]
            
            # Evaluate all possible targets
            for target_idx in range(num_vars):
                for t in range(test_start_idx, data.shape[0] - 1):
                    # Construct window
                    window = data[t - window_size:t].unsqueeze(0) # (1, window_size, 7) -> Wait, data is (T, 7). window is (3, 7). We need (1, 7, 3)
                    
                    window = window.transpose(1, 2) # (1, 7, window_size)
                    
                    # Append target flag
                    is_target = torch.zeros((1, num_vars, 1))
                    is_target[0, target_idx, 0] = 1.0
                    
                    obs = torch.cat([window, is_target], dim=2) # (1, 7, 4)
                    
                    with torch.no_grad():
                        action_logits, _ = policy(obs)
                        action = torch.argmax(action_logits, dim=1).item() # 0 or 1
                        
                    # Momentum heuristic (predict UP if it went UP last step)
                    last_diff = data[t, target_idx] - data[t-1, target_idx]
                    momentum_action = 1 if last_diff > 0 else 0
                    
                    actual_diff = data[t+1, target_idx] - data[t, target_idx]
                    actual_action = 1 if actual_diff > 0 else 0
                    
                    if action == actual_action:
                        rai_correct += 1
                    if momentum_action == actual_action:
                        momentum_correct += 1
                    total_steps += 1
                    
        acc_rai = rai_correct / total_steps if total_steps > 0 else 0
        acc_mom = momentum_correct / total_steps if total_steps > 0 else 0
        
        print(f"Total Steps Evaluated: {total_steps}")
        print(f"Random Baseline Accuracy:   50.00%")
        print(f"Momentum Heuristic Acc:     {acc_mom*100:.2f}%")
        print(f"Frozen RAI-RL Accuracy:     {acc_rai*100:.2f}%")
        
    evaluate_group(val_countries, "VALIDATION COUNTRIES")
    evaluate_group(test_countries, "FINAL SEALED TEST COUNTRIES")
    
if __name__ == "__main__":
    test_on_real_data()
