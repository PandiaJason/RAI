import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from rai.learning.rl_mini_ppo import RAIPolicy
from scripts.train_ablations import RAIPolicyNoZWorld, RAIPolicyNoAttention, RAIPolicyHistoryMLP

def test_ablations():
    device = torch.device("cpu")
    tensors = torch.load("data/rl_macro_test/macro_tensors.pt")
    
    val_countries = list(tensors.keys())[:10]
    window_size = 3
    test_start_idx = 312
    
    models = {
        "Full RAI v0.2": RAIPolicy(window_size=3, hidden_dim=64),
        "RAI_NoZWorld": RAIPolicyNoZWorld(window_size=3, hidden_dim=64),
        "RAI_NoAttention": RAIPolicyNoAttention(window_size=3, hidden_dim=64),
        "RAI_HistoryMLP": RAIPolicyHistoryMLP(num_vars=10, window_size=3, hidden_dim=128),
        "RAI_ShuffledLabels": RAIPolicy(window_size=3, hidden_dim=64),
        "RAI_Untrained": RAIPolicy(window_size=3, hidden_dim=64)
    }
    
    paths = {
        "Full RAI v0.2": "data/rl_macro_test/rai_policy.pt",
        "RAI_NoZWorld": "data/rl_macro_test/rai_policy_nozworld.pt",
        "RAI_NoAttention": "data/rl_macro_test/rai_policy_noattention.pt",
        "RAI_HistoryMLP": "data/rl_macro_test/rai_policy_historymlp.pt",
        "RAI_ShuffledLabels": "data/rl_macro_test/rai_policy_shuffled.pt",
        "RAI_Untrained": "data/rl_macro_test/rai_policy_untrained.pt"
    }
    
    for name, model in models.items():
        try:
            model.load_state_dict(torch.load(paths[name], map_location=device))
        except Exception as e:
            print(f"Warning: Could not load {name} from {paths[name]}: {e}")
        model.eval()
        model.to(device)
        
    y_true_all = []
    y_mom_all = []
    
    predictions = {name: [] for name in models.keys()}
    
    for country in val_countries:
        data = tensors[country]
        if data.shape[0] <= test_start_idx: continue
            
        num_vars = data.shape[1]
        
        for target_idx in range(num_vars):
            for t in range(test_start_idx, data.shape[0] - 1):
                window = data[t - window_size:t].unsqueeze(0).transpose(1, 2)
                is_target = torch.zeros((1, num_vars, 1))
                is_target[0, target_idx, 0] = 1.0
                obs = torch.cat([window, is_target], dim=2)
                
                last_diff = data[t, target_idx] - data[t-1, target_idx]
                momentum_action = 1 if last_diff > 0 else 0
                
                actual_diff = data[t+1, target_idx] - data[t, target_idx]
                actual_action = 1 if actual_diff > 0 else 0
                
                y_true_all.append(actual_action)
                y_mom_all.append(momentum_action)
                
                with torch.no_grad():
                    for name, model in models.items():
                        if name == "RAI_HistoryMLP":
                            # Pad to 10 vars
                            padded_obs = torch.zeros((1, 10, obs.shape[2]))
                            padded_obs[:, :num_vars, :] = obs
                            logits, _ = model(padded_obs)
                        else:
                            logits, _ = model(obs)
                        action = torch.argmax(logits, dim=1).item()
                        predictions[name].append(action)
                        
    print(f"Total Predictions: {len(y_true_all)}\n")
    print(f"{'Model':<25} | {'Accuracy':<10} | {'Inv Acc':<10}")
    print("-" * 55)
    
    y_true = np.array(y_true_all)
    y_mom = np.array(y_mom_all)
    
    print(f"{'Random Baseline':<25} | 50.00%     | 50.00%")
    mom_acc = accuracy_score(y_true, y_mom)
    print(f"{'Momentum Heuristic':<25} | {mom_acc*100:.2f}%     | {(1-mom_acc)*100:.2f}%")
    
    for name in models.keys():
        acc = accuracy_score(y_true, predictions[name])
        inv_acc = accuracy_score(y_true, 1 - np.array(predictions[name]))
        print(f"{name:<25} | {acc*100:.2f}%     | {inv_acc*100:.2f}%")
        
    print("\n\n--- CONDITIONAL MOMENTUM ANALYSIS ---")
    mom_correct_mask = (y_mom == y_true)
    mom_wrong_mask = (y_mom != y_true)
    
    print(f"Momentum Correct cases: {mom_correct_mask.sum()}")
    print(f"Momentum Wrong cases:   {mom_wrong_mask.sum()}\n")
    
    print(f"{'Model':<20} | {'P(RAI=Mom)':<15} | {'Acc (Mom Correct)':<20} | {'Acc (Mom Wrong)':<20}")
    print("-" * 85)
    
    for name in models.keys():
        preds = np.array(predictions[name])
        
        p_agree = np.mean(preds == y_mom)
        acc_when_mom_correct = accuracy_score(y_true[mom_correct_mask], preds[mom_correct_mask])
        acc_when_mom_wrong = accuracy_score(y_true[mom_wrong_mask], preds[mom_wrong_mask])
        
        print(f"{name:<20} | {p_agree*100:5.2f}%          | {acc_when_mom_correct*100:5.2f}%               | {acc_when_mom_wrong*100:5.2f}%")

if __name__ == "__main__":
    test_ablations()
