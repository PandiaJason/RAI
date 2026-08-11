import torch
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from rai.learning.rl_mini_ppo import RAIPolicy

def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=1000, ci=95):
    """Computes confidence interval for balanced accuracy using block bootstrap."""
    if len(y_true) < 10:
        return 0.0, (0.0, 0.0)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    bootstrapped_scores = []
    n = len(y_true)
    
    # Block size of 12 (1 year) to account for temporal autocorrelation
    block_size = 12
    if n < block_size:
        block_size = 1
        
    for _ in range(n_bootstraps):
        indices = []
        for _ in range(n // block_size + 1):
            start = np.random.randint(0, n - block_size + 1)
            indices.extend(range(start, start + block_size))
        indices = indices[:n]
        
        score = balanced_accuracy_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    lower = np.percentile(bootstrapped_scores, (100 - ci) / 2.0)
    upper = np.percentile(bootstrapped_scores, 100 - (100 - ci) / 2.0)
    
    return np.mean(bootstrapped_scores), (lower, upper)

def test_on_real_data():
    device = torch.device("cpu")
    print("Loading completely sealed real-world macroeconomic data...")
    tensors = torch.load("data/rl_macro_test/macro_tensors.pt")
    
    countries = list(tensors.keys())
    val_countries = countries[:10]
    
    print(f"\n--- RAI-RL MINI v0.2 VALIDATION EVALUATION ---")
    print(f"Validation countries: {val_countries}")
    print(f"Skipping 5 Sealed Test Countries (Strict Isolation Protocol)")
    
    policy = RAIPolicy(window_size=3, hidden_dim=64).to(device)
    policy.load_state_dict(torch.load("data/rl_macro_test/rai_policy.pt", map_location=device))
    policy.eval()
    
    window_size = 3
    test_start_idx = 312 # 2016 onwards
    
    y_true_all = []
    y_pred_rai_all = []
    y_pred_mom_all = []
    
    for country in val_countries:
        print(f"\nEvaluating {country}...")
        data = tensors[country] # (T, 4)
        if data.shape[0] <= test_start_idx:
            continue
            
        num_vars = data.shape[1]
        
        y_true_country = []
        y_pred_rai_country = []
        y_pred_mom_country = []
        
        for target_idx in range(num_vars):
            for t in range(test_start_idx, data.shape[0] - 1):
                window = data[t - window_size:t].unsqueeze(0)
                window = window.transpose(1, 2) 
                
                is_target = torch.zeros((1, num_vars, 1))
                is_target[0, target_idx, 0] = 1.0
                
                obs = torch.cat([window, is_target], dim=2)
                
                with torch.no_grad():
                    action_logits, _ = policy(obs)
                    action = torch.argmax(action_logits, dim=1).item()
                    
                last_diff = data[t, target_idx] - data[t-1, target_idx]
                momentum_action = 1 if last_diff > 0 else 0
                
                actual_diff = data[t+1, target_idx] - data[t, target_idx]
                actual_action = 1 if actual_diff > 0 else 0
                
                y_true_country.append(actual_action)
                y_pred_rai_country.append(action)
                y_pred_mom_country.append(momentum_action)
                
        y_true_all.extend(y_true_country)
        y_pred_rai_all.extend(y_pred_rai_country)
        y_pred_mom_all.extend(y_pred_mom_country)
        
        # Compute country stats
        _, rai_ci = compute_bootstrap_ci(y_true_country, y_pred_rai_country, n_bootstraps=200)
        _, mom_ci = compute_bootstrap_ci(y_true_country, y_pred_mom_country, n_bootstraps=200)
        
        rai_bal = balanced_accuracy_score(y_true_country, y_pred_rai_country)
        mom_bal = balanced_accuracy_score(y_true_country, y_pred_mom_country)
        
        print(f"  Samples: {len(y_true_country)}")
        print(f"  Momentum: {mom_bal*100:.2f}% (95% CI: {mom_ci[0]*100:.2f}% - {mom_ci[1]*100:.2f}%)")
        print(f"  RAI-RL:   {rai_bal*100:.2f}% (95% CI: {rai_ci[0]*100:.2f}% - {rai_ci[1]*100:.2f}%)")

    print(f"\n=== AGGREGATE RESULTS ACROSS 10 VALIDATION COUNTRIES ===")
    print(f"Total Prediction Steps: {len(y_true_all)}")
    
    rai_mean, rai_ci = compute_bootstrap_ci(y_true_all, y_pred_rai_all, n_bootstraps=1000)
    mom_mean, mom_ci = compute_bootstrap_ci(y_true_all, y_pred_mom_all, n_bootstraps=1000)
    
    print(f"Random Baseline: 50.00%")
    print(f"Momentum Heuristic: {mom_mean*100:.2f}% (95% CI: {mom_ci[0]*100:.2f}% - {mom_ci[1]*100:.2f}%)")
    print(f"Frozen RAI-RL v0.2: {rai_mean*100:.2f}% (95% CI: {rai_ci[0]*100:.2f}% - {rai_ci[1]*100:.2f}%)")

if __name__ == "__main__":
    test_on_real_data()
