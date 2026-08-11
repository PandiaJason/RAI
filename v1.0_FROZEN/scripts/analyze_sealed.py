import torch
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from rai.learning.rl_mini_ppo import RAIPolicy

def block_bootstrap_diff(y_true, y_mom, y_rai, block_size=12, num_bootstraps=1000):
    n = len(y_true)
    num_blocks = n // block_size
    diffs = []
    
    for _ in range(num_bootstraps):
        indices = np.random.choice(num_blocks, size=num_blocks, replace=True)
        boot_true = []
        boot_mom = []
        boot_rai = []
        for idx in indices:
            start = idx * block_size
            end = start + block_size
            boot_true.extend(y_true[start:end])
            boot_mom.extend(y_mom[start:end])
            boot_rai.extend(y_rai[start:end])
            
        acc_mom = np.mean(np.array(boot_true) == np.array(boot_mom))
        acc_rai = np.mean(np.array(boot_true) == np.array(boot_rai))
        diffs.append(acc_rai - acc_mom)
        
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def analyze_sealed():
    device = torch.device("cpu")
    tensors = torch.load("data/rl_macro_test/macro_tensors.pt")
    
    all_countries = list(tensors.keys())
    sealed_countries = all_countries[10:]
    
    window_size = 3
    test_start_idx = 312
    
    policy = RAIPolicy(window_size=3, hidden_dim=64)
    policy.load_state_dict(torch.load("data/rl_macro_test/seeds/rai_policy_seed_13.pt", map_location=device))
    policy.eval()
    
    y_true = []
    y_mom = []
    y_rai = []
    
    for country in sealed_countries:
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
                mom_action = 1 if last_diff > 0 else 0
                
                actual_diff = data[t+1, target_idx] - data[t, target_idx]
                true_action = 1 if actual_diff > 0 else 0
                
                with torch.no_grad():
                    logits, _ = policy(obs)
                    rai_action = torch.argmax(logits, dim=1).item()
                    
                y_true.append(true_action)
                y_mom.append(mom_action)
                y_rai.append(rai_action)
                
    y_true = np.array(y_true)
    y_mom = np.array(y_mom)
    y_rai = np.array(y_rai)
    
    # Contingency Table
    # A = RAI correct, Mom correct
    # B = RAI correct, Mom wrong
    # C = RAI wrong, Mom correct
    # D = RAI wrong, Mom wrong
    
    rai_correct = (y_rai == y_true)
    mom_correct = (y_mom == y_true)
    
    A = np.sum(rai_correct & mom_correct)
    B = np.sum(rai_correct & ~mom_correct)
    C = np.sum(~rai_correct & mom_correct)
    D = np.sum(~rai_correct & ~mom_correct)
    
    print("--- CONTINGENCY TABLE ---")
    print(f"{'':<15} | {'Mom Correct':<12} | {'Mom Wrong':<10}")
    print("-" * 45)
    print(f"{'RAI Correct':<15} | {A:<12} | {B:<10}")
    print(f"{'RAI Wrong':<15} | {C:<12} | {D:<10}")
    print()
    
    # McNemar's Test
    table = [[A, B], [C, D]]
    result = mcnemar(table, exact=False, correction=True)
    print("--- MCNEMAR'S TEST ---")
    print(f"Statistic: {result.statistic:.4f}, p-value: {result.pvalue:.4e}")
    if result.pvalue < 0.05:
        print("Difference is statistically significant (p < 0.05).")
    else:
        print("Difference is NOT statistically significant (p >= 0.05).")
        
    print()
    
    # Bootstrap CI for Delta
    print("--- BLOCK BOOTSTRAP DELTA ACCURACY ---")
    delta = np.mean(rai_correct) - np.mean(mom_correct)
    print(f"Point Estimate Delta: {delta*100:.2f}%")
    
    low, high = block_bootstrap_diff(y_true, y_mom, y_rai)
    print(f"95% CI for Delta: [{low*100:.2f}%, {high*100:.2f}%]")
    
    if low > 0:
        print("Delta CI does NOT contain zero. RAI reliably beats Momentum.")
    else:
        print("Delta CI contains zero. Cannot claim RAI reliably beats Momentum.")

if __name__ == "__main__":
    analyze_sealed()
