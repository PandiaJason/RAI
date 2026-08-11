import torch
import numpy as np
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

def evaluate_cross_domain():
    device = torch.device("cpu")
    tensors = torch.load("data/rl_macro_test/epi_tensors.pt")
    
    # 15 States loaded. We will evaluate on all 15 states as one huge sealed set.
    # We've never looked at this data during development.
    states = list(tensors.keys())
    
    print("--- RAI-RL v0.4 ZERO-SHOT TARGET-BLIND EVALUATION ---")
    print(f"LOCKED DATASET: Epidemiology (JHU COVID-19 Confirmed Cases)")
    print(f"REGIONS: {len(states)} US States (10 counties each)")
    
    window_size = 3
    # Use the last 500 days of the time series
    
    # Load Ensemble Top 5
    top_seeds = torch.load("data/rl_macro_test/v04_top_seeds.pt")
    ensemble = []
    for seed in top_seeds:
        policy = RAIPolicy(window_size=3, hidden_dim=64)
        policy.load_state_dict(torch.load(f"data/rl_macro_test/v04_seeds/rai_policy_seed_{seed:02d}.pt", map_location=device))
        policy.eval()
        policy.to(device)
        ensemble.append(policy)
        
    total_true = []
    total_mom = []
    total_rai = []
    
    for state in states:
        data = tensors[state]
        num_vars = data.shape[1] # Should be 10
        test_start_idx = data.shape[0] - 500
        
        c_true, c_mom, c_rai = [], [], []
        
        for target_idx in range(num_vars):
            for t in range(test_start_idx, data.shape[0] - 1):
                window = data[t - window_size:t].unsqueeze(0).transpose(1, 2).clone()
                # TARGET-BLIND MASKING
                window[0, target_idx, :] = 0.0
                
                is_target = torch.zeros((1, num_vars, 1))
                is_target[0, target_idx, 0] = 1.0
                obs = torch.cat([window, is_target], dim=2)
                
                last_diff = data[t, target_idx] - data[t-1, target_idx]
                mom_action = 1 if last_diff > 0 else 0
                
                actual_diff = data[t+1, target_idx] - data[t, target_idx]
                true_action = 1 if actual_diff > 0 else 0
                
                # Ensemble Majority Vote
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
        
        c_acc = np.mean(np.array(c_true) == np.array(c_rai))
        print(f"Evaluating {state}...")
        print(f"  RAI-RL Ensemble: {c_acc*100:.2f}%")
        
    print("\n=== AGGREGATE RESULTS ON EPIDEMIOLOGY DATASET ===")
    print(f"Total Prediction Steps: {len(total_true)}")
    
    y_true = np.array(total_true)
    y_mom = np.array(total_mom)
    y_rai = np.array(total_rai)
    
    mom_acc = np.mean(y_true == y_mom)
    rai_acc = np.mean(y_true == y_rai)
    
    print(f"Random Baseline: 50.00%")
    print(f"Momentum Heuristic: {mom_acc*100:.2f}%")
    print(f"Frozen RAI-RL v0.4 (Target-Blind Ensemble): {rai_acc*100:.2f}%")
    
    print("\n--- BLOCK BOOTSTRAP DELTA (RAI - Momentum) ---")
    delta = rai_acc - mom_acc
    print(f"Point Estimate Delta: {delta*100:.2f}%")
    low, high = block_bootstrap_diff(y_true, y_mom, y_rai)
    print(f"95% CI for Delta: [{low*100:.2f}%, {high*100:.2f}%]")
    
    if low > 0:
        print(">>> SUCCESS: RAI-RL significantly outperformed Momentum on a completely unseen non-economic dataset!")
    else:
        print(">>> STATISTICAL TIE or LOSS: Cannot claim RAI significantly beats momentum.")
        
    # Contingency Table
    rai_correct = (y_rai == y_true)
    mom_correct = (y_mom == y_true)
    A = np.sum(rai_correct & mom_correct)
    B = np.sum(rai_correct & ~mom_correct)
    C = np.sum(~rai_correct & mom_correct)
    D = np.sum(~rai_correct & ~mom_correct)
    
    print("\n--- CONTINGENCY TABLE ---")
    print(f"{'':<15} | {'Mom Correct':<12} | {'Mom Wrong':<10}")
    print("-" * 45)
    print(f"{'RAI Correct':<15} | {A:<12} | {B:<10}")
    print(f"{'RAI Wrong':<15} | {C:<12} | {D:<10}")

if __name__ == "__main__":
    evaluate_cross_domain()
