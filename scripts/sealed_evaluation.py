import torch
import numpy as np
from rai.learning.rl_mini_ppo import RAIPolicy

def block_bootstrap(y_true, y_pred, block_size=12, num_bootstraps=1000):
    n = len(y_true)
    num_blocks = n // block_size
    accs = []
    
    for _ in range(num_bootstraps):
        indices = np.random.choice(num_blocks, size=num_blocks, replace=True)
        boot_true = []
        boot_pred = []
        for idx in indices:
            start = idx * block_size
            end = start + block_size
            boot_true.extend(y_true[start:end])
            boot_pred.extend(y_pred[start:end])
            
        accs.append(np.mean(np.array(boot_true) == np.array(boot_pred)))
        
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)

def evaluate_sealed():
    device = torch.device("cpu")
    tensors = torch.load("data/rl_macro_test/macro_tensors.pt")
    
    all_countries = list(tensors.keys())
    sealed_countries = all_countries[10:] # The 5 final locked countries
    
    print("--- RAI-RL v0.2 FINAL SEALED EVALUATION ---")
    print(f"LOCKED COUNTRIES: {sealed_countries}")
    print("WARNING: THIS IS THE FINAL ONE-SHOT EVALUATION. NO RETRIES PERMITTED.\n")
    
    window_size = 3
    test_start_idx = 312
    
    # Load Seed 13 (The locked deterministic policy)
    policy = RAIPolicy(window_size=3, hidden_dim=64)
    policy.load_state_dict(torch.load("data/rl_macro_test/seeds/rai_policy_seed_13.pt", map_location=device))
    policy.eval()
    policy.to(device)
    
    total_true = []
    total_mom = []
    total_rai = []
    
    for country in sealed_countries:
        data = tensors[country]
        if data.shape[0] <= test_start_idx: continue
            
        num_vars = data.shape[1]
        c_true, c_mom, c_rai = [], [], []
        
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
                    
                c_true.append(true_action)
                c_mom.append(mom_action)
                c_rai.append(rai_action)
                
        total_true.extend(c_true)
        total_mom.extend(c_mom)
        total_rai.extend(c_rai)
        
        c_acc = np.mean(np.array(c_true) == np.array(c_rai))
        print(f"Evaluating {country}...")
        print(f"  RAI-RL: {c_acc*100:.2f}%")
        
    print("\n=== AGGREGATE RESULTS ACROSS 5 SEALED COUNTRIES ===")
    print(f"Total Prediction Steps: {len(total_true)}")
    
    y_true = np.array(total_true)
    y_mom = np.array(total_mom)
    y_rai = np.array(total_rai)
    
    # Block Bootstrap CIs
    mom_low, mom_high = block_bootstrap(y_true, y_mom)
    rai_low, rai_high = block_bootstrap(y_true, y_rai)
    
    mom_acc = np.mean(y_true == y_mom)
    rai_acc = np.mean(y_true == y_rai)
    
    print(f"Random Baseline: 50.00%")
    print(f"Momentum Heuristic: {mom_acc*100:.2f}% (95% CI: {mom_low*100:.2f}% - {mom_high*100:.2f}%)")
    print(f"Frozen RAI-RL v0.2: {rai_acc*100:.2f}% (95% CI: {rai_low*100:.2f}% - {rai_high*100:.2f}%)")
    
    # Also report the Momentum/RAI agreement on the sealed set
    mom_wrong_mask = (y_mom != y_true)
    rai_mom_wrong_acc = np.mean(y_true[mom_wrong_mask] == y_rai[mom_wrong_mask])
    
    print("\n--- CONDITIONAL MOMENTUM VERIFICATION ---")
    print(f"Momentum Wrong Cases: {mom_wrong_mask.sum()}")
    print(f"RAI-RL Accuracy when Momentum is WRONG: {rai_mom_wrong_acc*100:.2f}%")

if __name__ == "__main__":
    evaluate_sealed()
