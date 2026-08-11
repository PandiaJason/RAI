import torch
import numpy as np
import pandas as pd
from rai.learning.rl_mini_ppo import RAIPolicy
import torch.nn.functional as F

def analyze_ensemble():
    device = torch.device("cpu")
    tensors = torch.load("data/rl_macro_test/macro_tensors.pt")
    
    val_countries = list(tensors.keys())[:10]
    window_size = 3
    test_start_idx = 312
    
    policy = RAIPolicy(window_size=3, hidden_dim=64)
    policy.load_state_dict(torch.load("data/rl_macro_test/rai_policy.pt", map_location=device))
    policy.eval()
    policy.to(device)
    
    results = []
    
    print("Evaluating RAI + Momentum Ensemble on Validation Set...")
    
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
                mom_action = 1 if last_diff > 0 else 0
                
                actual_diff = data[t+1, target_idx] - data[t, target_idx]
                true_action = 1 if actual_diff > 0 else 0
                
                with torch.no_grad():
                    logits, _ = policy(obs)
                    probs = F.softmax(logits, dim=1)
                    rai_prob_up = probs[0, 1].item()
                    rai_prob_down = probs[0, 0].item()
                    
                    rai_action = 1 if rai_prob_up > rai_prob_down else 0
                    rai_conf = max(rai_prob_up, rai_prob_down)
                    
                results.append({
                    "true_action": true_action,
                    "mom_action": mom_action,
                    "rai_action": rai_action,
                    "rai_conf": rai_conf
                })
                
    df = pd.DataFrame(results)
    
    df["mom_correct"] = df["mom_action"] == df["true_action"]
    df["rai_correct"] = df["rai_action"] == df["true_action"]
    
    # 1. Stratify by Action Agreement
    print("\n--- ACTION AGREEMENT ANALYSIS ---")
    def label_agreement(row):
        mom_dir = "UP" if row["mom_action"] == 1 else "DOWN"
        rai_dir = "UP" if row["rai_action"] == 1 else "DOWN"
        return f"Mom {mom_dir} | RAI {rai_dir}"
    
    df["Agreement"] = df.apply(label_agreement, axis=1)
    agg = df.groupby("Agreement").agg(
        Count=("true_action", "count"),
        Mom_Acc=("mom_correct", "mean"),
        RAI_Acc=("rai_correct", "mean")
    )
    for col in ["Mom_Acc", "RAI_Acc"]:
        agg[col] = (agg[col] * 100).map("{:.2f}%".format)
    print(agg.to_string())
    
    # 2. Stratify by RAI Confidence Disagreement
    print("\n--- HIGH CONFIDENCE DISAGREEMENT ---")
    print("When Momentum and RAI disagree, how often is RAI right when it is highly confident?")
    
    disagree_mask = df["mom_action"] != df["rai_action"]
    df_disagree = df[disagree_mask]
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    print(f"{'Threshold':<10} | {'Cases':<6} | {'RAI Win Rate (Accuracy)':<25}")
    print("-" * 50)
    for t in thresholds:
        mask = df_disagree["rai_conf"] >= t
        subset = df_disagree[mask]
        cases = len(subset)
        if cases > 0:
            win_rate = subset["rai_correct"].mean() * 100
            print(f"> {t:<8} | {cases:<6} | {win_rate:.2f}%")
        else:
            print(f"> {t:<8} | 0      | N/A")

if __name__ == "__main__":
    analyze_ensemble()
