import torch
import numpy as np
from rai.learning.rl_mini_ppo import RAIPolicy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def attack_v04():
    device = torch.device("cpu")
    tensors = torch.load("data/rl_macro_test/epi_tensors.pt")
    states = list(tensors.keys())
    
    print("--- RAI-RL v0.4 ATTACK SUITE ---")
    window_size = 3
    
    top_seeds = torch.load("data/rl_macro_test/v04_top_seeds.pt")
    ensemble = []
    for seed in top_seeds:
        policy = RAIPolicy(window_size=3, hidden_dim=64)
        policy.load_state_dict(torch.load(f"data/rl_macro_test/v04_seeds/rai_policy_seed_{seed:02d}.pt", map_location=device))
        policy.eval()
        ensemble.append(policy)
        
    y_true = []
    y_mom = []
    y_maj = []
    y_med = []
    y_rai = []
    y_rai_scram = []
    y_rai_unrel = []
    
    X_linear = []
    y_linear = []
    
    np.random.seed(42)
    
    for s_idx, state in enumerate(states):
        data = tensors[state]
        num_vars = data.shape[1]
        test_start_idx = data.shape[0] - 500
        
        unrelated_state = states[(s_idx + 1) % len(states)]
        unrelated_data = tensors[unrelated_state]
        
        for target_idx in range(num_vars):
            for t in range(test_start_idx, data.shape[0] - 1):
                actual_diff = data[t+1, target_idx] - data[t, target_idx]
                true_action = 1 if actual_diff > 0 else 0
                
                last_diff = data[t, target_idx] - data[t-1, target_idx]
                mom_action = 1 if last_diff > 0 else 0
                
                context_diffs = []
                for j in range(num_vars):
                    if j != target_idx:
                        context_diffs.append(data[t, j].item() - data[t-1, j].item())
                        
                maj_action = 1 if sum(1 for d in context_diffs if d > 0) > len(context_diffs)/2 else 0
                med_action = 1 if np.median(context_diffs) > 0 else 0
                
                X_linear.append(context_diffs)
                y_linear.append(true_action)
                
                window = data[t - window_size:t].unsqueeze(0).transpose(1, 2).clone()
                window[0, target_idx, :] = 0.0
                is_target = torch.zeros((1, num_vars, 1))
                is_target[0, target_idx, 0] = 1.0
                obs = torch.cat([window, is_target], dim=2)
                
                window_scram = window.clone()
                for j in range(num_vars):
                    if j != target_idx:
                        shift = np.random.randint(10, 100)
                        window_scram[0, j, :] = data[t - shift - window_size:t - shift, j].clone()
                obs_scram = torch.cat([window_scram, is_target], dim=2)
                
                window_unrel = window.clone()
                for j in range(num_vars):
                    if j != target_idx:
                        if t < unrelated_data.shape[0]:
                            window_unrel[0, j, :] = unrelated_data[t - window_size:t, j].clone()
                        else:
                            window_unrel[0, j, :] = unrelated_data[-window_size:, j].clone()
                obs_unrel = torch.cat([window_unrel, is_target], dim=2)
                
                def get_vote(o):
                    votes = []
                    with torch.no_grad():
                        for policy in ensemble:
                            logits, _ = policy(o)
                            votes.append(torch.argmax(logits, dim=1).item())
                    return 1 if sum(votes) > len(votes)/2 else 0
                    
                y_true.append(true_action)
                y_mom.append(mom_action)
                y_maj.append(maj_action)
                y_med.append(med_action)
                y_rai.append(get_vote(obs))
                y_rai_scram.append(get_vote(obs_scram))
                y_rai_unrel.append(get_vote(obs_unrel))

    y_true = np.array(y_true)
    
    X_linear = np.array(X_linear)
    y_linear = np.array(y_linear)
    split = int(len(X_linear) * 0.8)
    clf = LogisticRegression()
    clf.fit(X_linear[:split], y_linear[:split])
    y_lin_pred = clf.predict(X_linear[split:])
    acc_lin = accuracy_score(y_linear[split:], y_lin_pred)
    
    y_true_shuffled = np.random.permutation(y_true)
    
    print("\n=== FINAL v0.4 ATTACK SUITE RESULTS ===")
    print(f"Total Prediction Steps: {len(y_true)}")
    print("-" * 50)
    print(f"Random Baseline: 50.00%")
    print(f"Target-Only Momentum: {np.mean(y_true == np.array(y_mom))*100:.2f}%")
    print(f"Context Majority Direction: {np.mean(y_true == np.array(y_maj))*100:.2f}%")
    print(f"Context Median Direction: {np.mean(y_true == np.array(y_med))*100:.2f}%")
    print(f"Linear Context Model (Test Split): {acc_lin*100:.2f}%")
    print("-" * 50)
    print(f"Frozen Target-Blind RAI v0.4: {np.mean(y_true == np.array(y_rai))*100:.2f}%")
    print("-" * 50)
    print(f"RAI + Time-Scrambled Context: {np.mean(y_true == np.array(y_rai_scram))*100:.2f}%")
    print(f"RAI + Unrelated Context: {np.mean(y_true == np.array(y_rai_unrel))*100:.2f}%")
    print(f"RAI + Shuffled Labels (Neg Control): {np.mean(y_true_shuffled == np.array(y_rai))*100:.2f}%")

if __name__ == "__main__":
    attack_v04()
