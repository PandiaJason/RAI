import torch
import pandas as pd
import numpy as np
from rai.learning.v1_model import RAIV1
import random

def causal_normalize(data, window=144):
    """
    Normalizes data using strictly trailing window statistics to prevent future leakage.
    data: (T, N)
    """
    T, N = data.shape
    normed = np.zeros_like(data)
    
    for t in range(T):
        start_idx = max(0, t - window)
        context = data[start_idx:t+1, :]
        mean = context.mean(axis=0)
        std = context.std(axis=0) + 1e-8
        normed[t, :] = (data[t, :] - mean) / std
        
    return normed

def extract_sequences(data, seq_len=50, step=10):
    sequences = []
    for i in range(0, data.shape[0] - seq_len, step):
        seq = data[i:i+seq_len, :]
        sequences.append(seq)
    return np.stack(sequences, axis=0)

def main():
    print("Loading UCI Appliances Energy Data...")
    try:
        df = pd.read_csv("data/energydata_complete.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # Exclude date and random variables
    cols_to_drop = ['date', 'rv1', 'rv2']
    features = [c for c in df.columns if c not in cols_to_drop]
    data = df[features].values.astype(np.float32)
    
    print(f"Data shape: {data.shape}")
    
    print("Applying strict trailing-window causal normalization...")
    data_norm = causal_normalize(data, window=1000)
    
    seq_len = 50
    X = extract_sequences(data_norm, seq_len=seq_len, step=10)
    print(f"Extracted {X.shape[0]} sequences of length {seq_len}.")
    
    device = torch.device("cpu")
    model = RAIV1(seq_len=seq_len, window_size=5, num_vars=10).to(device)
    model.load_state_dict(torch.load("data/v1/rai_v1_model.pt", map_location=device))
    model.eval()
    
    num_subgraphs = 10
    total_mse_intact = 0
    total_mse_scram = 0
    total_mse_momentum = 0
    total_samples = 0
    
    print("\n--- RAI v1 REAL WORLD EVALUATION (Anonymized Building Physics) ---")
    
    with torch.no_grad():
        batch_size = 64
        num_batches = int(np.ceil(X.shape[0] / batch_size))
        
        for sg_idx in range(num_subgraphs):
            # Select 10 random anonymous variables
            indices = np.random.choice(data_norm.shape[1], size=10, replace=False)
            X_sub = X[:, :, indices]
            
            subgraph_mse_intact = 0
            subgraph_mse_scram = 0
            subgraph_mse_mom = 0
            subgraph_samples = 0
            
            for b in range(num_batches):
                X_batch = X_sub[b*batch_size : (b+1)*batch_size]
                X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=device)
                B = X_tensor.shape[0]
                
                I_tensor = torch.zeros_like(X_tensor)
                
                g_logits, preds = model(X_tensor, I_tensor)
                g_prob = torch.sigmoid(g_logits)
                
                mask = torch.eye(10, device=device).bool().unsqueeze(0).expand(B, -1, -1)
                g_prob_scram = torch.rand(B, 10, 10, device=device)
                g_prob_scram = g_prob_scram.masked_fill(mask, 0.0)
                
                window_size = 5
                preds_scram = []
                for t in range(window_size - 1, seq_len - 1):
                    x_window = X_tensor[:, t - window_size + 1 : t + 1, :]
                    pred_t1 = model.pred_mod(x_window, g_prob_scram)
                    preds_scram.append(pred_t1.unsqueeze(1))
                preds_scram = torch.cat(preds_scram, dim=1)
                
                targets = X_tensor[:, window_size:, :]
                
                preds_mom = X_tensor[:, window_size-1:-1, :]
                
                mse_i = torch.pow(preds - targets, 2).sum()
                mse_s = torch.pow(preds_scram - targets, 2).sum()
                mse_m = torch.pow(preds_mom - targets, 2).sum()
                
                samples = targets.numel()
                
                subgraph_mse_intact += mse_i.item()
                subgraph_mse_scram += mse_s.item()
                subgraph_mse_mom += mse_m.item()
                subgraph_samples += samples
                
            total_mse_intact += subgraph_mse_intact
            total_mse_scram += subgraph_mse_scram
            total_mse_momentum += subgraph_mse_mom
            total_samples += subgraph_samples
            
            print(f"Subgraph {sg_idx+1:02d}/{num_subgraphs} | Intact MSE: {subgraph_mse_intact/subgraph_samples:.4f} | Scrambled MSE: {subgraph_mse_scram/subgraph_samples:.4f} | Momentum: {subgraph_mse_mom/subgraph_samples:.4f}")
            
    final_mse_intact = total_mse_intact / total_samples
    final_mse_scram = total_mse_scram / total_samples
    final_mse_mom = total_mse_momentum / total_samples
    
    print("\n--- FINAL RESULTS ---")
    print(f"Target-Only Momentum Baseline : {final_mse_mom:.4f}")
    print(f"RAI v1.0 (Scrambled G)        : {final_mse_scram:.4f}")
    print(f"RAI v1.0 (Intact Inferred G)  : {final_mse_intact:.4f}")

if __name__ == "__main__":
    main()
