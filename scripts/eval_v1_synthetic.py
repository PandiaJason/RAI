import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rai.learning.v1_worlds import SyntheticWorldsDataset
from rai.learning.v1_model import RAIV1
import numpy as np

def evaluate():
    device = torch.device("cpu")
    print("Generating validation dataset (Holdout Family C - Multiplicative)...")
    val_dataset = SyntheticWorldsDataset(num_samples=1000, families=['multiplicative'])
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = RAIV1(seq_len=50, window_size=5, num_vars=10).to(device)
    model.load_state_dict(torch.load("data/v1/rai_v1_model.pt", map_location=device))
    model.eval()
    
    total_g_acc = 0
    total_mse_intact = 0
    total_mse_scram = 0
    
    with torch.no_grad():
        for batch in val_loader:
            X = batch['X'].to(device)
            I = batch['I'].to(device)
            G_true = batch['G'].to(device)
            
            g_logits, preds = model(X, I)
            
            g_pred = (torch.sigmoid(g_logits) > 0.5).float()
            B = X.shape[0]
            mask = torch.eye(10, device=device).bool().unsqueeze(0).expand(B, -1, -1)
            g_pred_masked = g_pred[~mask]
            g_true_masked = G_true[~mask]
            
            acc = (g_pred_masked == g_true_masked).float().mean()
            total_g_acc += acc.item()
            
            window_size = 5
            targets = X[:, window_size:, :]
            i_mask = I[:, window_size:, :]
            valid_mask = (i_mask == 0).float()
            
            mse_intact = (torch.pow(preds - targets, 2) * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            total_mse_intact += mse_intact.item()
            
            B = X.shape[0]
            g_prob_scram = torch.rand(B, 10, 10, device=device)
            g_prob_scram = g_prob_scram.masked_fill(mask, 0.0)
            
            preds_scram = []
            for t in range(window_size - 1, X.shape[1] - 1):
                x_window = X[:, t - window_size + 1 : t + 1, :]
                pred_t1 = model.pred_mod(x_window, g_prob_scram)
                preds_scram.append(pred_t1.unsqueeze(1))
            preds_scram = torch.cat(preds_scram, dim=1)
            
            mse_scram = (torch.pow(preds_scram - targets, 2) * valid_mask).sum() / (valid_mask.sum() + 1e-8)
            total_mse_scram += mse_scram.item()
            
    print("\n--- RAI v1 HARD GATE EVALUATION ---")
    print(f"Holdout Family: Multiplicative (Unseen)")
    print(f"Graph Inference Accuracy: {total_g_acc/len(val_loader)*100:.2f}% (Random=50%)")
    print(f"Prediction MSE (Intact Inferred G): {total_mse_intact/len(val_loader):.4f}")
    print(f"Prediction MSE (Scrambled G): {total_mse_scram/len(val_loader):.4f}")

if __name__ == "__main__":
    evaluate()
