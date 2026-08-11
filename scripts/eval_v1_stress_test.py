import torch
from torch.utils.data import DataLoader
from rai.learning.v1_worlds import SyntheticWorldsDataset
from rai.learning.v1_model import RAIV1
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score

def compute_metrics(y_true, y_pred, y_prob):
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return precision, recall, f1, roc_auc, pr_auc, acc

def run_evaluation(model, dataset, device, scramble=False, zero_interventions=False, permute_nodes=False, subgraph_eval=False):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            X = batch['X'].to(device)
            I = batch['I'].to(device)
            G_true = batch['G'].to(device)
            
            B, T, N = X.shape
            
            if subgraph_eval and N > 10:
                indices = torch.randperm(N)[:10].to(device)
                X = X[:, :, indices]
                I = I[:, :, indices]
                G_true = G_true[:, indices][:, :, indices]
                N = 10
            
            if permute_nodes:
                perm = torch.randperm(N).to(device)
                X = X[:, :, perm]
                I = I[:, :, perm]
                G_true = G_true[:, perm][:, :, perm]
            
            if zero_interventions:
                I_input = torch.zeros_like(I)
            else:
                I_input = I
                
            g_logits, preds = model(X, I_input)
            g_prob = torch.sigmoid(g_logits)
                
            g_pred = (g_prob > 0.5).float()
            
            mask = torch.eye(N, device=device).bool().unsqueeze(0).expand(B, -1, -1)
            g_pred_masked = g_pred[~mask].cpu().numpy()
            g_true_masked = G_true[~mask].cpu().numpy()
            g_prob_masked = g_prob[~mask].cpu().numpy()
            
            all_y_true.extend(g_true_masked)
            all_y_pred.extend(g_pred_masked)
            all_y_prob.extend(g_prob_masked)
            
            window_size = 5
            targets = X[:, window_size:, :]
            i_mask = I[:, window_size:, :]
            valid_mask = (i_mask == 0).float()
            
            if scramble:
                g_prob_scram = torch.rand(B, N, N, device=device)
                g_prob_scram = g_prob_scram.masked_fill(mask, 0.0)
                preds_scram = []
                for t in range(window_size - 1, T - 1):
                    x_window = X[:, t - window_size + 1 : t + 1, :]
                    pred_t1 = model.pred_mod(x_window, g_prob_scram)
                    preds_scram.append(pred_t1.unsqueeze(1))
                preds = torch.cat(preds_scram, dim=1)
                
            mse = (torch.pow(preds - targets, 2) * valid_mask).sum()
            total_mse += mse.item()
            total_samples += valid_mask.sum().item()
            
    precision, recall, f1, roc_auc, pr_auc, acc = compute_metrics(all_y_true, all_y_pred, all_y_prob)
    mse_final = total_mse / (total_samples + 1e-8)
    
    return {
        'Precision': precision, 'Recall': recall, 'F1': f1, 
        'ROC-AUC': roc_auc, 'PR-AUC': pr_auc, 'Dir-Acc': acc, 
        'MSE': mse_final
    }

def print_metrics(title, metrics, scrambled_mse=None):
    print(f"\n=== {title} ===")
    print(f"Dir-Edge Acc : {metrics['Dir-Acc']*100:.2f}%")
    print(f"Precision    : {metrics['Precision']:.4f}")
    print(f"Recall       : {metrics['Recall']:.4f}")
    print(f"F1 Score     : {metrics['F1']:.4f}")
    print(f"ROC-AUC      : {metrics['ROC-AUC']:.4f}")
    print(f"PR-AUC       : {metrics['PR-AUC']:.4f}")
    print(f"MSE (Intact) : {metrics['MSE']:.4f}")
    if scrambled_mse is not None:
        print(f"MSE (Scramb) : {scrambled_mse:.4f}")

def main():
    device = torch.device("cpu")
    model = RAIV1(seq_len=50, window_size=5, num_vars=10).to(device)
    model.load_state_dict(torch.load("data/v1/rai_v1_model.pt", map_location=device))
    model.eval()
    
    print("Generating Datasets...")
    base_ds = SyntheticWorldsDataset(num_samples=1000, families=['linear', 'tanh'])
    law_d_ds = SyntheticWorldsDataset(num_samples=1000, families=['threshold'])
    law_e_ds = SyntheticWorldsDataset(num_samples=1000, families=['cycles_regime'])
    
    res_base = run_evaluation(model, base_ds, device)
    res_base_scram = run_evaluation(model, base_ds, device, scramble=True)
    print_metrics("Baseline (Laws A & B)", res_base, res_base_scram['MSE'])
    
    res_perm = run_evaluation(model, base_ds, device, permute_nodes=True)
    print_metrics("Test A: Node-ID Permutation", res_perm)
    
    for N in [15, 20, 30]:
        ds_n = SyntheticWorldsDataset(num_samples=1000, num_vars=N, families=['linear', 'tanh'])
        res_n = run_evaluation(model, ds_n, device, subgraph_eval=True)
        print_metrics(f"Test B: Scale Invariance (N={N}, 10-node Subgraphs)", res_n)
        
    res_law_d = run_evaluation(model, law_d_ds, device)
    res_law_d_scram = run_evaluation(model, law_d_ds, device, scramble=True)
    print_metrics("Test C: Law D (Threshold/Delay)", res_law_d, res_law_d_scram['MSE'])
    
    res_law_e = run_evaluation(model, law_e_ds, device)
    res_law_e_scram = run_evaluation(model, law_e_ds, device, scramble=True)
    print_metrics("Test C: Law E (Cycles + Regime Switching)", res_law_e, res_law_e_scram['MSE'])
    
    res_no_int = run_evaluation(model, base_ds, device, zero_interventions=True)
    print_metrics("Test D: Intervention Ablation (I=0 at test time)", res_no_int)

if __name__ == "__main__":
    main()
