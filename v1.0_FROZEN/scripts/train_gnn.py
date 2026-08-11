import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from rai.learning.gnn import RAIGNN
from rai.learning.synthetic_dataset import SyntheticLinkPredictionDataset
import numpy as np
import os

def train_gnn():
    print("Generating Synthetic Graphs (Train: 10,000, Test: 1000)...")
    train_dataset = SyntheticLinkPredictionDataset(root='data/synthetic/train', num_graphs=10000)
    test_dataset = SyntheticLinkPredictionDataset(root='data/synthetic/test', num_graphs=1000)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = RAIGNN(hidden_channels=64, num_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 10
    
    print("Starting Supervised Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in train_loader:
            optimizer.zero_grad()
            
            # Forward
            logits = model(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
            loss = criterion(logits.view(-1), data.candidate_labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_dataset):.4f}")
        
    os.makedirs('results', exist_ok=True)
    torch.save(model.state_dict(), 'results/gnn_v0.2.pt')
    
    # Evaluation
    print("\nEvaluating on 1,000 completely unseen held-out synthetic graphs...")
    
    untrained_model = RAIGNN(hidden_channels=64, num_layers=3)
    
    model.eval()
    untrained_model.eval()
    
    y_true = []
    y_pred_trained = []
    y_pred_untrained = []
    
    with torch.no_grad():
        for data in test_loader:
            logits_t = model(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
            logits_u = untrained_model(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
            
            y_true.extend(data.candidate_labels.numpy())
            y_pred_trained.extend(torch.sigmoid(logits_t).view(-1).numpy())
            y_pred_untrained.extend(torch.sigmoid(logits_u).view(-1).numpy())
            
    # Random Baseline
    y_pred_random = np.random.uniform(0, 1, len(y_true))
    
    def calc(y_t, y_p):
        preds = (np.array(y_p) > 0.5).astype(int)
        f1 = f1_score(y_t, preds, zero_division=0)
        auc = roc_auc_score(y_t, y_p)
        pr = average_precision_score(y_t, y_p)
        return f1, auc, pr
        
    f1_r, auc_r, pr_r = calc(y_true, y_pred_random)
    f1_u, auc_u, pr_u = calc(y_true, y_pred_untrained)
    f1_t, auc_t, pr_t = calc(y_true, y_pred_trained)
    
    print("\n--- SYNTHETIC ZERO-SHOT LINK PREDICTION ---")
    print(f"{'Model':<15} | {'F1 Score':<10} | {'ROC-AUC':<10} | {'PR-AUC':<10}")
    print(f"{'Random':<15} | {f1_r:.4f}     | {auc_r:.4f}     | {pr_r:.4f}")
    print(f"{'Untrained GNN':<15} | {f1_u:.4f}     | {auc_u:.4f}     | {pr_u:.4f}")
    print(f"{'RAI-GNN v0.2':<15} | {f1_t:.4f}     | {auc_t:.4f}     | {pr_t:.4f}")
    
if __name__ == "__main__":
    train_gnn()
