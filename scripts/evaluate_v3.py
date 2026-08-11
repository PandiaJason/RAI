import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from rai.learning.gnn import RAIGNN
from rai.learning.synthetic_dataset import SyntheticLinkPredictionDataset
import numpy as np
import networkx as nx
import os

def heuristic_jaccard(data):
    # Reconstruct undirected X-X graph from PyG data
    # edge_index_X_to_R: (src, rel)
    # edge_index_R_to_X: (rel, dst)
    x_to_r = data['entity', 'inputs', 'relation'].edge_index.t().numpy()
    r_to_x = data['relation', 'outputs', 'entity'].edge_index.t().numpy()
    
    r_to_src = {r: src for src, r in x_to_r}
    edges = []
    for r, dst in r_to_x:
        if r in r_to_src:
            edges.append((r_to_src[r], dst))
            
    G = nx.Graph()
    G.add_nodes_from(range(data['entity'].x.shape[0]))
    G.add_edges_from(edges)
    
    preds = []
    for src, dst in zip(data.candidate_src.numpy(), data.candidate_dst.numpy()):
        if src == dst:
            preds.append(0.0)
            continue
        try:
            p = list(nx.jaccard_coefficient(G, [(src, dst)]))[0][2]
        except Exception:
            p = 0.0
        preds.append(p)
    return preds

def run_evaluation():
    print("--- RAI Experiment 002: Hidden-Law Recovery ---")
    
    # 1. Train on Families A, B, C (Small)
    print("\nGenerating Training Set: Families A,B,C (N=10-30), 5000 graphs...")
    train_dataset = SyntheticLinkPredictionDataset(root='data/v3/train_ABC', families=['A', 'B', 'C'], num_graphs=5000, min_entities=10, max_entities=30)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = RAIGNN(hidden_channels=64, num_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Training RAI-GNN v0.3...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            logits = model(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
            loss = criterion(logits.view(-1), data.candidate_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        print(f"Epoch {epoch+1}/10 | Loss: {total_loss / len(train_dataset):.4f}")
        
    model.eval()
    untrained_model = RAIGNN(hidden_channels=64, num_layers=3)
    untrained_model.eval()
    
    def evaluate_set(dataset_name, dataset):
        print(f"\nEvaluating on {dataset_name} ({len(dataset)} graphs)...")
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        y_true = []
        y_pred_trained = []
        y_pred_untrained = []
        y_pred_jaccard = []
        
        with torch.no_grad():
            for data in loader:
                logits_t = model(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
                logits_u = untrained_model(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
                
                y_true.extend(data.candidate_labels.numpy())
                y_pred_trained.extend(torch.sigmoid(logits_t).view(-1).numpy())
                y_pred_untrained.extend(torch.sigmoid(logits_u).view(-1).numpy())
                
        # Heuristics have to be done per-graph because they use NetworkX
        for i in range(len(dataset)):
            data = dataset[i]
            y_pred_jaccard.extend(heuristic_jaccard(data))
            
        y_pred_random = np.random.uniform(0, 1, len(y_true))
        
        def metrics(y_t, y_p):
            return roc_auc_score(y_t, y_p), average_precision_score(y_t, y_p)
            
        auc_r, pr_r = metrics(y_true, y_pred_random)
        auc_h, pr_h = metrics(y_true, y_pred_jaccard)
        auc_u, pr_u = metrics(y_true, y_pred_untrained)
        auc_t, pr_t = metrics(y_true, y_pred_trained)
        
        print(f"{'Model':<20} | {'ROC-AUC':<10} | {'PR-AUC':<10}")
        print(f"{'Random':<20} | {auc_r:.4f}     | {pr_r:.4f}")
        print(f"{'Jaccard Heuristic':<20} | {auc_h:.4f}     | {pr_h:.4f}")
        print(f"{'Untrained GNN':<20} | {auc_u:.4f}     | {pr_u:.4f}")
        print(f"{'RAI-GNN v0.3':<20} | {auc_t:.4f}     | {pr_t:.4f}")

    # 2. Same-Family, Larger Graphs
    test_abc = SyntheticLinkPredictionDataset(root='data/v3/test_ABC_large', families=['A', 'B', 'C'], num_graphs=500, min_entities=50, max_entities=100)
    evaluate_set("Same-Family Scale Generalization (Families A, B, C | N=50-100)", test_abc)
    
    # 3. Cross-Family Transfer
    test_de = SyntheticLinkPredictionDataset(root='data/v3/test_DE', families=['D', 'E'], num_graphs=500, min_entities=10, max_entities=30)
    evaluate_set("Cross-Family Zero-Shot Transfer (Unseen Families D, E | N=10-30)", test_de)

if __name__ == "__main__":
    run_evaluation()
