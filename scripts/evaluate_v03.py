import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from rai.learning.gnn import RAIGNN
from rai.learning.gnn_v04 import CandidateRelativeRAIGNN, build_candidate_relative_batch
from rai.learning.synthetic_dataset import SyntheticLinkPredictionDataset
import numpy as np
import networkx as nx
import os
import random
import time

def heuristic_jaccard(data):
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

def train_model(model, loader, epochs=3, is_v4=False):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        for i, data in enumerate(loader):
            optimizer.zero_grad()
            if is_v4:
                batch = build_candidate_relative_batch(data)
                logits = model(batch)
                loss = criterion(logits.view(-1), batch.candidate_labels)
            else:
                logits = model(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
                loss = criterion(logits.view(-1), data.candidate_labels)
            loss.backward()
            optimizer.step()
            if i > 0 and i % 50 == 0:
                print(f"      Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")
        print(f"    Epoch {epoch} completed in {time.time()-t0:.1f}s")

def evaluate(model_v3, model_v4, loader):
    model_v3.eval()
    model_v4.eval()
    
    y_true = []
    y_v3 = []
    y_v4 = []
    
    with torch.no_grad():
        for data in loader:
            y_true.extend(data.candidate_labels.numpy())
            
            logits_v3 = model_v3(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
            y_v3.extend(torch.sigmoid(logits_v3).view(-1).numpy())
            
            batch = build_candidate_relative_batch(data)
            logits_v4 = model_v4(batch)
            y_v4.extend(torch.sigmoid(logits_v4).view(-1).numpy())
            
    auc_v3 = roc_auc_score(y_true, y_v3)
    auc_v4 = roc_auc_score(y_true, y_v4)
    
    return auc_v3, auc_v4

def run_progressive_evaluation(num_seeds=30):
    print("--- RAI Side-by-Side Progressive Evaluation ---")
    
    results_v3 = {k: [] for k in ['Same Family', '10x Scale', 'Shift', 'Composition', 'Held-Out D', 'Held-Out E', 'Null']}
    results_v4 = {k: [] for k in results_v3.keys()}
    jaccard_results = {k: [] for k in results_v3.keys()}
    
    print("Generating Datasets...")
    t0 = time.time()
    train_A = SyntheticLinkPredictionDataset('data/v03/train_A', [['A']], 3000, 10, 30)
    test_A = SyntheticLinkPredictionDataset('data/v03/test_A', [['A']], 500, 10, 30)
    test_A_scale = SyntheticLinkPredictionDataset('data/v03/test_A_scale', [['A']], 500, 100, 200)
    test_A_noise = SyntheticLinkPredictionDataset('data/v03/test_A_noise', [['A']], 500, 10, 30, noise_ratio=0.3)
    
    train_ABC = SyntheticLinkPredictionDataset('data/v03/train_ABC', [['A'], ['B'], ['C']], 3000, 10, 30)
    test_comp = SyntheticLinkPredictionDataset('data/v03/test_comp', [['A', 'B'], ['B', 'C']], 500, 10, 30)
    test_D = SyntheticLinkPredictionDataset('data/v03/test_D', [['D']], 500, 10, 30)
    test_E = SyntheticLinkPredictionDataset('data/v03/test_E', [['E']], 500, 10, 30)
    test_Null = SyntheticLinkPredictionDataset('data/v03/test_Null', [['NULL']], 500, 10, 30)
    
    loader_train_A = DataLoader(train_A, batch_size=1, shuffle=True)
    loader_train_ABC = DataLoader(train_ABC, batch_size=1, shuffle=True)
    
    loaders_test = {
        'Same Family': DataLoader(test_A, batch_size=1),
        '10x Scale': DataLoader(test_A_scale, batch_size=1),
        'Shift': DataLoader(test_A_noise, batch_size=1),
        'Composition': DataLoader(test_comp, batch_size=1),
        'Held-Out D': DataLoader(test_D, batch_size=1),
        'Held-Out E': DataLoader(test_E, batch_size=1),
        'Null': DataLoader(test_Null, batch_size=1)
    }
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print(f"\n--- Seed {seed+1}/{num_seeds} ---")
        
        # 1. Train Models A
        m3_A = RAIGNN(hidden_channels=64, num_layers=3)
        m4_A = CandidateRelativeRAIGNN(hidden_channels=64, num_layers=3)
        train_model(m3_A, loader_train_A, epochs=3, is_v4=False)
        train_model(m4_A, loader_train_A, epochs=3, is_v4=True)
        
        for key in ['Same Family', '10x Scale', 'Shift']:
            auc_3, auc_4 = evaluate(m3_A, m4_A, loaders_test[key])
            results_v3[key].append(auc_3)
            results_v4[key].append(auc_4)
            
        # 2. Train Models ABC
        m3_ABC = RAIGNN(hidden_channels=64, num_layers=3)
        m4_ABC = CandidateRelativeRAIGNN(hidden_channels=64, num_layers=3)
        train_model(m3_ABC, loader_train_ABC, epochs=3, is_v4=False)
        train_model(m4_ABC, loader_train_ABC, epochs=3, is_v4=True)
        
        for key in ['Composition', 'Held-Out D', 'Held-Out E', 'Null']:
            auc_3, auc_4 = evaluate(m3_ABC, m4_ABC, loaders_test[key])
            results_v3[key].append(auc_3)
            results_v4[key].append(auc_4)
            
        print("Running Means (v0.4):")
        r_str = ""
        for k in results_v4.keys():
            r_str += f"{k}: {np.mean(results_v4[k]):.3f} | "
        print(r_str)
            
    print("\n\n=== FINAL RESULTS (ROC-AUC) OVER 30 SEEDS ===")
    
    def format_ci(data_list):
        m = np.mean(data_list)
        s = np.std(data_list)
        ci = 1.96 * s / np.sqrt(len(data_list))
        return f"{m:.3f} ± {ci:.3f}"

    print(f"{'Model':<20} | {'Same Family':<15} | {'10x Scale':<15} | {'Shift':<15} | {'Composition':<15} | {'Held-Out D':<15} | {'Held-Out E':<15} | {'Null':<15}")
    print("-" * 140)
    
    v3_row = f"{'GraphSAGE v0.3':<20} | "
    v4_row = f"{'Relative v0.4':<20} | "
    
    for key in results_v4.keys():
        v3_row += f"{format_ci(results_v3[key]):<15} | "
        v4_row += f"{format_ci(results_v4[key]):<15} | "
        
    print(v3_row)
    print(v4_row)

if __name__ == "__main__":
    # Ensure reproducibility of dataset generation
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Full 30-seed evaluation
    run_progressive_evaluation(num_seeds=30)
