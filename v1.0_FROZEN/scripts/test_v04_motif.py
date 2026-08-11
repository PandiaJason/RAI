import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import random
import numpy as np

from rai.learning.gnn import RAIGNN
from rai.learning.gnn_v04 import CandidateRelativeRAIGNN, build_candidate_relative_batch

def generate_motif_graph(num_entities=30, num_base_edges=50):
    """
    Generates a graph where the ONLY predictable rule is Triangle Closure:
    If a -> b and b -> c, then a -> c.
    We hide the a -> c edges as positive candidates.
    """
    base_edges = set()
    attempts = 0
    while len(base_edges) < num_base_edges and attempts < num_base_edges * 10:
        attempts += 1
        u = random.randint(0, num_entities - 1)
        v = random.randint(0, num_entities - 1)
        if u != v:
            base_edges.add((u, v))
            
    # Find all a -> b -> c
    adj = {i: set() for i in range(num_entities)}
    for u, v in base_edges:
        adj[u].add(v)
        
    triangle_closures = set()
    for a in range(num_entities):
        for b in adj[a]:
            for c in adj[b]:
                if a != c and c not in adj[a]:
                    triangle_closures.add((a, c))
                    
    pos_candidates = list(triangle_closures)
    # If no closures found, retry (recursively)
    if len(pos_candidates) < 5:
        return generate_motif_graph(num_entities, num_base_edges)
        
    # Cap positives to 30 to keep batch sizes reasonable
    pos_candidates = pos_candidates[:30]
    
    neg_candidates = set()
    while len(neg_candidates) < len(pos_candidates):
        u = random.randint(0, num_entities - 1)
        v = random.randint(0, num_entities - 1)
        if u != v and (u, v) not in base_edges and (u, v) not in triangle_closures:
            neg_candidates.add((u, v))
            
    neg_candidates = list(neg_candidates)
    
    all_cands = pos_candidates + neg_candidates
    labels = [1]*len(pos_candidates) + [0]*len(neg_candidates)
    
    cands_and_labels = list(zip(all_cands, labels))
    random.shuffle(cands_and_labels)
    
    data = HeteroData()
    data['entity'].x = torch.ones((num_entities, 1), dtype=torch.float)
    data['relation'].x = torch.ones((len(base_edges), 1), dtype=torch.float)
    
    edge_index_X_to_R = []
    edge_index_R_to_X = []
    for rel_idx, (src, dst) in enumerate(base_edges):
        edge_index_X_to_R.append([src, rel_idx])
        edge_index_R_to_X.append([rel_idx, dst])
        
    data['entity', 'inputs', 'relation'].edge_index = torch.tensor(edge_index_X_to_R, dtype=torch.long).t().contiguous()
    data['relation', 'outputs', 'entity'].edge_index = torch.tensor(edge_index_R_to_X, dtype=torch.long).t().contiguous()
    data['relation', 'rev_inputs', 'entity'].edge_index = torch.tensor(edge_index_X_to_R, dtype=torch.long).t().contiguous().flip([0])
    data['entity', 'rev_outputs', 'relation'].edge_index = torch.tensor(edge_index_R_to_X, dtype=torch.long).t().contiguous().flip([0])
    
    data.candidate_src = torch.tensor([c[0][0] for c in cands_and_labels], dtype=torch.long)
    data.candidate_dst = torch.tensor([c[0][1] for c in cands_and_labels], dtype=torch.long)
    data.candidate_labels = torch.tensor([c[1] for c in cands_and_labels], dtype=torch.float)
    
    return data

def run_motif_test():
    print("--- RAI v0.4 Test 1: Motif Recovery (Triangle Closure) ---")
    
    train_graphs = [generate_motif_graph(30, 40) for _ in range(200)]
    test_graphs = [generate_motif_graph(30, 40) for _ in range(50)]
    
    # Train v0.3 (Baseline GraphSAGE)
    print("\nTraining v0.3 (GraphSAGE)...")
    model_v3 = RAIGNN(hidden_channels=32, num_layers=2)
    opt_v3 = optim.Adam(model_v3.parameters(), lr=0.01)
    
    for epoch in range(15):
        model_v3.train()
        total_loss = 0
        for data in train_graphs:
            opt_v3.zero_grad()
            logits = model_v3(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), data.candidate_labels)
            loss.backward()
            opt_v3.step()
            total_loss += loss.item()
        if epoch % 5 == 0: print(f"Epoch {epoch} | Loss: {total_loss/len(train_graphs):.4f}")
            
    # Train v0.4 (Candidate-Relative Encoder)
    print("\nTraining v0.4 (Candidate-Relative)...")
    model_v4 = CandidateRelativeRAIGNN(hidden_channels=32, num_layers=2)
    opt_v4 = optim.Adam(model_v4.parameters(), lr=0.01)
    
    for epoch in range(15):
        model_v4.train()
        total_loss = 0
        for data in train_graphs:
            opt_v4.zero_grad()
            batch = build_candidate_relative_batch(data)
            logits = model_v4(batch)
            # Batch offsets candidate_labels identically
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), batch.candidate_labels)
            loss.backward()
            opt_v4.step()
            total_loss += loss.item()
        if epoch % 5 == 0: print(f"Epoch {epoch} | Loss: {total_loss/len(train_graphs):.4f}")
            
    # Evaluate
    print("\nEvaluating...")
    model_v3.eval()
    model_v4.eval()
    
    y_true, y_v3, y_v4 = [], [], []
    with torch.no_grad():
        for data in test_graphs:
            y_true.extend(data.candidate_labels.numpy())
            
            l3 = model_v3(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
            y_v3.extend(l3.view(-1).numpy())
            
            batch = build_candidate_relative_batch(data)
            l4 = model_v4(batch)
            y_v4.extend(l4.view(-1).numpy())
            
    auc_3 = roc_auc_score(y_true, y_v3)
    auc_4 = roc_auc_score(y_true, y_v4)
    
    print(f"\nFinal ROC-AUC:")
    print(f"RAI-GNN v0.3 (GraphSAGE):           {auc_3:.4f}")
    print(f"RAI-GNN v0.4 (Candidate-Relative):  {auc_4:.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    run_motif_test()
