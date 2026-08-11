import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import random
import numpy as np

from rai.learning.gnn import RAIGNN
from rai.learning.gnn_v04 import CandidateRelativeRAIGNN, build_candidate_relative_batch
from rai.learning.synthetic_dataset import SyntheticLinkPredictionDataset

def run_scale_test():
    print("--- RAI v0.4 Test 2: Scale Transfer (N=20 -> N=200) ---")
    
    print("\nGenerating datasets...")
    # Train N=20
    train_data = SyntheticLinkPredictionDataset('data/v04/train_scale', [['A']], 500, 20, 20)
    # Test N=200
    test_data = SyntheticLinkPredictionDataset('data/v04/test_scale', [['A']], 100, 200, 200)
    
    # Train v0.3 (GraphSAGE)
    print("\nTraining v0.3 (GraphSAGE)...")
    model_v3 = RAIGNN(hidden_channels=32, num_layers=2)
    opt_v3 = optim.Adam(model_v3.parameters(), lr=0.01)
    
    for epoch in range(10):
        model_v3.train()
        total_loss = 0
        for data in train_data:
            opt_v3.zero_grad()
            logits = model_v3(data.x_dict, data.edge_index_dict, data.candidate_src, data.candidate_dst)
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), data.candidate_labels)
            loss.backward()
            opt_v3.step()
            total_loss += loss.item()
        if epoch % 3 == 0: print(f"Epoch {epoch} | Loss: {total_loss/len(train_data):.4f}")
            
    # Train v0.4 (Candidate-Relative Encoder)
    print("\nTraining v0.4 (Candidate-Relative)...")
    model_v4 = CandidateRelativeRAIGNN(hidden_channels=32, num_layers=2)
    opt_v4 = optim.Adam(model_v4.parameters(), lr=0.01)
    
    for epoch in range(10):
        model_v4.train()
        total_loss = 0
        for data in train_data:
            opt_v4.zero_grad()
            batch = build_candidate_relative_batch(data)
            logits = model_v4(batch)
            loss = nn.BCEWithLogitsLoss()(logits.view(-1), batch.candidate_labels)
            loss.backward()
            opt_v4.step()
            total_loss += loss.item()
        if epoch % 3 == 0: print(f"Epoch {epoch} | Loss: {total_loss/len(train_data):.4f}")
            
    # Evaluate
    print("\nEvaluating on N=200 Scale Transfer...")
    model_v3.eval()
    model_v4.eval()
    
    y_true, y_v3, y_v4 = [], [], []
    with torch.no_grad():
        for i, data in enumerate(test_data):
            # Explicit permutation of IDs
            num_nodes = data['entity'].x.shape[0]
            perm = torch.randperm(num_nodes)
            inv_perm = torch.zeros(num_nodes, dtype=torch.long)
            inv_perm[perm] = torch.arange(num_nodes)
            
            p_data = data.clone()
            for edge_type in p_data.edge_types:
                if edge_type[0] == 'entity':
                    p_data[edge_type].edge_index[0] = inv_perm[p_data[edge_type].edge_index[0]]
                if edge_type[2] == 'entity':
                    p_data[edge_type].edge_index[1] = inv_perm[p_data[edge_type].edge_index[1]]
            p_data.candidate_src = inv_perm[p_data.candidate_src]
            p_data.candidate_dst = inv_perm[p_data.candidate_dst]
            
            y_true.extend(p_data.candidate_labels.numpy())
            
            l3 = model_v3(p_data.x_dict, p_data.edge_index_dict, p_data.candidate_src, p_data.candidate_dst)
            y_v3.extend(l3.view(-1).numpy())
            
            batch = build_candidate_relative_batch(p_data)
            l4 = model_v4(batch)
            y_v4.extend(l4.view(-1).numpy())
            
            if i % 10 == 0: print(f"Evaluated {i}/100 large graphs...")
            
    auc_3 = roc_auc_score(y_true, y_v3)
    auc_4 = roc_auc_score(y_true, y_v4)
    
    print(f"\nFinal ROC-AUC:")
    print(f"RAI-GNN v0.3 (GraphSAGE):           {auc_3:.4f}")
    print(f"RAI-GNN v0.4 (Candidate-Relative):  {auc_4:.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    run_scale_test()
