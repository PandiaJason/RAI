import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import random
import numpy as np

from rai.learning.gnn import RAIGNN
from rai.learning.gnn_v04 import CandidateRelativeRAIGNN, build_candidate_relative_batch
from rai.learning.synthetic_dataset import SyntheticLinkPredictionDataset

def run_composition_test():
    print("--- RAI v0.4 Test 3: Composition Transfer (Train P, Q -> Test P+Q) ---")
    
    print("\nGenerating datasets...")
    # Train P (Family A), Train Q (Family C)
    # The dataset allows shuffling different families, so we just use ['A'], ['C']
    train_data = SyntheticLinkPredictionDataset('data/v04/train_comp', [['A'], ['C']], 1000, 20, 40)
    # Test P+Q (Family A+C)
    test_data = SyntheticLinkPredictionDataset('data/v04/test_comp', [['A', 'C']], 200, 20, 40)
    
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
    print("\nEvaluating on Composition Transfer P+Q...")
    model_v3.eval()
    model_v4.eval()
    
    y_true, y_v3, y_v4 = [], [], []
    with torch.no_grad():
        for i, data in enumerate(test_data):
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
    run_composition_test()
