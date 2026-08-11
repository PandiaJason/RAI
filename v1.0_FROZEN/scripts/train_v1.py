import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from rai.learning.v1_worlds import SyntheticWorldsDataset
from rai.learning.v1_model import RAIV1
import os

def train():
    device = torch.device("cpu")
    print("Generating training dataset (Families A & B)...")
    train_dataset = SyntheticWorldsDataset(num_samples=5000, families=['linear', 'tanh'])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = RAIV1(seq_len=50, window_size=5, num_vars=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss(reduction='none') 
    
    lambda_graph = 1.0
    gamma_interv = 5.0
    
    os.makedirs("data/v1", exist_ok=True)
    
    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_l_graph = 0
        total_l_fut = 0
        total_l_int = 0
        
        for batch in train_loader:
            X = batch['X'].to(device)
            I = batch['I'].to(device)
            G_true = batch['G'].to(device)
            
            optimizer.zero_grad()
            
            g_logits, preds = model(X, I)
            
            l_graph = criterion_bce(g_logits, G_true)
            
            window_size = 5
            targets = X[:, window_size:, :]
            i_mask = I[:, window_size:, :] 
            i_mask_past = I[:, window_size-1:-1, :] 
            
            valid_mask = (i_mask == 0).float()
            raw_mse = criterion_mse(preds, targets) * valid_mask
            
            any_intervention_past = (i_mask_past.sum(dim=-1) > 0).float().unsqueeze(-1)
            
            l_fut_mask = (1.0 - any_intervention_past) * valid_mask
            l_int_mask = any_intervention_past * valid_mask
            
            l_fut = (raw_mse * l_fut_mask).sum() / (l_fut_mask.sum() + 1e-8)
            l_int = (raw_mse * l_int_mask).sum() / (l_int_mask.sum() + 1e-8)
            
            loss = l_fut + lambda_graph * l_graph + gamma_interv * l_int
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_l_graph += l_graph.item()
            total_l_fut += l_fut.item()
            total_l_int += l_int.item()
            
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | L_G: {total_l_graph/len(train_loader):.4f} | L_F: {total_l_fut/len(train_loader):.4f} | L_I: {total_l_int/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), "data/v1/rai_v1_model.pt")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
