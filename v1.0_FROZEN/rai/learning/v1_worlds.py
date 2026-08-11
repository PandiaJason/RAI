import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticWorldsDataset(Dataset):
    def __init__(self, num_samples=10000, num_vars=10, seq_len=50, families=['linear']):
        self.num_samples = num_samples
        self.num_vars = num_vars
        self.seq_len = seq_len
        self.families = families
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # 1. Randomly sample a graph W
        # W_ij means j causes i. W_ii = 0 (no target history dependence)
        W = np.zeros((self.num_vars, self.num_vars))
        density = np.random.uniform(0.2, 0.5)
        for i in range(self.num_vars):
            for j in range(self.num_vars):
                if i != j and np.random.rand() < density:
                    W[i, j] = np.random.uniform(-1.0, 1.0)
                    # Push weights away from 0
                    if abs(W[i, j]) < 0.3:
                        W[i, j] = np.sign(W[i, j]) * 0.3
                    
        # 2. Random delays for each edge (1 to 3)
        delays = np.random.randint(1, 4, size=(self.num_vars, self.num_vars))
        
        # 3. Select law family
        family = np.random.choice(self.families)
        
        # Multiplicative fixed partner graph
        mult_partners = np.random.randint(0, self.num_vars, size=(self.num_vars,))
        
        # 4. Generate sequence
        X = np.zeros((self.seq_len, self.num_vars))
        I_mask = np.zeros((self.seq_len, self.num_vars))
        
        # Initialize past with noise
        for t in range(5):
            X[t] = np.random.normal(0, 0.1, size=(self.num_vars,))
            
        for t in range(4, self.seq_len - 1):
            next_X = np.zeros(self.num_vars)
            
            # Interventions: randomly apply a shock
            if np.random.rand() < 0.2: # 20% chance of shock at any step
                shock_var = np.random.randint(0, self.num_vars)
                shock_val = np.random.choice([-5.0, 5.0]) * np.random.uniform(0.8, 1.2)
                next_X[shock_var] = shock_val
                I_mask[t+1, shock_var] = 1.0
                
            for i in range(self.num_vars):
                if I_mask[t+1, i] == 1.0:
                    continue # Intervened, already set
                    
                val = 0
                for j in range(self.num_vars):
                    if W[i, j] != 0:
                        d = delays[i, j]
                        past_val = X[t - d + 1, j] if (t - d + 1) >= 0 else 0
                        
                        if family == 'linear':
                            val += W[i, j] * past_val
                        elif family == 'tanh':
                            val += np.tanh(W[i, j] * past_val)
                        elif family == 'multiplicative':
                            k = mult_partners[j]
                            past_k = X[t - d + 1, k] if (t - d + 1) >= 0 else 0
                            val += W[i, j] * past_val * np.tanh(past_k)
                        elif family == 'threshold':
                            if abs(past_val) > 1.5:
                                val += W[i, j] * past_val
                        elif family == 'cycles_regime':
                            regime = (t // 10) % 2
                            active_W = W[i, j] if regime == 0 else -W[i, j]
                            val += active_W * past_val
                            
                next_X[i] = val + np.random.normal(0, 0.1) # irreducible noise
                
            X[t+1] = np.clip(next_X, -10, 10)
            
        # Binary graph representing causal existence
        G_true = (W != 0).astype(np.float32)
        
        return {
            'X': torch.tensor(X, dtype=torch.float32), 
            'I': torch.tensor(I_mask, dtype=torch.float32), 
            'G': torch.tensor(G_true, dtype=torch.float32)
        }
