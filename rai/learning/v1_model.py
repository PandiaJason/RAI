import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphInferenceModule(nn.Module):
    def __init__(self, seq_len=50, num_vars=10, hidden_dim=256):
        super().__init__()
        self.num_vars = num_vars
        in_dim = seq_len * 2 * num_vars
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_vars * num_vars)
        )
        
    def forward(self, X, I):
        # X: (B, T, N)
        # I: (B, T, N)
        B = X.shape[0]
        x_flat = X.reshape(B, -1)
        i_flat = I.reshape(B, -1)
        combined = torch.cat([x_flat, i_flat], dim=-1)
        
        g_logits = self.net(combined).view(B, self.num_vars, self.num_vars)
        # Mask diagonal (no self loops)
        mask = torch.eye(self.num_vars, device=X.device).bool().unsqueeze(0)
        g_logits = g_logits.masked_fill(mask, -1e9)
        return g_logits

class PredictionModule(nn.Module):
    def __init__(self, window_size=5, num_vars=10, hidden_dim=64):
        super().__init__()
        self.num_vars = num_vars
        self.window_size = window_size
        
        # Message function: maps a neighbor's historical window to a message vector
        self.msg_net = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Predict function: maps aggregated messages to the next value
        self.pred_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_window, g_prob):
        # x_window: (B, W, N)
        # g_prob: (B, N, N) - g_prob[b, i, j] is prob that j causes i
        B = x_window.shape[0]
        
        x_hist = x_window.transpose(1, 2) # (B, N, W)
        x_hist_flat = x_hist.reshape(-1, self.window_size)
        
        m_flat = self.msg_net(x_hist_flat)
        m = m_flat.view(B, self.num_vars, -1) # (B, N_j, H)
        
        # Aggregate messages
        # BMM: (B, N_i, N_j) x (B, N_j, H) -> (B, N_i, H)
        agg_m = torch.bmm(g_prob, m)
        
        agg_m_flat = agg_m.view(-1, agg_m.shape[-1])
        pred_flat = self.pred_net(agg_m_flat)
        pred = pred_flat.view(B, self.num_vars)
        
        return pred

class RAIV1(nn.Module):
    def __init__(self, seq_len=50, window_size=5, num_vars=10):
        super().__init__()
        self.window_size = window_size
        self.graph_inf = GraphInferenceModule(seq_len=seq_len, num_vars=num_vars)
        self.pred_mod = PredictionModule(window_size=window_size, num_vars=num_vars)
        
    def forward(self, X, I):
        g_logits = self.graph_inf(X, I)
        g_prob = torch.sigmoid(g_logits)
        
        preds = []
        for t in range(self.window_size - 1, X.shape[1] - 1):
            x_window = X[:, t - self.window_size + 1 : t + 1, :]
            pred_t1 = self.pred_mod(x_window, g_prob)
            preds.append(pred_t1.unsqueeze(1))
            
        preds = torch.cat(preds, dim=1)
        return g_logits, preds
