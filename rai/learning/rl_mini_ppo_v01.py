import torch
import torch.nn as nn
from torch.distributions import Categorical

class RAIPolicy(nn.Module):
    """
    A permutation-invariant policy using Self-Attention to process the graph of anonymous variables,
    extracting the target variable's embedding to make a strictly structural UP/DOWN prediction.
    """
    def __init__(self, num_vars=10, window_size=3, hidden_dim=32):
        super().__init__()
        # Features: window_size (recent history) + 1 (is_target flag)
        self.node_mlp = nn.Sequential(
            nn.Linear(window_size + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Fully connected interaction between all variables to deduce relationships
        self.interaction = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.interaction_norm = nn.LayerNorm(hidden_dim)
        
        # Actor: Operates ONLY on the target node's embedding! Guaranteed anonymous/permutation invariant.
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # Critic: Operates on the global graph state
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_vars, window_size + 1)
        is_target = x[:, :, -1] # (batch_size, num_vars)
        
        node_embs = self.node_mlp(x) # (batch, num_vars, hidden_dim)
        
        # Self-attention message passing
        attn_out, _ = self.interaction(node_embs, node_embs, node_embs)
        node_embs = self.interaction_norm(node_embs + attn_out) # Residual + Norm
        
        # Extract target node embedding
        batch_size = x.shape[0]
        # Find index of target node for each item in batch
        target_indices = torch.argmax(is_target, dim=1) # (batch_size,)
        
        # Gather target embeddings
        target_embs = node_embs[torch.arange(batch_size), target_indices] # (batch, hidden_dim)
        
        # Actor computes logits directly from the target variable's context-aware embedding
        action_logits = self.actor(target_embs) # (batch, 2)
        
        # Critic aggregates the whole world state
        global_emb = node_embs.mean(dim=1) # (batch, hidden_dim)
        value = self.critic(global_emb) # (batch, 1)
        
        return action_logits, value
        
    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value
