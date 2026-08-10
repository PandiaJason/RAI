import torch
import torch.nn as nn
from torch.distributions import Categorical

class RAIPolicy(nn.Module):
    """
    RAI-RL Mini v0.2: Latent World Inference
    
    Architecture:
    1. Independent Node Processing (History -> Local Embs)
    2. Global Pooling -> z_world (Latent World State)
    3. z_world broadcasted back to all nodes
    4. Relational Message Passing (Attention)
    5. Actor uses target node embedding -> UP/DOWN
    """
    def __init__(self, window_size=3, hidden_dim=64):
        super().__init__()
        
        # 1. Local history processor
        self.node_mlp = nn.Sequential(
            nn.Linear(window_size + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. Latent world state extractor
        self.world_inference = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Fuse local embedding with global z_world
        self.fuse_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 4. Relational Message Passing conditioned on z_world
        self.interaction = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.interaction_norm = nn.LayerNorm(hidden_dim)
        
        # 5. Actor decodes from Target node
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        # 6. Critic decodes from z_world
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_vars, window_size + 1)
        is_target = x[:, :, -1] 
        
        # 1. Local processing
        node_embs = self.node_mlp(x) # (batch, num_vars, hidden_dim)
        
        # 2. Infer z_world
        global_context = node_embs.mean(dim=1) # (batch, hidden_dim)
        z_world = self.world_inference(global_context) # (batch, hidden_dim)
        
        # 3. Broadcast and Fuse
        z_world_expanded = z_world.unsqueeze(1).expand(-1, node_embs.shape[1], -1) 
        fused_embs = torch.cat([node_embs, z_world_expanded], dim=-1)
        node_embs = self.fuse_mlp(fused_embs)
        
        # 4. Relational Message Passing
        attn_out, _ = self.interaction(node_embs, node_embs, node_embs)
        node_embs = self.interaction_norm(node_embs + attn_out)
        
        # 5. Extract target node
        batch_size = x.shape[0]
        target_indices = torch.argmax(is_target, dim=1)
        target_embs = node_embs[torch.arange(batch_size), target_indices]
        
        action_logits = self.actor(target_embs)
        value = self.critic(z_world)
        
        return action_logits, value
        
    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value
