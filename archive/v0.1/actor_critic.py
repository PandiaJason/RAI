import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SharedActorCritic(nn.Module):
    """
    A simple MLP Actor-Critic policy shared among all agents.
    Observations are flattened vectors of (inventory, knowledge_flags).
    Actions are discrete integer indices.
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int = 128):
        super(SharedActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_size, num_actions)
        
        # Critic head (value function)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        features = self.feature_net(obs)
        
        logits = self.actor_head(features)
        
        if action_mask is not None:
            # Mask out invalid actions (replace their logits with a large negative number)
            # action_mask should be 1 for valid actions, 0 for invalid
            # Add a small epsilon to avoid inf, but typically -1e8 is fine for Softmax
            logits = logits + (action_mask - 1.0) * 1e8
            
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        value = self.critic_head(features).squeeze(-1)
        
        return dist, value

    def act(self, obs: torch.Tensor, action_mask: torch.Tensor = None):
        dist, value = self.forward(obs, action_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def get_value(self, obs: torch.Tensor):
        features = self.feature_net(obs)
        return self.critic_head(features).squeeze(-1)
