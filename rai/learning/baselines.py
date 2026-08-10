import torch
from rai.learning.actor_critic import SharedActorCritic

class RandomPolicy:
    """
    Selects an action uniformly at random from the valid (unmasked) actions.
    """
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        
    def act(self, obs: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
        # action_masks shape is (batch, num_actions) where False means valid (or True means valid depending on how we did it)
        # In our env, we did:
        # action_mask = torch.zeros(self.num_actions, dtype=torch.bool) # False means valid
        # So we want to sample from where mask == False
        
        # Invert mask so True means valid
        valid_mask = (action_masks == 0)
        
        # Replace False (invalid) with -inf, True (valid) with 0
        logits = torch.where(valid_mask, torch.zeros_like(action_masks, dtype=torch.float), torch.tensor(-1e9))
        
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()

class HeuristicPolicy:
    """
    Greedily selects the highest-index valid transform action available.
    If no transform is available (only action 0 is valid, which is EXPLORE), it chooses 0.
    """
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        
    def act(self, obs: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
        # We want the highest index that is valid (mask == 0)
        valid_mask = (action_masks == 0)
        
        # Create a tensor of indices [0, 1, 2, ..., num_actions-1]
        indices = torch.arange(self.num_actions).unsqueeze(0).expand(valid_mask.shape[0], -1)
        
        # Zero out indices of invalid actions
        valid_indices = indices * valid_mask.long()
        
        # The argmax of this will give the highest index that is valid
        # because valid_indices will have 0 for invalid, and the index for valid
        action = torch.argmax(valid_indices, dim=-1)
        return action

class UntrainedNeuralBaseline:
    """
    A standard PPO network initialized from scratch, exactly like the RAI policy,
    but it has never been trained on the synthetic graphs.
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden_size: int = 128):
        self.policy = SharedActorCritic(obs_dim=obs_dim, num_actions=num_actions, hidden_size=hidden_size)
        self.policy.eval()
        
    def act(self, obs: torch.Tensor, action_masks: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dist, _ = self.policy(obs, action_mask=action_masks)
            return dist.probs.argmax(dim=-1)
