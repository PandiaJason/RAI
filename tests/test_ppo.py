import torch
from rai.learning.actor_critic import SharedActorCritic
from rai.learning.ppo import PPOUpdate

def test_ppo_initialization():
    policy = SharedActorCritic(obs_dim=10, num_actions=5, hidden_size=32)
    ppo = PPOUpdate(policy)
    
    obs = torch.randn(2, 10) # 2 agents
    action_masks = torch.ones(2, 5)
    
    dist, value = policy(obs, action_mask=action_masks)
    
    assert dist.probs.shape == (2, 5)
    assert value.shape == (2,)

def test_ppo_update_step():
    policy = SharedActorCritic(obs_dim=10, num_actions=5, hidden_size=32)
    ppo = PPOUpdate(policy, ppo_epochs=1)
    
    rollouts = {
        'obs': torch.randn(2, 10),
        'actions': torch.tensor([0, 1]),
        'log_probs_old': torch.randn(2),
        'returns': torch.tensor([1.0, -1.0]),
        'advantages': torch.tensor([0.5, -0.5]),
        'action_masks': torch.ones(2, 5)
    }
    
    loss = ppo.update(rollouts)
    assert isinstance(loss, float)
