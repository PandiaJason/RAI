import torch
import torch.nn as nn
import torch.optim as optim

class PPOUpdate:
    """
    Handles the PPO proximal policy optimization updates.
    """
    def __init__(self, policy: nn.Module, lr: float = 3e-4, gamma: float = 0.99, clip_param: float = 0.2, ppo_epochs: int = 4):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        
    def compute_gae(self, rewards, values, next_value, masks, tau=0.95):
        returns = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * masks[step] - values[step]
            gae = delta + self.gamma * tau * masks[step] * gae
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns

    def update(self, rollouts):
        """
        rollouts should contain a dictionary of batched tensors:
        obs, actions, log_probs_old, returns, advantages, masks
        """
        obs = rollouts['obs']
        actions = rollouts['actions']
        log_probs_old = rollouts['log_probs_old']
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        action_masks = rollouts['action_masks']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            dist, value = self.policy(obs, action_mask=action_masks)
            entropy = dist.entropy().mean()
            log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - value).pow(2).mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
        return loss.item()
