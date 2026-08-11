import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import time
from rai.learning.rl_mini_env import MacroSyntheticEnv
from rai.learning.rl_mini_ppo import RAIPolicy

# 1. No z_world
class RAIPolicyNoZWorld(nn.Module):
    def __init__(self, window_size=3, hidden_dim=64):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(window_size + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.interaction = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.interaction_norm = nn.LayerNorm(hidden_dim)
        self.actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        is_target = x[:, :, -1]
        node_embs = self.node_mlp(x)
        attn_out, _ = self.interaction(node_embs, node_embs, node_embs)
        node_embs = self.interaction_norm(node_embs + attn_out)
        batch_size = x.shape[0]
        target_indices = torch.argmax(is_target, dim=1)
        target_embs = node_embs[torch.arange(batch_size), target_indices]
        
        action_logits = self.actor(target_embs)
        global_emb = node_embs.mean(dim=1)
        value = self.critic(global_emb)
        return action_logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

# 2. No Attention (Just Node MLP)
class RAIPolicyNoAttention(nn.Module):
    def __init__(self, window_size=3, hidden_dim=64):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(window_size + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        is_target = x[:, :, -1]
        node_embs = self.node_mlp(x)
        batch_size = x.shape[0]
        target_indices = torch.argmax(is_target, dim=1)
        target_embs = node_embs[torch.arange(batch_size), target_indices]
        
        action_logits = self.actor(target_embs)
        global_emb = node_embs.mean(dim=1)
        value = self.critic(global_emb)
        return action_logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

# 3. History MLP (Flattens everything, no node independence)
class RAIPolicyHistoryMLP(nn.Module):
    def __init__(self, num_vars=10, window_size=3, hidden_dim=128):
        super().__init__()
        input_dim = num_vars * (window_size + 1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.actor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        embs = self.mlp(x_flat)
        action_logits = self.actor(embs)
        value = self.critic(embs)
        return action_logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


def train_ppo_ablation(policy_class, save_path, shuffle_labels=False, num_envs=8, num_steps=120, total_timesteps=150000, lr=3e-4):
    device = torch.device("cpu")
    print(f"Training {save_path} on {device}")
    
    envs = [MacroSyntheticEnv(num_vars=10, window_size=3, max_steps=num_steps) for _ in range(num_envs)]
    
    if policy_class == "Full":
        policy = RAIPolicy(window_size=3, hidden_dim=64).to(device)
    elif policy_class == "NoZWorld":
        policy = RAIPolicyNoZWorld(window_size=3, hidden_dim=64).to(device)
    elif policy_class == "NoAttention":
        policy = RAIPolicyNoAttention(window_size=3, hidden_dim=64).to(device)
    elif policy_class == "HistoryMLP":
        policy = RAIPolicyHistoryMLP(num_vars=10, window_size=3, hidden_dim=128).to(device)
        
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    update_epochs = 4
    
    num_updates = total_timesteps // (num_envs * num_steps)
    
    obs = np.stack([env.reset()[0] for env in envs])
    obs = torch.Tensor(obs).to(device)
    
    for update in range(num_updates):
        batch_obs = torch.zeros((num_steps, num_envs) + envs[0].observation_space.shape).to(device)
        batch_actions = torch.zeros((num_steps, num_envs)).to(device)
        batch_logprobs = torch.zeros((num_steps, num_envs)).to(device)
        batch_rewards = torch.zeros((num_steps, num_envs)).to(device)
        batch_dones = torch.zeros((num_steps, num_envs)).to(device)
        batch_values = torch.zeros((num_steps, num_envs)).to(device)
        
        for step in range(num_steps):
            batch_obs[step] = obs
            
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(obs)
                batch_values[step] = value.flatten()
            batch_actions[step] = action
            batch_logprobs[step] = logprob
            
            next_obs, rewards, dones = [], [], []
            for i, env in enumerate(envs):
                o, r, terminated, truncated, _ = env.step(action[i].item())
                d = terminated or truncated
                
                if shuffle_labels:
                    # Random reward destroys the environment logic completely
                    r = 1.0 if np.random.rand() < 0.5 else -1.0
                    
                if d: o, _ = env.reset()
                next_obs.append(o)
                rewards.append(r)
                dones.append(d)
                
            obs = torch.Tensor(np.stack(next_obs)).to(device)
            batch_rewards[step] = torch.tensor(rewards).to(device).view(-1)
            batch_dones[step] = torch.tensor(dones).to(device).view(-1)
            
        with torch.no_grad():
            next_value = policy.get_action_and_value(obs)[3].flatten()
            advantages = torch.zeros_like(batch_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - batch_dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - batch_dones[t]
                    nextvalues = batch_values[t + 1]
                delta = batch_rewards[t] + gamma * nextvalues * nextnonterminal - batch_values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + batch_values
            
        b_obs = batch_obs.reshape((-1,) + envs[0].observation_space.shape)
        b_logprobs = batch_logprobs.reshape(-1)
        b_actions = batch_actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = batch_values.reshape(-1)
        
        b_inds = np.arange(num_envs * num_steps)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_envs * num_steps, 256):
                end = start + 256
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss_unclipped = (newvalue.flatten() - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue.flatten() - b_values[mb_inds], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                
                loss = pg_loss - ent_coef * entropy.mean() + v_loss * vf_coef
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                
    torch.save(policy.state_dict(), save_path)
    print(f"Training Complete. Policy saved to {save_path}")

if __name__ == "__main__":
    import os
    os.makedirs('data/rl_macro_test', exist_ok=True)
    
    untrained = RAIPolicy(window_size=3, hidden_dim=64)
    torch.save(untrained.state_dict(), "data/rl_macro_test/rai_policy_untrained.pt")
    print("Saved untrained policy.")
    
    train_ppo_ablation("NoZWorld", "data/rl_macro_test/rai_policy_nozworld.pt")
    train_ppo_ablation("NoAttention", "data/rl_macro_test/rai_policy_noattention.pt")
    train_ppo_ablation("HistoryMLP", "data/rl_macro_test/rai_policy_historymlp.pt")
    train_ppo_ablation("Full", "data/rl_macro_test/rai_policy_shuffled.pt", shuffle_labels=True)
