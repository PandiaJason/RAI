import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import time
import os
from rai.learning.rl_mini_env import MacroSyntheticEnv
from rai.learning.rl_mini_ppo import RAIPolicy
from sklearn.metrics import accuracy_score

def train_ppo_seed(seed, save_path, num_envs=8, num_steps=120, total_timesteps=150000, lr=3e-4):
    device = torch.device("cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    envs = [MacroSyntheticEnv(num_vars=10, window_size=3, max_steps=num_steps) for _ in range(num_envs)]
    
    policy = RAIPolicy(window_size=3, hidden_dim=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    update_epochs = 4
    
    num_updates = total_timesteps // (num_envs * num_steps)
    
    obs = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i)
        obs.append(o)
    obs = torch.Tensor(np.stack(obs)).to(device)
    
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
                logits, value = policy(obs)
                probs = Categorical(logits=logits)
                action = probs.sample()
                logprob = probs.log_prob(action)
                batch_values[step] = value.flatten()
            batch_actions[step] = action
            batch_logprobs[step] = logprob
            
            next_obs, rewards, dones = [], [], []
            for i, env in enumerate(envs):
                o, r, terminated, truncated, _ = env.step(action[i].item())
                d = terminated or truncated
                if d: o, _ = env.reset()
                next_obs.append(o)
                rewards.append(r)
                dones.append(d)
                
            obs = torch.Tensor(np.stack(next_obs)).to(device)
            batch_rewards[step] = torch.tensor(rewards).to(device).view(-1)
            batch_dones[step] = torch.tensor(dones).to(device).view(-1)
            
        with torch.no_grad():
            _, next_value = policy(obs)
            next_value = next_value.flatten()
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
                
                logits, newvalue = policy(b_obs[mb_inds])
                probs = Categorical(logits=logits)
                newlogprob = probs.log_prob(b_actions[mb_inds])
                entropy = probs.entropy()
                
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

if __name__ == "__main__":
    os.makedirs('data/rl_macro_test/seeds', exist_ok=True)
    tensors = torch.load("data/rl_macro_test/macro_tensors.pt")
    val_countries = list(tensors.keys())[:10]
    
    results = []
    
    print("Training 20 Independent Seeds...")
    for seed in range(1, 21):
        save_path = f"data/rl_macro_test/seeds/rai_policy_seed_{seed:02d}.pt"
        if not os.path.exists(save_path):
            train_ppo_seed(seed, save_path)
            
        policy = RAIPolicy(window_size=3, hidden_dim=64)
        policy.load_state_dict(torch.load(save_path))
        policy.eval()
        
        y_true_all = []
        preds = []
        for country in val_countries:
            data = tensors[country]
            if data.shape[0] <= 312: continue
            for target_idx in range(data.shape[1]):
                for t in range(312, data.shape[0] - 1):
                    window = data[t - 3:t].unsqueeze(0).transpose(1, 2)
                    is_target = torch.zeros((1, data.shape[1], 1))
                    is_target[0, target_idx, 0] = 1.0
                    obs = torch.cat([window, is_target], dim=2)
                    
                    actual_diff = data[t+1, target_idx] - data[t, target_idx]
                    y_true_all.append(1 if actual_diff > 0 else 0)
                    
                    with torch.no_grad():
                        logits, _ = policy(obs)
                        preds.append(torch.argmax(logits, dim=1).item())
                        
        acc = accuracy_score(y_true_all, preds)
        results.append(acc)
        print(f"Seed {seed:02d} | Val Accuracy: {acc*100:.2f}%", flush=True)
        
    res = np.array(results)
    print("-" * 30)
    print(f"Mean Accuracy: {np.mean(res)*100:.2f}%")
    print(f"Std Dev:       {np.std(res)*100:.2f}%")
    print(f"Min: {np.min(res)*100:.2f}% | Max: {np.max(res)*100:.2f}%")
