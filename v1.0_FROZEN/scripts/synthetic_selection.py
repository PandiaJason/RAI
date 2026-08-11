import torch
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rai.learning.rl_mini_env import MacroSyntheticEnv
from rai.learning.rl_mini_ppo import RAIPolicy
import warnings

warnings.filterwarnings("ignore")

class ControlledMacroSyntheticEnv(MacroSyntheticEnv):
    def __init__(self, regime="Momentum", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regime = regime
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.W = np.zeros((self.num_vars, self.num_vars))
        self.bias = np.zeros(self.num_vars)
        self.inertia = np.zeros(self.num_vars)
        self.mean_reversion_strength = np.zeros(self.num_vars)
        self.W_delay = np.zeros((self.num_vars, self.num_vars))
        
        if self.regime == "Momentum":
            self.inertia = self.np_random.uniform(0.5, 0.9, size=(self.num_vars,))
        elif self.regime == "MeanReversion":
            self.mean_reversion_strength = self.np_random.uniform(0.5, 0.9, size=(self.num_vars,))
            self.baseline = self.np_random.normal(0, 1.0, size=(self.num_vars,))
        elif self.regime == "Interaction":
            self.W = self.np_random.normal(0, 0.5, size=(self.num_vars, self.num_vars))
        elif self.regime == "Delay":
            self.W_delay = self.np_random.normal(0, 0.5, size=(self.num_vars, self.num_vars))
            self.delay_idx = 2
        elif self.regime == "Shocks":
            self.W = np.zeros((self.num_vars, self.num_vars))
            self.inertia = np.zeros(self.num_vars)
            
        return self._get_obs(), {}

def evaluate_model_synthetic(seed):
    device = torch.device("cpu")
    policy = RAIPolicy(window_size=3, hidden_dim=64)
    model_path = f"data/rl_macro_test/v04_seeds/rai_policy_seed_{seed:02d}.pt"
    
    if not os.path.exists(model_path):
        return None
        
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()
    
    # 1. z_world Probing Accuracy
    regimes = ["Momentum", "MeanReversion", "Interaction", "Delay", "Shocks"]
    X = []
    y = []
    
    for label_idx, regime in enumerate(regimes):
        env = ControlledMacroSyntheticEnv(regime=regime, num_vars=10, window_size=3, max_steps=120, target_blind=True)
        # 200 episodes per regime for fast evaluation
        for ep in range(200):
            obs, _ = env.reset()
            steps_to_take = np.random.randint(50, 110)
            for _ in range(steps_to_take):
                action = env.action_space.sample()
                if regime == "Shocks" and np.random.rand() < 0.1:
                    env.history[-1] += np.random.normal(0, 3.0, size=(env.num_vars,))
                obs, _, _, _, _ = env.step(action)
                
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                node_embs = policy.node_mlp(obs_tensor)
                global_context = node_embs.mean(dim=1)
                z_world = policy.world_inference(global_context)
                
            X.append(z_world.squeeze(0).numpy())
            y.append(label_idx)
            
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    probe_acc = accuracy_score(y_test, y_pred)
    
    # 2. Evaluation on Mixed Synthetic Env
    env = MacroSyntheticEnv(num_vars=10, window_size=3, max_steps=120, target_blind=True)
    total_reward = 0
    for _ in range(50):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = policy(obs_tensor)
                action = torch.argmax(logits, dim=1).item()
            obs, r, term, trunc, _ = env.step(action)
            total_reward += r
            done = term or trunc
            
    avg_reward = total_reward / 50.0
    
    # Simple score: normalize both and add
    # Probe acc is [0, 1]. Max theoretically 1.0
    # Avg reward is [-120, 120].
    score = probe_acc * 100.0 + (avg_reward / 120.0) * 100.0
    
    return {
        "seed": seed,
        "probe_acc": probe_acc,
        "avg_reward": avg_reward,
        "score": score
    }

if __name__ == "__main__":
    print("Evaluating 50 TARGET-BLIND models purely on synthetic dynamics...")
    results = []
    for seed in range(200, 250):
        res = evaluate_model_synthetic(seed)
        if res is not None:
            results.append(res)
            
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print("\n--- TOP 5 PURE SYNTHETIC MODELS ---")
    for i in range(5):
        r = results[i]
        print(f"Rank {i+1} | Seed {r['seed']} | Score: {r['score']:.2f} | Probe Acc: {r['probe_acc']*100:.1f}% | Avg Reward: {r['avg_reward']:.1f}")
        
    # Save the top 5 seeds
    top_seeds = [r["seed"] for r in results[:5]]
    torch.save(top_seeds, "data/rl_macro_test/v04_top_seeds.pt")
    print(f"\nTop 5 seeds saved to data/rl_macro_test/v04_top_seeds.pt: {top_seeds}")
