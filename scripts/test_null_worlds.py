import torch
import numpy as np
from rai.learning.rl_mini_env import MacroSyntheticEnv
from rai.learning.rl_mini_ppo import RAIPolicy

class NullSyntheticEnv(MacroSyntheticEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Completely destroy all cross-variable interactions
        self.W = np.zeros((self.num_vars, self.num_vars))
        self.W_delay = np.zeros((self.num_vars, self.num_vars))
        self.bias = np.zeros(self.num_vars)
        
        # Give them random independent momentum/mean reversion
        self.inertia = self.np_random.uniform(-0.8, 0.8, size=(self.num_vars,))
        self.mean_reversion_strength = self.np_random.uniform(0.0, 0.5, size=(self.num_vars,))
        self.baseline = self.np_random.normal(0, 1.0, size=(self.num_vars,))
        
        return self._get_obs(), {}

def evaluate_null_worlds():
    device = torch.device("cpu")
    
    top_seeds = torch.load("data/rl_macro_test/v04_top_seeds.pt")
    ensemble = []
    for seed in top_seeds:
        policy = RAIPolicy(window_size=3, hidden_dim=64)
        policy.load_state_dict(torch.load(f"data/rl_macro_test/v04_seeds/rai_policy_seed_{seed:02d}.pt", map_location=device))
        policy.eval()
        ensemble.append(policy)
        
    env = NullSyntheticEnv(num_vars=10, window_size=3, max_steps=120, target_blind=True)
    
    y_true = []
    y_rai = []
    
    for _ in range(500): # 500 episodes * ~100 steps = ~50,000 steps
        obs, _ = env.reset()
        done = False
        while not done:
            votes = []
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                for policy in ensemble:
                    logits, _ = policy(obs_tensor)
                    votes.append(torch.argmax(logits, dim=1).item())
                    
            action = 1 if sum(votes) > len(votes)/2 else 0
            
            # Reconstruct true action from reward logic in env
            curr_x = env.history[-1]
            past_x = env.history[-(env.delay_idx + 1)] if len(env.history) > env.delay_idx else curr_x
            velocity = curr_x - env.history[-2] if len(env.history) > 1 else np.zeros(env.num_vars)
            momentum_term = env.inertia * velocity
            mean_rev_term = env.mean_reversion_strength * (env.baseline - curr_x)
            noise = env.np_random.normal(0, 0.1, size=(env.num_vars,))
            next_x = curr_x + momentum_term + mean_rev_term + noise
            next_x = np.clip(next_x, -10.0, 10.0)
            
            actual_diff = next_x[env.target_var] - curr_x[env.target_var]
            actual_up = 1 if actual_diff > 0 else 0
            
            y_true.append(actual_up)
            y_rai.append(action)
            
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            
    acc = np.mean(np.array(y_true) == np.array(y_rai))
    print(f"--- NULL SYNTHETIC WORLDS EVALUATION ---")
    print(f"Prediction Steps: {len(y_true)}")
    print(f"RAI v0.4 Target-Blind Ensemble Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    evaluate_null_worlds()
