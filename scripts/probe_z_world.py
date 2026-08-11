import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from rai.learning.rl_mini_env import MacroSyntheticEnv
from rai.learning.rl_mini_ppo import RAIPolicy
import warnings

# Suppress LogisticRegression max_iter warnings
warnings.filterwarnings("ignore")

class ControlledMacroSyntheticEnv(MacroSyntheticEnv):
    def __init__(self, regime="Momentum", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regime = regime
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Zero out everything
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
            # For pure shocks, everything is 0 except the random epsilon is huge
            self.W = np.zeros((self.num_vars, self.num_vars))
            self.inertia = np.zeros(self.num_vars)
            
        return self._get_obs(), {}

def probe_z_world():
    device = torch.device("cpu")
    policy = RAIPolicy(window_size=3, hidden_dim=64)
    # Using the primary verified v0.2 policy
    policy.load_state_dict(torch.load("data/rl_macro_test/rai_policy.pt", map_location=device))
    policy.eval()
    
    regimes = ["Momentum", "MeanReversion", "Interaction", "Delay", "Shocks"]
    
    X = []
    y = []
    
    print("Collecting z_world vectors for diagnostic regimes...")
    for label_idx, regime in enumerate(regimes):
        env = ControlledMacroSyntheticEnv(regime=regime, num_vars=10, window_size=3, max_steps=120)
        
        # Collect 1000 examples per regime
        for ep in range(1000):
            obs, _ = env.reset()
            # Randomly progress to build history
            steps_to_take = np.random.randint(50, 110)
            for _ in range(steps_to_take):
                action = env.action_space.sample()
                if regime == "Shocks":
                    # Manually inject huge shocks occasionally
                    if np.random.rand() < 0.1:
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
    
    print("\n--- LINEAR PROBING OF z_world ---")
    print("Classes:", regimes)
    print("Random Chance:", 1.0 / len(regimes))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nLinear Probe Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=regimes))

if __name__ == "__main__":
    probe_z_world()
