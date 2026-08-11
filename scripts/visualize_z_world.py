import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from rai.learning.rl_mini_env import MacroSyntheticEnv
from rai.learning.rl_mini_ppo import RAIPolicy
import os

class ControlledMacroSyntheticEnv(MacroSyntheticEnv):
    def __init__(self, regime="Momentum", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regime = regime
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Zero out everything first
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
            
        return self._get_obs(), {}

def visualize_latent_world():
    device = torch.device("cpu")
    policy = RAIPolicy(window_size=3, hidden_dim=64)
    policy.load_state_dict(torch.load("data/rl_macro_test/rai_policy.pt", map_location=device))
    policy.eval()
    
    regimes = ["Momentum", "MeanReversion", "Interaction", "Delay"]
    z_worlds = []
    labels = []
    
    print("Collecting z_world representations...")
    for regime in regimes:
        env = ControlledMacroSyntheticEnv(regime=regime, num_vars=10, window_size=3, max_steps=120)
        
        for ep in range(100): # 100 environments per regime
            obs, _ = env.reset()
            for step in range(100):
                # We don't care about actions, just step with random actions to build history
                action = env.action_space.sample()
                obs, _, _, _, _ = env.step(action)
                
            # At step 100, extract z_world
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                node_embs = policy.node_mlp(obs_tensor)
                global_context = node_embs.mean(dim=1)
                z_world = policy.world_inference(global_context)
                
            z_worlds.append(z_world.squeeze(0).numpy())
            labels.append(regime)
            
    z_worlds = np.array(z_worlds)
    
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    z_2d = reducer.fit_transform(z_worlds)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=z_2d[:, 0], y=z_2d[:, 1],
        hue=labels,
        palette="tab10",
        s=100, alpha=0.8
    )
    
    plt.title("UMAP Projection of Latent z_world Space")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title="True Generating Regime")
    plt.tight_layout()
    
    os.makedirs("/Users/admin/.gemini/antigravity/brain/06a2b185-4a00-4d54-92b5-9a005945b0b2/scratch", exist_ok=True)
    save_path = "/Users/admin/.gemini/antigravity/brain/06a2b185-4a00-4d54-92b5-9a005945b0b2/scratch/z_world_clusters.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    visualize_latent_world()
