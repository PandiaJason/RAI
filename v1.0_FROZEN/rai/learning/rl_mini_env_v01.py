import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MacroSyntheticEnv(gym.Env):
    """
    A synthetic macroeconomic environment using a random nonlinear VAR process.
    Tests whether an agent can learn relational strategies (interpreting graph dynamics)
    and zero-shot transfer them to completely unseen worlds or real data.
    """
    def __init__(self, num_vars=10, window_size=3, max_steps=120):
        super().__init__()
        self.num_vars = num_vars
        self.window_size = window_size
        self.max_steps = max_steps
        
        # Action: Predict DOWN (0) or UP (1) for the target variable.
        self.action_space = spaces.Discrete(2)
        
        # Obs: shape (num_vars, window_size + 1). The +1 is for the 'is_target' flag.
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(self.num_vars, self.window_size + 1), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate random hidden causal matrix W
        # We scale it down so the VAR process is stable (spectral radius < 1).
        self.W = self.np_random.normal(0, 0.5, size=(self.num_vars, self.num_vars))
        self.bias = self.np_random.normal(0, 0.1, size=(self.num_vars,))
        
        self.target_var = self.np_random.integers(0, self.num_vars)
        
        self.history = []
        # Initialize random history for the window
        for _ in range(self.window_size):
            self.history.append(self.np_random.normal(0, 1.0, size=(self.num_vars,)))
            
        self.current_step = 0
        return self._get_obs(), {}
        
    def _get_obs(self):
        # Concatenate last `window_size` steps
        recent_hist = np.stack(self.history[-self.window_size:], axis=1) # (num_vars, window_size)
        
        is_target_flag = np.zeros((self.num_vars, 1), dtype=np.float32)
        is_target_flag[self.target_var, 0] = 1.0
        
        obs = np.concatenate([recent_hist, is_target_flag], axis=1).astype(np.float32)
        return obs
        
    def step(self, action):
        # Compute next state using our hidden VAR-like process
        curr_x = self.history[-1]
        
        # Nonlinear transition: X_{t+1} = \tanh(W * X_t + bias) + noise
        next_x = np.tanh(self.W @ curr_x + self.bias) + self.np_random.normal(0, 0.1, size=(self.num_vars,))
        
        actual_diff = next_x[self.target_var] - curr_x[self.target_var]
        actual_up = 1 if actual_diff > 0 else 0
        
        reward = 1.0 if action == actual_up else -1.0
        
        self.history.append(next_x)
        self.current_step += 1
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), float(reward), terminated, truncated, {}
