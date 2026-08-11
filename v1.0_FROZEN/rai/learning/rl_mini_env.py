import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MacroSyntheticEnv(gym.Env):
    """
    RAI-RL Mini v0.2
    A universal dynamical environment containing a distribution over generic rules:
    - Persistence/Inertia (Momentum)
    - Mean Reversion
    - Delays
    - Structural Shocks
    - Oscillators
    - Generic dense nonlinear interaction
    
    The agent must infer the rules of the current world from the short observation history.
    """
    def __init__(self, num_vars=10, window_size=3, max_steps=120, target_blind=False):
        super().__init__()
        self.num_vars = num_vars
        self.window_size = window_size
        self.max_steps = max_steps
        self.target_blind = target_blind
        
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(self.num_vars, self.window_size + 1), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Base linear interactions (dense but weak)
        self.W = self.np_random.normal(0, 0.1, size=(self.num_vars, self.num_vars))
        self.bias = self.np_random.normal(0, 0.1, size=(self.num_vars,))
        
        # 2. Persistence/Inertia (Momentum)
        # Random inertia per node (-0.8 to 0.8)
        self.inertia = self.np_random.uniform(-0.8, 0.8, size=(self.num_vars,))
        
        # 3. Mean Reversion
        self.mean_reversion_strength = self.np_random.uniform(0.0, 0.5, size=(self.num_vars,))
        self.baseline = self.np_random.normal(0, 1.0, size=(self.num_vars,))
        
        # 4. Delays
        if self.window_size > 1:
            self.W_delay = self.np_random.normal(0, 0.2, size=(self.num_vars, self.num_vars))
            self.delay_idx = self.np_random.integers(1, self.window_size)
        else:
            self.W_delay = np.zeros((self.num_vars, self.num_vars))
            self.delay_idx = 0
            
        self.target_var = self.np_random.integers(0, self.num_vars)
        
        self.history = []
        for _ in range(self.window_size):
            self.history.append(self.np_random.normal(0, 1.0, size=(self.num_vars,)))
            
        self.current_step = 0
        return self._get_obs(), {}
        
    def _get_obs(self):
        recent_hist = np.stack(self.history[-self.window_size:], axis=1) # (num_vars, window_size)
        if self.target_blind:
            recent_hist[self.target_var, :] = 0.0 # Force relational reasoning
            
        is_target_flag = np.zeros((self.num_vars, 1), dtype=np.float32)
        is_target_flag[self.target_var, 0] = 1.0
        obs = np.concatenate([recent_hist, is_target_flag], axis=1).astype(np.float32)
        return obs
        
    def step(self, action):
        curr_x = self.history[-1]
        past_x = self.history[-(self.delay_idx + 1)] if len(self.history) > self.delay_idx else curr_x
        
        base_interaction = np.tanh(self.W @ curr_x)
        delayed_interaction = np.tanh(self.W_delay @ past_x)
        
        # True Velocity Momentum (Inertia of movement)
        velocity = curr_x - self.history[-2] if len(self.history) > 1 else np.zeros(self.num_vars)
        momentum_term = self.inertia * velocity
        
        # True Mean Reversion (pulls towards baseline)
        mean_rev_term = self.mean_reversion_strength * (self.baseline - curr_x)
        
        # Shocks
        shock = np.zeros(self.num_vars)
        if self.np_random.uniform(0, 1) < 0.1:
            shock_var = self.np_random.integers(0, self.num_vars)
            shock[shock_var] = self.np_random.normal(0, 3.0)
            
        noise = self.np_random.normal(0, 0.1, size=(self.num_vars,))
        
        # X_{t+1} = X_t + \Delta X
        # For base interaction and delays, they act as forcing functions on the state
        next_x = curr_x + base_interaction + delayed_interaction + momentum_term + mean_rev_term + self.bias + shock + noise
        next_x = np.clip(next_x, -10.0, 10.0)
        
        actual_diff = next_x[self.target_var] - curr_x[self.target_var]
        actual_up = 1 if actual_diff > 0 else 0
        
        reward = 1.0 if action == actual_up else -1.0
        
        self.history.append(next_x)
        self.current_step += 1
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), float(reward), terminated, truncated, {}
