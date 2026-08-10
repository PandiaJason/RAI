import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rai.world.engine import World
from collections import deque

STAGE_CONFIGS = {
    1: {"num_resources": 1,  "num_agents": 1,  "enable_shocks": False, "enable_production": False, "fixed_prices": True},
    2: {"num_resources": 3,  "num_agents": 5,  "enable_shocks": False, "enable_production": False, "fixed_prices": False},
    3: {"num_resources": 5,  "num_agents": 10, "enable_shocks": False, "enable_production": True,  "fixed_prices": False},
    4: {"num_resources": 10, "num_agents": 20, "enable_shocks": False, "enable_production": True,  "fixed_prices": False},
    5: {"num_resources": 20, "num_agents": 50, "enable_shocks": True,  "enable_production": True,  "fixed_prices": False},
}

class RAIWorldEnv(gym.Env):
    """
    Gymnasium environment for RAI World Curriculum (XEconomics).
    Pads feature observations to 20 resources (102 single_obs_dim, 3264 full obs_dim)
    so policy neural network transfers seamlessly across curriculum stages.
    """
    def __init__(self, stage=1, history_len=32, max_resources=20):
        super().__init__()
        self.stage = stage
        self.history_len = history_len
        self.max_resources = max_resources
        
        cfg = STAGE_CONFIGS[stage]
        self.num_agents = cfg["num_agents"]
        self.num_resources = cfg["num_resources"]
        
        # Action space fixed to max_resources so network architecture doesn't change
        self.action_space = spaces.MultiDiscrete([4, max_resources])
        
        # Single observation size: 2 + 5 * max_resources = 102
        self.single_obs_dim = 2 + 5 * max_resources
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(self.history_len * self.single_obs_dim,), 
            dtype=np.float32
        )
        
        self.world = None
        self.obs_history = deque(maxlen=self.history_len)
        self.last_action_type = 0
        
    def set_stage(self, stage):
        self.stage = stage
        cfg = STAGE_CONFIGS[stage]
        self.num_agents = cfg["num_agents"]
        self.num_resources = cfg["num_resources"]
        self.world = None
        
    def _get_single_obs(self):
        agent = self.world.agents[0]
        prices = self.world.get_prices()
        
        # Pad features to max_resources (20)
        X_pad = np.zeros(self.max_resources, dtype=np.float32)
        X_pad[:self.num_resources] = agent.X
        
        sub_pad = np.zeros(self.max_resources, dtype=np.float32)
        sub_pad[:self.num_resources] = agent.subsistence
        
        prices_pad = np.ones(self.max_resources, dtype=np.float32)
        prices_pad[:self.num_resources] = prices
        
        inputs_pad = np.zeros(self.max_resources, dtype=np.float32)
        inputs_pad[:self.num_resources] = agent.inputs
        
        out_pad = np.zeros(self.max_resources, dtype=np.float32)
        if agent.output_idx < self.max_resources:
            out_pad[agent.output_idx] = agent.output_amount
            
        obs = np.concatenate([
            [agent.Q, agent.capacity],
            X_pad,
            sub_pad,
            prices_pad,
            inputs_pad,
            out_pad
        ])
        return obs.astype(np.float32)
        
    def _get_obs(self):
        while len(self.obs_history) < self.history_len:
            self.obs_history.append(np.zeros(self.single_obs_dim, dtype=np.float32))
            
        return np.concatenate(self.obs_history).astype(np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        cfg = STAGE_CONFIGS[self.stage]
        if self.world is None:
            self.world = World(
                num_agents=cfg["num_agents"],
                num_resources=cfg["num_resources"],
                enable_shocks=cfg["enable_shocks"],
                enable_production=cfg["enable_production"],
                fixed_prices=cfg["fixed_prices"]
            )
            self.world.agents[0].Q = np.random.uniform(100, 300)
            self.world.agents[0].X[:self.num_resources] = np.random.uniform(5, 15, size=self.num_resources)
        else:
            self.world.agents[0].respawn()
            
        self.obs_history.clear()
        initial_obs = self._get_single_obs()
        for _ in range(self.history_len):
            self.obs_history.append(initial_obs)
            
        return self._get_obs(), {}
        
    def _get_survival_buffer(self, agent):
        prices = self.world.get_prices()
        cost_per_step = np.sum(agent.subsistence * prices)
        if cost_per_step < 1e-6:
            return 100.0
        total_wealth = agent.Q + np.sum(agent.X * prices)
        return total_wealth / cost_per_step

    def step(self, action):
        agent = self.world.agents[0]
        
        if agent.bankrupt:
            return self._get_obs(), -5.0, True, False, {}
            
        prices = self.world.get_prices()
        old_w = agent.Q + np.sum(agent.X * prices)
        old_buffer = self._get_survival_buffer(agent)
        
        act_type = int(action[0])
        res_idx = int(action[1]) % self.num_resources # Wrap to active resource range
        self.last_action_type = act_type
        
        if act_type == 0:
            pass # Hold
        elif act_type == 1: # Buy
            if agent.Q >= prices[res_idx]:
                units = min(1.0, agent.Q / prices[res_idx])
                cost = units * prices[res_idx]
                agent.Q -= cost
                agent.X[res_idx] += units
                self.world.amm_Q[res_idx] += cost
                self.world.amm_X[res_idx] = max(1.0, self.world.amm_X[res_idx] - units)
        elif act_type == 2: # Sell
            if agent.X[res_idx] >= 1.0:
                units = 1.0
                revenue = units * prices[res_idx]
                agent.X[res_idx] -= units
                agent.Q += revenue
                self.world.amm_Q[res_idx] = max(1.0, self.world.amm_Q[res_idx] - revenue)
                self.world.amm_X[res_idx] += units
        elif act_type == 3: # Produce
            if self.stage >= 3:
                can_produce = np.all(agent.X >= agent.inputs * agent.capacity)
                if can_produce:
                    agent.X -= agent.inputs * agent.capacity
                    agent.X[agent.output_idx] += agent.output_amount * agent.capacity
                    
        bankruptcies = self.world.step()
        done = agent.bankrupt
        
        current_w = agent.Q + np.sum(agent.X * prices)
        current_buffer = self._get_survival_buffer(agent)
        
        self.obs_history.append(self._get_single_obs())
        
        if done:
            reward = -5.0
        else:
            eps = 1e-4
            delta_log_w = np.log(current_w + eps) - np.log(old_w + eps)
            delta_buffer = current_buffer - old_buffer
            
            reward_w = 0.20 * np.clip(delta_log_w, -1.0, 1.0)
            reward_b = 0.30 * np.clip(delta_buffer, -1.0, 1.0)
            reward = reward_w + reward_b + 0.01
            
        return self._get_obs(), reward, done, False, {}
