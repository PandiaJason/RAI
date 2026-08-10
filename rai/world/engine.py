import numpy as np

class Agent:
    def __init__(self, agent_id, num_resources):
        self.agent_id = agent_id
        self.num_resources = num_resources
        
        # Q is wealth/money
        self.Q = np.random.uniform(100, 500)
        
        # X is inventory of resources
        self.X = np.random.uniform(5, 20, size=num_resources)
        
        # Subsistence requirement per step
        self.subsistence = np.zeros(num_resources)
        if num_resources == 1:
            self.subsistence[0] = 0.1
        else:
            req_indices = np.random.choice(num_resources, size=np.random.randint(1, min(4, num_resources + 1)), replace=False)
            self.subsistence[req_indices] = np.random.uniform(0.05, 0.2, size=len(req_indices))
        
        # Production capacity
        self.capacity = np.random.uniform(1.0, 5.0)
        
        # Recipe: inputs
        self.inputs = np.zeros(num_resources)
        if num_resources > 1:
            in_idx = np.random.choice(num_resources, size=np.random.randint(1, min(3, num_resources)), replace=False)
            self.inputs[in_idx] = np.random.uniform(0.5, 2.0, size=len(in_idx))
            avail_out = [i for i in range(num_resources) if i not in in_idx]
            self.output_idx = np.random.choice(avail_out) if len(avail_out) > 0 else 0
            self.output_amount = np.random.uniform(1.0, 5.0)
        else:
            self.output_idx = 0
            self.output_amount = 0.0
            
        self.bankrupt = False
        
    def respawn(self):
        # Q is wealth/money (re-enter economy with small loan)
        self.Q = np.random.uniform(50, 200)
        
        # X is inventory of resources
        self.X = np.random.uniform(2, 10, size=self.num_resources)
        
        # Subsistence requirement per step
        self.subsistence = np.zeros(self.num_resources)
        if self.num_resources == 1:
            self.subsistence[0] = 0.1
        else:
            req_indices = np.random.choice(self.num_resources, size=np.random.randint(1, min(4, self.num_resources + 1)), replace=False)
            self.subsistence[req_indices] = np.random.uniform(0.05, 0.2, size=len(req_indices))
        
        # Production capacity
        self.capacity = np.random.uniform(1.0, 5.0)
        
        # Recipe: inputs
        self.inputs = np.zeros(self.num_resources)
        if self.num_resources > 1:
            in_idx = np.random.choice(self.num_resources, size=np.random.randint(1, min(3, self.num_resources)), replace=False)
            self.inputs[in_idx] = np.random.uniform(0.5, 2.0, size=len(in_idx))
            avail_out = [i for i in range(self.num_resources) if i not in in_idx]
            self.output_idx = np.random.choice(avail_out) if len(avail_out) > 0 else 0
            self.output_amount = np.random.uniform(1.0, 5.0)
        else:
            self.output_idx = 0
            self.output_amount = 0.0
            
        self.bankrupt = False
        
    def step_subsistence(self, multiplier=1.0):
        if self.bankrupt: return False
        
        actual_subsistence = self.subsistence * multiplier
        if np.all(self.X >= actual_subsistence):
            self.X -= actual_subsistence
            return True
        else:
            self.bankrupt = True
            return False

class World:
    def __init__(self, num_agents=50, num_resources=20, enable_shocks=True, enable_production=True, fixed_prices=False):
        self.num_agents = num_agents
        self.num_resources = num_resources
        self.enable_shocks = enable_shocks
        self.enable_production = enable_production
        self.fixed_prices = fixed_prices
        
        self.agents = [Agent(i, num_resources) for i in range(num_agents)]
        
        if not self.enable_production:
            for a in self.agents:
                a.inputs = np.zeros(num_resources)
                a.output_amount = 0.0
        
        # Automated Market Maker (AMM) pools for each resource
        self.amm_Q = np.random.uniform(1000, 5000, size=num_resources)
        self.amm_X = np.random.uniform(1000, 5000, size=num_resources)
        
        self.current_step = 0
        
    def get_prices(self):
        if self.fixed_prices:
            return np.ones(self.num_resources, dtype=np.float32)
        return self.amm_Q / np.maximum(1e-4, self.amm_X)
        
    def stochastic_evolution(self):
        """Randomly alters the macroeconomic fabric with curriculum scaling."""
        if not self.enable_shocks:
            return
        
        # Curriculum (Linear Interpolation)
        t = self.current_step
        if t < 5000:
            shock_prob_mult = 0.1
        elif t < 15000:
            shock_prob_mult = 0.1 + 0.9 * ((t - 5000) / 10000.0)
        else:
            shock_prob_mult = min(2.0, 1.0 + ((t - 15000) / 50000.0))
            
        # 1. Tech Discovery (0.1% chance per step)
        if np.random.rand() < 0.001 * shock_prob_mult:
            a = np.random.choice(self.agents)
            if not a.bankrupt:
                # Agent discovers a new recipe
                in_idx = np.random.choice(self.num_resources, size=np.random.randint(1, 3), replace=False)
                a.inputs = np.zeros(self.num_resources)
                a.inputs[in_idx] = np.random.uniform(0.1, 1.5, size=len(in_idx))
                avail_out = [i for i in range(self.num_resources) if i not in in_idx]
                a.output_idx = np.random.choice(avail_out)
                a.output_amount = np.random.uniform(2.0, 8.0) # Often better than average
                
        # 2. Demand Shift (0.1% chance per step)
        if np.random.rand() < 0.001 * shock_prob_mult:
            a = np.random.choice(self.agents)
            if not a.bankrupt:
                req_indices = np.random.choice(self.num_resources, size=np.random.randint(1, 4), replace=False)
                a.subsistence = np.zeros(self.num_resources)
                a.subsistence[req_indices] = np.random.uniform(0.1, 0.4, size=len(req_indices))
                
        # 3. Supply Shock (0.01% chance per step)
        if np.random.rand() < 0.0001 * shock_prob_mult:
            res_idx = np.random.randint(self.num_resources)
            if np.random.rand() > 0.5:
                # Discovery (flood market)
                self.amm_X[res_idx] += np.random.uniform(5000, 20000)
            else:
                # Depletion (dry up market)
                self.amm_X[res_idx] = max(1.0, self.amm_X[res_idx] * 0.1)
                self.amm_Q[res_idx] *= 5.0 # Price spikes
        
    def get_prices(self):
        """Returns Q per X for each resource."""
        return self.amm_Q / self.amm_X
        
    def buy(self, agent_id, res_idx, q_amount):
        """Agent spends q_amount of wealth to buy res_idx."""
        agent = self.agents[agent_id]
        if agent.bankrupt or agent.Q < q_amount or q_amount <= 0:
            return 0.0
            
        Q_p = self.amm_Q[res_idx]
        X_p = self.amm_X[res_idx]
        
        # Constant product AMM: x_out = X - (Q * X) / (Q + q_in)
        x_out = X_p - (Q_p * X_p) / (Q_p + q_amount)
        
        self.amm_Q[res_idx] += q_amount
        self.amm_X[res_idx] -= x_out
        agent.Q -= q_amount
        agent.X[res_idx] += x_out
        return x_out
        
    def sell(self, agent_id, res_idx, x_amount):
        """Agent sells x_amount of res_idx to get wealth."""
        agent = self.agents[agent_id]
        if agent.bankrupt or agent.X[res_idx] < x_amount or x_amount <= 0:
            return 0.0
            
        Q_p = self.amm_Q[res_idx]
        X_p = self.amm_X[res_idx]
        
        # Constant product AMM: q_out = Q - (Q * X) / (X + x_in)
        q_out = Q_p - (Q_p * X_p) / (X_p + x_amount)
        
        self.amm_Q[res_idx] -= q_out
        self.amm_X[res_idx] += x_amount
        agent.X[res_idx] -= x_amount
        agent.Q += q_out
        return q_out
        
    def produce(self, agent_id, scale=1.0):
        """Agent uses inputs to produce output, up to capacity * scale."""
        agent = self.agents[agent_id]
        if agent.bankrupt: return False
        
        scale = max(0.0, min(scale, 1.0))
        actual_cap = agent.capacity * scale
        
        req_inputs = agent.inputs * actual_cap
        if np.all(agent.X >= req_inputs):
            agent.X -= req_inputs
            agent.X[agent.output_idx] += agent.output_amount * actual_cap
            return True
        return False
        
    def step(self):
        """Runs the world environment one tick (subsistence, liquidation, and evolution)."""
        self.current_step += 1
        bankruptcies = 0
        
        t = self.current_step
        if t < 5000:
            sub_mult = 0.5
        elif t < 15000:
            sub_mult = 0.5 + 0.5 * ((t - 5000) / 10000.0)
        else:
            sub_mult = 1.0
        
        for a in self.agents:
            if not a.bankrupt:
                survived = a.step_subsistence(multiplier=sub_mult)
                if not survived:
                    bankruptcies += 1
                    # Liquidation: distribute assets to AMM
                    self.amm_X += a.X
                    self.amm_Q += a.Q / self.num_resources
                    a.X[:] = 0.0
                    a.Q = 0.0
            else:
                # Dead agents have a small chance of respawning as new entrants
                # (Except Agent 0, the RL agent, which is controlled externally)
                if a.agent_id != 0 and np.random.rand() < 0.01:
                    a.respawn()
                    
        self.stochastic_evolution()
        return bankruptcies
