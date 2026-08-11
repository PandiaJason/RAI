import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from rai.world.env import RAIWorldEnv

def eval_agent(agent_type, seed, num_steps=100_000, model_path=None):
    if agent_type == "ppo":
        model = RecurrentPPO.load(model_path)
        
    env = RAIWorldEnv()
    obs, _ = env.reset(seed=seed)
        
    total_reward = 0
    step = 0
    deaths = 0
    
    # We run for `num_steps` (e.g. 100 years). If it dies, it respawns automatically via reset.
    # To track deaths correctly since SB3 resets for us? We are not using SB3 VecEnv here, 
    # we are calling step manually. So if done=True, we must call reset() manually.
    
        while step < num_steps:
            if agent_type == "random":
                action = env.action_space.sample()
            elif agent_type == "buy_and_hold":
                # Buy during first 5 steps of life, then hold
                if step % 1000 < 5:
                    action = np.array([1, np.random.choice(range(0, 20))])
                else:
                    action = np.array([0, 0]) # Hold
            elif agent_type == "momentum":
                # Look at price history in obs
                # Obs shape is (history_len * single_obs_dim)
                # single_obs_dim = 2 + 5N
                # prices start at index 2 + 2N (Q, Cap, X, Sub)
                N = env.num_resources
                hist_len = env.history_len
                obs_dim = env.single_obs_dim
                
                prices_t = obs[(hist_len-1)*obs_dim + 2 + 2*N : (hist_len-1)*obs_dim + 2 + 3*N]
                prices_t1 = obs[(hist_len-2)*obs_dim + 2 + 2*N : (hist_len-2)*obs_dim + 2 + 3*N]
                
                deltas = prices_t - prices_t1
                best_buy = np.argmax(deltas)
                worst_sell = np.argmin(deltas)
                
                if deltas[best_buy] > 0 and np.random.rand() > 0.5:
                    action = np.array([1, best_buy])
                elif deltas[worst_sell] < 0:
                    action = np.array([2, worst_sell])
                else:
                    action = np.array([0, 0])
            elif agent_type == "greedy":
                # Greedy Oracle Baseline
                a = env.world.agents[0]
                prices = env.world.get_prices()
                
                # Check subsistence
                missing_sub = np.maximum(0, a.subsistence - a.X)
                needs_buy_idx = np.where(missing_sub > 0)[0]
                
                if len(needs_buy_idx) > 0:
                    action = np.array([1, needs_buy_idx[0]])
                else:
                    # Check if production is profitable
                    input_cost = np.sum(a.inputs * a.capacity * prices)
                    output_rev = a.output_amount * a.capacity * prices[a.output_idx]
                    
                    if output_rev > input_cost:
                        missing_in = np.maximum(0, a.inputs * a.capacity - a.X)
                        needs_buy_in = np.where(missing_in > 0)[0]
                        if len(needs_buy_in) > 0:
                            action = np.array([1, needs_buy_in[0]])
                        else:
                            action = np.array([3, 0]) # Produce
                    else:
                        # Find something to sell that we don't need
                        needed = a.subsistence.copy()
                        if output_rev > input_cost:
                            needed += a.inputs * a.capacity
                        excess = a.X - needed
                        sell_candidates = np.where(excess > 1.0)[0]
                        if len(sell_candidates) > 0:
                            # Sell highest priced
                            best_sell = sell_candidates[np.argmax(prices[sell_candidates])]
                            action = np.array([2, best_sell])
                        else:
                            action = np.array([0, 0])
            elif agent_type == "ppo":
                # For RecurrentPPO, we need hidden state, but since we reset, we can just pass none.
                # Actually sb3 handles recurrent states automatically if not provided in simple .predict
                action, _ = model.predict(obs, deterministic=True)
                
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step += 1
        
        if done:
            deaths += 1
            # Respawn
            obs, _ = env.reset(seed=seed) # We could pass no seed to avoid resetting the World!
            # Wait, if we call env.reset(), it will NOT rebuild self.world because self.world is not None!
            # This perfectly matches the persistence logic.
            
    a = env.world.agents[0]
    avg_wealth = a.Q
    return deaths, avg_wealth, total_reward

def main():
    print("--- RAI World 2 Evaluation (100 Years / 100,000 steps) ---")
    
    # Generate ONE unseen world
    seed = 424242
    
    baselines = ["random", "buy_and_hold", "momentum", "greedy", "ppo"]
    
    for b in baselines:
        print(f"Evaluating {b}...")
        try:
            path = "data/v0.2_rl_checkpoints/rai_world_ppo_final" if b == "ppo" else None
            deaths, wealth, ret = eval_agent(b, seed, num_steps=100_000, model_path=path)
            print(f"{b.ljust(15)} | Deaths: {deaths:5d} | Final Wealth: {wealth:7.1f} | Total Return: {ret:7.1f}")
        except Exception as e:
            print(f"Failed to evaluate {b}: {e}")

if __name__ == "__main__":
    main()
