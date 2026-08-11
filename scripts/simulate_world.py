import numpy as np
import matplotlib.pyplot as plt
from rai.world.engine import World

def run_simulation(steps=1000, num_agents=50, num_resources=20):
    world = World(num_agents=num_agents, num_resources=num_resources)
    
    history_prices = []
    history_wealth = []
    history_bankruptcies = []
    total_bankrupt = 0
    
    print("Starting RAI World v0.1 Simulation...")
    for t in range(steps):
        # 1. Heuristic Actions for all living agents
        for a in world.agents:
            if a.bankrupt: continue
            
            # 1a. Ensure subsistence (buy missing)
            missing_sub = np.maximum(0, a.subsistence - a.X)
            for res_idx, amount in enumerate(missing_sub):
                if amount > 0:
                    price = world.get_prices()[res_idx]
                    # spend enough Q to get `amount` of X (approximate via price)
                    spend_q = amount * price * 1.1 # 10% buffer
                    world.buy(a.agent_id, res_idx, spend_q)
                    
            # 1b. Try to produce
            req_inputs = a.inputs * a.capacity
            missing_inputs = np.maximum(0, req_inputs - a.X)
            
            # Buy missing inputs if affordable
            for res_idx, amount in enumerate(missing_inputs):
                if amount > 0:
                    price = world.get_prices()[res_idx]
                    spend_q = min(a.Q * 0.5, amount * price * 1.1) # don't spend all wealth on one input
                    if spend_q > 0:
                        world.buy(a.agent_id, res_idx, spend_q)
                        
            # Produce
            produced = world.produce(a.agent_id, scale=1.0)
            
            # 1c. Sell excess output
            if produced:
                # Sell 80% of output to get wealth
                sell_amt = a.output_amount * a.capacity * 0.8
                world.sell(a.agent_id, a.output_idx, sell_amt)
                
        # 2. Step Environment (Subsistence consumption & Liquidation)
        bankruptcies_this_step = world.step()
        total_bankrupt += bankruptcies_this_step
        
        # 3. Logging
        history_prices.append(world.get_prices())
        
        total_q = sum([a.Q for a in world.agents if not a.bankrupt])
        history_wealth.append(total_q)
        history_bankruptcies.append(total_bankrupt)
        
    print(f"Simulation complete. {total_bankrupt}/{num_agents} agents went bankrupt.")
    
    # Plotting
    history_prices = np.array(history_prices)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    for i in range(num_resources):
        plt.plot(history_prices[:, i], alpha=0.5)
    plt.title("Resource Prices (Q/X) over Time")
    plt.ylabel("Price")
    
    plt.subplot(3, 1, 2)
    plt.plot(history_wealth, color='green')
    plt.title("Total Agent Wealth (Q) over Time")
    plt.ylabel("Q")
    
    plt.subplot(3, 1, 3)
    plt.plot(history_bankruptcies, color='red')
    plt.title("Cumulative Bankruptcies")
    plt.xlabel("Time Step")
    plt.ylabel("Dead Agents")
    
    plt.tight_layout()
    plt.savefig("data/rai_world_sim.png")
    print("Saved plot to data/rai_world_sim.png")

if __name__ == "__main__":
    run_simulation()
