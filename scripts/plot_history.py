import json
import matplotlib.pyplot.subplots as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_civilization_history(log_path, save_path):
    print(f"Loading history from {log_path}...")
    data = []
    with open(log_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
            
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data found!")
        return
        
    print(f"Loaded {len(df)} years of history.")
    
    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # 1. Macroeconomy: Total Wealth & Population
    ax1 = axs[0]
    ax1.plot(df["year"], df["total_wealth"], color="gold", label="Total Wealth")
    ax1.set_ylabel("Total Wealth", color="gold")
    ax1.tick_params(axis='y', labelcolor="gold")
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df["year"], df["population"], color="blue", label="Agent Population", alpha=0.6)
    ax1_twin.set_ylabel("Population", color="blue")
    ax1_twin.tick_params(axis='y', labelcolor="blue")
    ax1.set_title("Macroeconomic Growth & Population")
    
    # 2. Prices & Inequality
    ax2 = axs[1]
    ax2.plot(df["year"], df["avg_prices"], color="green", label="Avg Prices")
    ax2.plot(df["year"], df["max_price"], color="red", label="Max Price", alpha=0.5)
    ax2.set_ylabel("Prices (Q/X)")
    ax2.legend()
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df["year"], df["inequality_gini"], color="purple", linestyle="--", label="Wealth Gini")
    ax2_twin.set_ylabel("Gini Coefficient (0 to 1)", color="purple")
    ax2.set_title("Market Dynamics & Wealth Inequality")
    
    # 3. RAI Performance
    ax3 = axs[2]
    ax3.plot(df["year"], df["rai_wealth"], color="orange", label="RAI Wealth (Q)")
    ax3.set_ylabel("RAI Wealth")
    ax3.legend()
    ax3.set_title("RAI Accumulated Wealth Over Time")
    
    # 4. RAI Lifespan
    ax4 = axs[3]
    ax4.plot(df["year"], df["rai_lifespan"], color="cyan", label="RAI Lifespan (Years)")
    ax4.set_ylabel("Lifespan")
    ax4.set_xlabel("Virtual Year")
    ax4.legend()
    ax4.set_title("RAI Agent Survival Duration")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_civilization_history(
        "data/v0.2_rl_checkpoints/civilization_history.jsonl", 
        "data/v0.2_rl_checkpoints/civilization_history_plot.png"
    )
