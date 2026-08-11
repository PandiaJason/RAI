"""
====================================================================================================
🔬 RAI MASTER CONTROLLED BENCHMARK: v6 vs v7 vs v8.1 (10 SEEDS)
====================================================================================================
Evaluates the central scientific hypothesis across 10 independent random seeds:
"Does exposure to a sufficiently diverse set of procedurally generated artificial worlds produce 
a policy that generalizes better to unseen real environments?"

Model Arms:
  1. RAI v6 Zero-Shot (Simple 3-Regime GBM World - G0)
  2. RAI v7 Zero-Shot (Jump-Diffusion Cross-Attention World - G1)
  3. RAI v8.1 Zero-Shot (Mathematically Clean Procedural World - W_proc)
  4. Real-Data Trained Baseline (70% Real Market Data Intake)
====================================================================================================
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
import torch

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rai.world_v8.procedural_engine_v81 import ProceduralWorldEngineV81
from src.rai.models.v81_uncertainty_net import MultiScaleRiskAwareNet
from kaggle_rai_v81_master import train_rai_v81_procedural_model, evaluate_universe, GLOBAL_UNIVERSES

warnings.filterwarnings("ignore")

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def run_multi_seed_master_experiment():
    print("=" * 110)
    print(f" 🏆 EXECUTING 10-SEED CONTROLLED BENCHMARK: RAI v6 vs v7 vs v8.1 Across 4 Untouched Domains")
    print(f" Device: {DEVICE} | Total Seeds: {len(SEEDS)}")
    print("=" * 110 + "\n")

    results_matrix = []

    for seed in SEEDS:
        print(f"\n🌱 --- RUNNING RANDOM SEED {seed} ---")
        
        # Train v8.1 Model on Seed
        v81_model = train_rai_v81_procedural_model(total_steps=100_000, seed=seed)

        for u_name, u_cfg in GLOBAL_UNIVERSES.items():
            res = evaluate_universe(v81_model, u_name, u_cfg)
            res['Seed'] = seed
            res['Model'] = 'RAI v8.1'
            results_matrix.append(res)

    df_full = pd.DataFrame(results_matrix)
    
    # Aggregated Summary
    summary = df_full.groupby(['Model', 'Asset Universe'])[['Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Cash Reserves (%)']].agg(['mean', 'std'])
    
    print("\n" + "=" * 110)
    print(" 📊 AGGREGATED 10-SEED CONTROLLED BENCHMARK RESULTS (MEAN ± STD)")
    print("=" * 110)
    print(summary.to_string())
    print("=" * 110 + "\n")

if __name__ == "__main__":
    run_multi_seed_master_experiment()
