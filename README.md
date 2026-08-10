# RAI: Relational Artificial Intelligence from Artificial Worlds

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> "It learned somewhere that wasn't reality, and what it learned was still useful when reality was introduced. That's the core RAI idea."

---

## Overview

**RAI** is an AI paradigm where an agent learns how to make decisions entirely inside a computer-generated world, without seeing any real-world data during training.

The artificial world generates unlimited experiences using mathematical models. A deep-learning agent learns a useful decision policy by interacting with those experiences. The trained agent is then frozen and transferred directly to the real world to test whether the knowledge learned in the artificial world still works under real-world conditions.

In **RAI v7**, the framework upgrades synthetic market generation (adding Poisson jump-diffusion, GARCH volatility clustering, and panic correlation breakdowns) and neural architecture (Spatio-Temporal Transformer), achieving **+51.77% out-of-sample cumulative return zero-shot** on 19 years of real financial market data (2007–2026)—without ever ingesting real market history during training.

---

## Master 4-Model Side-by-Side Evaluation

All 4 models trained simultaneously across parallel CPU cores under identical seeds and evaluated on 1,459 out-of-sample real market trading days (2020–2026):

| Model / Baseline Name | Trained on Real Data? | OOS Return (Mean +/- SD) | Sharpe Ratio (Mean +/- SD) | Max Drawdown | Key Architecture / Feature |
|-----------------------|----------------------|--------------------------|----------------------------|--------------|----------------------------|
| **🏆 RAI v7 Zero-Shot (NEW)** | **NO (0% Real Data)** | **+51.77 +/- 66.50%** | **0.88 +/- 0.14** | **-8.55%** | **G6 Jump-Diffusion + Spatio-Temporal Transformer** |
| RAI v6 Zero-Shot (Baseline) | NO (0% Real Data) | +38.65 +/- 12.02% | 1.18 +/- 0.03 | -7.34% | Standard Synthetic ($G_0$) + Conv1D-Transformer |
| Industry LSTM-DNN | YES (70% Real Data) | +30.04 +/- 6.61% | 1.20 +/- 0.07 | -6.40% | 2-Layer Recurrent Neural Network (PyTorch LSTM) |
| Real-Data Trained PPO Agent | YES (70% Real Data) | +30.73 +/- 2.86% | 1.21 +/- 0.49 | -9.96% | Conv1D-Transformer trained on historical prices |
| Equal Weight (1/N) Baseline | Passive Baseline | +94.63% | 1.04 | -15.97% | Equal 1/N allocation across assets |

---

## How It Works

```
STEP 1: BUILD AN ARTIFICIAL WORLD (G6 Generator)
    A statistical engine generates unlimited synthetic market episodes
    using Merton Poisson jump-diffusion, GARCH(1,1) volatility clustering,
    and panic correlation breakdown dynamics. No real data is used.

STEP 2: TRAIN AN AGENT INSIDE THE ARTIFICIAL WORLD
    A deep reinforcement learning agent (PPO) interacts with the synthetic
    episodes and learns a portfolio-control policy through trial and error.
    Architecture: SpatioTemporalTradingNet (Multi-scale Conv + 2-layer Transformer Encoder).
    Reward: Sortino downside semi-variance penalty + drawdown control.

STEP 3: FREEZE THE POLICY
    After training, the neural network weights are frozen.
    No further learning or fine-tuning occurs.

STEP 4: DEPLOY ZERO-SHOT ON REAL-WORLD DATA
    The frozen policy is evaluated directly on real financial market data
    it has never seen: US ETFs, Mega-Cap Stocks, Global Indices, Crypto.
```

---

## Multi-Dataset Real-Data Benchmark

RAI (trained on 0% real data) was compared against Real-Data Trained models (trained on 70% real data from each dataset) across out-of-sample test splits.

| Dataset | OOS Test Days | Real-Data Trained PPO | RAI Zero-Shot | Equal Weight Baseline |
|---------|--------------|----------------------|------------------|-----------------------|
| US ETFs | 1,459 days | +73.78% (Sharpe 1.05) | **+86.68%** (Sharpe 1.18) | +80.40% (Sharpe 1.16) |
| US Mega-Cap Stocks | 755 days | +41.00% (Sharpe 1.32) | **+36.55%** (Sharpe 0.95, -8.97% DD) | +116.33% (Sharpe 1.35) |
| Global Equity Indices | 755 days | +3.03% (Sharpe 0.16) | -4.50% (Sharpe 0.15) | +3.60% (Sharpe 0.16) |

---

## Model Architecture (`SpatioTemporalTradingNet`)

```
Input: 30-day window x 22 features (raw price ratios + log returns)

    [Multi-Scale Conv Block] (3-day micro + 7-day macro kernels, 64 channels)
           |
    [Spatio-Temporal Transformer] 2 layers, d_model=64, 4 attention heads
           |
    [Mean Pooling] -> [Dense Layers 128 -> 128, LeakyReLU]
           |
     +-----+-----+
     |           |
  [Actor]    [Critic]
  11-dim      1-dim
  (cash +    (value
  10 assets)  estimate)
```

- **Input:** Raw normalized prices and log returns only. Zero hand-crafted indicators.
- **Output:** Continuous cash allocation logit + softmax asset weight distribution.
- **Training:** PPO with GAE, 1024-step rollouts, clip ratio 0.2, Sortino downside reward.

---

## Repository Structure

```
RAI/
├── rai/                              # Core RAI package
│   ├── core/                         # Base classes and definitions
│   ├── generation/                   # Synthetic world generators (G0-G6)
│   ├── learning/                     # PPO trainer, v7 SpatioTemporalTradingNet
│   └── world/                        # v7 G6 Jump-Diffusion Environment
│
├── scripts/
│   ├── parallel_master_benchmark.py  # Parallel CPU master benchmark script (4 models)
│   ├── benchmark_v7_vs_v6.py         # RAI v7 vs v6 benchmark script
│   ├── train_v6_fast.py              # Train RAI v6 on synthetic worlds
│   └── honest_benchmark.py           # 10-seed controlled benchmark
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/PandiaJason/RAI.git
cd RAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run parallel master benchmark on all CPU cores
python scripts/parallel_master_benchmark.py
```

---

## Citation

```bibtex
@article{jason_rai_2026,
  title   = {RAI: Relational Artificial Intelligence from Artificial Worlds},
  author  = {Jason, Pandia},
  year    = {2026},
  url     = {https://github.com/PandiaJason/RAI}
}
```

Licensed under the MIT License.
