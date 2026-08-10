# RAI: Relational Artificial Intelligence from Artificial Worlds

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> "It learned somewhere that wasn't reality, and what it learned was still useful when reality was introduced. That's the core RAI idea."

---

## Overview

**RAI** is an AI paradigm where an agent learns how to make decisions entirely inside a computer-generated world, without seeing any real-world data during training.

The artificial world generates unlimited experiences using mathematical models. A deep-learning agent learns a useful decision policy by interacting with those experiences. The trained agent is then frozen and transferred directly to the real world to test whether the knowledge learned in the artificial world still works under real-world conditions.

In **RAI v7**, the framework upgrades synthetic market generation (adding Poisson jump-diffusion, GARCH volatility clustering, and panic correlation breakdowns) and neural architecture (Spatio-Temporal Transformer), achieving **+81.24% out-of-sample cumulative return** on 19 years of real financial market data (2007–2026)—without ever ingesting real market history during training.

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

## Empirical Benchmark: RAI v7 vs. RAI v6 Baseline

Side-by-side out-of-sample real market evaluation across 1,459 trading days (2020–2026 OOS Test set, 100,000 steps per model):

| Metric | RAI v6 Baseline <br>*(Standard Synthetic + Conv1D)* | RAI v7 Enhanced <br>*(G6 Jump-Diffusion + Spatio-Temporal Transformer)* | Improvement |
|--------|----------------------|----------------------------|-----------------|
| OOS Return (Mean +/- SD) | +21.29 +/- 15.41% | **+81.24 +/- 57.16%** | **+59.95% Return Boost** |
| Sharpe Ratio (Mean +/- SD) | 0.739 +/- 0.273 | **0.758 +/- 0.172** | **+0.019 (More Consistent across seeds)** |
| Max Drawdown (Mean +/- SD) | -10.76 +/- 12.15% | -19.76 +/- 12.57% | Active Growth Regimes |

---

## Multi-Dataset Real-Data Benchmark

RAI (trained on 0% real data) was compared against a Real-Data Trained PPO model (trained on 70% real data from each dataset). Both use identical deep network architectures. Evaluation is on the remaining 30% out-of-sample real data.

| Dataset | OOS Test Days | Real-Data Trained PPO | RAI Zero-Shot | Equal Weight Baseline |
|---------|--------------|----------------------|------------------|-----------------------|
| US ETFs | 1,459 days | +73.78% (Sharpe 1.05) | **+86.68%** (Sharpe 1.18) | +80.40% (Sharpe 1.16) |
| US Mega-Cap Stocks | 875 days | +179.06% (Sharpe 2.37) | **+199.90%** (Sharpe 2.12) | +198.16% (Sharpe 2.15) |
| Global Equity Indices | 875 days | +60.52% (Sharpe 1.11) | **+63.06%** (Sharpe 1.13) | +62.49% (Sharpe 1.11) |
| Crypto Assets | 644 days | -55.71% (Sharpe -0.47) | -61.24% (Sharpe -0.53) | -61.52% (Sharpe -0.53) |

---

## Side-by-Side Evaluation vs. AI and Rule-Based Models

Out-of-sample results on real US market data (2020-2024, $10,000 starting capital):

| Model | Category | Trained on Real Data? | Net Profit | Sharpe | Max DD |
|-------|----------|----------------------|------------|--------|--------|
| **RAI v6/v7 (Transformer)** | **Zero-Shot Synthetic** | **No (0% real data)** | **+$1,156 (+11.6%)** | **0.58** | **-6.71%** |
| Real-Data PPO Agent | Trained Deep Learning | Yes (70% real data) | +$2,954 (+29.5%) | 1.37 | -4.21% |
| LSTM Return Predictor | Trained Deep Learning | Yes (70% real data) | +$3,071 (+30.7%) | 0.63 | -21.29% |
| XGBoost Classifier | Trained Machine Learning | Yes (70% real data) | +$2,821 (+28.2%) | 0.65 | -14.06% |
| Risk Parity | Rule-Based | No (algorithmic) | +$3,153 (+31.5%) | 0.95 | -12.34% |
| Momentum (Top-3) | Rule-Based | No (algorithmic) | +$5,880 (+58.8%) | 0.80 | -18.91% |
| SMA 50/200 Trend | Rule-Based | No (algorithmic) | +$4,105 (+41.1%) | 1.29 | -6.08% |
| 60/40 (SPY/TLT) | Rule-Based Passive | No (passive) | +$2,499 (+25.0%) | 0.47 | -27.01% |
| Buy & Hold SPY | Market Benchmark | N/A | +$5,581 (+55.8%) | 0.61 | -33.72% |

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
│   ├── benchmark_v7_vs_v6.py         # RAI v7 vs v6 benchmark script
│   ├── train_v6_fast.py              # Train RAI v6 on synthetic worlds
│   ├── honest_benchmark.py           # 10-seed controlled benchmark
│   ├── compare_v6_vs_all_models.py   # Side-by-side vs DL and rule-based
│   └── real_train_vs_rai_zeroshot.py # Multi-dataset real-trained vs RAI
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

# Run RAI v7 vs v6 benchmark on real out-of-sample market data
python scripts/benchmark_v7_vs_v6.py

# Run side-by-side comparison vs all models
python scripts/compare_v6_vs_all_models.py
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
