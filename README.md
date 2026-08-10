# RAI v6: Zero-Shot Sim-to-Real Portfolio Control Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**RAI (Real Artificial Intelligence) v6** is an autonomous Deep Reinforcement Learning framework designed for **Zero-Shot Sim-to-Real transfer** in portfolio management and risk control. 

Instead of training on historical financial market data, RAI v6 is trained on **100% procedurally generated synthetic virtual worlds** ($G_0 \rightarrow G_6$) and evaluated **zero-shot** on real-world asset universes (US Sector ETFs, Mega-Cap Stocks, Global Equity Indices, and Cryptocurrencies).

---

## 🌟 Key Highlights

* **🔒 100% Zero-Knowledge Privacy**: Models are trained without ingesting real historical market data or confidential trade logs, guaranteeing zero data leakage and total privacy.
* **🧠 End-to-End Deep Neural Architecture**: Uses a hybrid **Conv1D + Transformer Encoder** feature extractor (`DeepEndToEndTradingNet`). Operates directly on raw price ratios and log-returns with **zero hand-crafted technical indicators** (no SMAs, RSIs, or MACDs).
* **🛡️ Dynamic Risk Shielding**: Learns emergent risk management primitives inside synthetic environments, dynamically adjusting cash allocation (e.g., 5% in bull markets vs. 40–80% in crash scenarios).
* **📊 Rigorous Controlled Benchmarks**: Fully reproducible multi-seed experimental protocol comparing Synthetic Zero-Shot agents against Real-Data trained agents under 100% identical architectural, hyperparameter, and fee controls.

---

## 🏗️ Model Architecture: `DeepEndToEndTradingNet`

```
  Raw Input (30-day history x 22 features)
                     │
                     ▼
        ┌─────────────────────────┐
        │  1D Conv Feature Layer   │ (Conv1D 22 -> 32 -> 64, LeakyReLU)
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │   Transformer Encoder   │ (Single-layer Encoder, d_model=64, nhead=2)
        └────────────┬────────────┘
                     │
                     ▼
        ┌─────────────────────────┐
        │  Latent Pooling & FC    │ (Mean-pooling across sequence -> Linear 128)
        └──────┬───────────┬──────┘
               │           │
               ▼           ▼
        ┌───────────┐ ┌───────────┐
        │Actor Head │ │Critic Head│ (Continuous action: Cash fraction + Softmax portfolio weights)
        └───────────┘ └───────────┘
```

---

## 📁 Repository Structure

```
RAI/
├── rai/                         # Core RAI package
│   ├── core/                    # Core definitions and base classes
│   ├── generation/              # Synthetic world generators (G0 -> G6)
│   ├── learning/                # PPO agent trainer & hybrid model definitions
│   └── world/                   # Gymnasium-compatible synthetic trading environments
│
├── scripts/                     # RAI v6 & Supporting Benchmark Scripts
│   ├── train_v6_fast.py         # Trains RAI v6 on synthetic multi-regime worlds
│   ├── honest_benchmark.py      # 10-seed controlled comparison (Synthetic vs Real-Data PPO)
│   ├── synthetic_ablation_ladder.py  # 7-level generator ablation experiment (G0 -> G6)
│   ├── cross_domain_eval.py     # Zero-shot evaluation across 4 real asset classes
│   └── allocation_forensics.py  # Portfolio weight & cash allocation forensic analysis
│
├── requirements.txt             # Dependency requirements
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/RAI.git
cd RAI

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train an RAI v6 Agent (Synthetic World)

Train an end-to-end RAI v6 model on synthetic multi-regime random walks ($100,000$ steps):

```bash
python scripts/train_v6_fast.py
```

### 3. Run the Controlled 10-Seed Benchmark

Run the controlled 10-seed comparison evaluating **Synthetic Zero-Shot RAI v6** vs. **Real-Data Trained PPO** under 100% identical controls ($100,000$ steps per model, 2,000,000 total steps):

```bash
python scripts/honest_benchmark.py
```

### 4. Run the Synthetic Ablation Ladder

Test how different synthetic data statistical properties ($G_0$ random walk to $G_6$ full stylized world) impact zero-shot transfer performance:

```bash
python scripts/synthetic_ablation_ladder.py
```

---

## 📊 Benchmark & Experimental Results

### 10-Seed Controlled Benchmark (100,000 Steps / Model)

Tested on 1,459 out-of-sample trading days (2020–2026) across US ETFs (`SPY`, `QQQ`, `EEM`, `VNQ`, `HYG`, `TLT`, `DBC`, `GLD`, `USO`, `UUP`):

| Performance Metric | Real-Data Trained PPO <br>*(Trained on 70% Real Data)* | RAI v6 Zero-Shot <br>*(Trained on 0% Real Data)* | Statistical Hypothesis Test |
|---|---|---|---|
| **OOS Return (Mean ± SD)** | **+29.54 ± 27.58%** | **+44.13 ± 33.14%** | **Welch's $t$-test: $p = 0.3241$ (ns)** |
| **Return 95% CI** | **[+8.75%, +50.34%]** | **[+19.14%, +69.11%]** | **Mann-Whitney $U$-test: $p = 0.2123$ (ns)** |
| **Sharpe Ratio (Mean ± SD)** | **1.371 ± 0.200** | **1.025 ± 0.333** | **Welch's $t$-test: $p = 0.0173$** |
| **Max Drawdown (Mean ± SD)** | **-4.21 ± 4.34%** | **-11.75 ± 7.56%** | **Welch's $t$-test: $p = 0.0209$** |

### Key Experimental Insights

1. **Sim-to-Real Transfer Feasibility**: Synthetic zero-shot training produces strong, profitable trading policies (**+44.13% mean OOS return**, 9/10 positive seeds) without ever ingesting real market data.
2. **Cumulative Return Parity**: There is no statistically significant difference in cumulative out-of-sample return between real-data trained agents (+29.54%) and synthetic agents (+44.13%, $p = 0.3241$).
3. **Tail-Risk Shielding**: Direct exposure to historical market crashes during training gives real-data trained agents higher Sharpe ratios (1.371 vs 1.025, $p = 0.0173$), establishing a clear research target for improving synthetic crash generator realism.

---

## 📜 Citation & Research License

If you use RAI v6 in your research, please cite:

```bibtex
@article{rai_v6_2026,
  title={RAI v6: Testing the Feasibility and Necessity of Synthetic Market Realism for Zero-Shot Portfolio Control},
  author={RAI Development Team},
  journal={GitHub Repository},
  year={2026}
}
```

This project is licensed under the **MIT License**.
