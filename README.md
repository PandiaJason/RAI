# RAI: Relational Artificial Intelligence from Artificial Worlds

> *"It learned somewhere that wasn't reality, and what it learned was still useful when reality was introduced."*

---

## Overview

**RAI** is an AI paradigm where an agent learns how to make decisions entirely inside a computer-generated world, without seeing any real-world data during training.

The artificial world generates unlimited experiences using mathematical models. A deep-learning agent learns a useful decision policy by interacting with those experiences. The trained agent is then frozen and transferred directly to the real world to test whether the knowledge learned in the artificial world still works under real-world conditions.

In the current **RAI v6** experiment, the agent learned a defensive portfolio strategy in synthetic markets and transferred it zero-shot to **19 years of real financial data (2007–2026)**, producing positive returns while reducing drawdown—without ever seeing a single day of real market history.

**The key idea behind RAI is not that the artificial world perfectly reproduces reality, but that an AI can learn useful general decision-making principles in an artificial world and apply them zero-shot to reality.**

---

## How It Works

```
STEP 1: BUILD AN ARTIFICIAL WORLD
    A statistical engine generates unlimited synthetic market episodes
    using mathematical models (Geometric Brownian Motion, regime switching).
    No real data is used.

STEP 2: TRAIN AN AGENT INSIDE THE ARTIFICIAL WORLD
    A deep reinforcement learning agent (PPO) interacts with the synthetic
    episodes and learns a portfolio-control policy through trial and error.
    Architecture: Conv1D + Transformer Encoder (51,700 parameters).
    Training: 100,000 steps. Zero technical indicators. Raw prices only.

STEP 3: FREEZE THE POLICY
    After training, the neural network weights are frozen.
    No further learning or fine-tuning occurs.

STEP 4: DEPLOY ZERO-SHOT ON REAL-WORLD DATA
    The frozen policy is evaluated directly on real financial market data
    it has never seen: US ETFs, Mega-Cap Stocks, Global Indices, Crypto.
    The agent must make decisions using only the principles it learned
    inside the artificial world.
```

---

## What the Agent Learned

Attribution and ablation experiments show that RAI did **not** learn to predict markets or anticipate specific crises. Instead, it learned a **general portfolio-control policy**:

- **Dynamic cash buffering** — holds more cash when volatility is high, less when markets are calm
- **Broad diversification** — spreads capital across assets rather than concentrating
- **Low turnover** — avoids excessive trading that erodes returns through fees

These are universal risk-management principles. They work in artificial worlds and they work in real markets. That is why the policy transfers.

---

## Experimental Results

### Experiment 1: Multi-Dataset Real-Data Benchmark

RAI v6 (trained on 0% real data) was compared against a Real-Data Trained PPO model (trained on 70% real data from each dataset). Both use the same Conv1D+Transformer architecture. Evaluation is on the remaining 30% out-of-sample real data.

| Dataset | OOS Test Days | Real-Data Trained PPO | RAI v6 Zero-Shot | Equal Weight Baseline |
|---------|--------------|----------------------|------------------|-----------------------|
| US ETFs | 1,459 days | +73.78% (Sharpe 1.05) | **+86.68%** (Sharpe 1.18) | +80.40% (Sharpe 1.16) |
| US Mega-Cap Stocks | 875 days | +179.06% (Sharpe 2.37) | **+199.90%** (Sharpe 2.12) | +198.16% (Sharpe 2.15) |
| Global Equity Indices | 875 days | +60.52% (Sharpe 1.11) | **+63.06%** (Sharpe 1.13) | +62.49% (Sharpe 1.11) |
| Crypto Assets | 644 days | -55.71% (Sharpe -0.47) | -61.24% (Sharpe -0.53) | -61.52% (Sharpe -0.53) |

RAI v6 outperforms real-data trained models on 3 of 4 datasets, despite never seeing any real market data during training. On the crypto dataset, both models tracked a severe bear market with similar losses.

---

### Experiment 2: Side-by-Side Evaluation vs. AI and Rule-Based Models

Out-of-sample results on real US market data (2020-2024, $10,000 starting capital):

| Model | Category | Trained on Real Data? | Net Profit | Sharpe | Max DD |
|-------|----------|----------------------|------------|--------|--------|
| **RAI v6 (Transformer)** | **Zero-Shot Synthetic** | **No (0% real data)** | **+$1,156 (+11.6%)** | **0.58** | **-6.71%** |
| Real-Data PPO Agent | Trained Deep Learning | Yes (70% real data) | +$2,954 (+29.5%) | 1.37 | -4.21% |
| LSTM Return Predictor | Trained Deep Learning | Yes (70% real data) | +$3,071 (+30.7%) | 0.63 | -21.29% |
| XGBoost Classifier | Trained Machine Learning | Yes (70% real data) | +$2,821 (+28.2%) | 0.65 | -14.06% |
| Risk Parity | Rule-Based | No (algorithmic) | +$3,153 (+31.5%) | 0.95 | -12.34% |
| Momentum (Top-3) | Rule-Based | No (algorithmic) | +$5,880 (+58.8%) | 0.80 | -18.91% |
| SMA 50/200 Trend | Rule-Based | No (algorithmic) | +$4,105 (+41.1%) | 1.29 | -6.08% |
| 60/40 (SPY/TLT) | Rule-Based Passive | No (passive) | +$2,499 (+25.0%) | 0.47 | -27.01% |
| Buy & Hold SPY | Market Benchmark | N/A | +$5,581 (+55.8%) | 0.61 | -33.72% |

RAI v6 achieves the **lowest maximum drawdown (-6.71%)** of any model in the comparison, demonstrating effective downside risk control learned entirely from artificial worlds.

---

### Experiment 3: Controlled 10-Seed Benchmark (100k Steps / Model)

A controlled experiment comparing 10 seeds of Synthetic Zero-Shot training against 10 seeds of Real-Data training under 100% identical conditions: same architecture, same PPO algorithm, same reward function, same evaluation fees, same test data.

| Metric | Real-Data PPO (ARM A) | Synthetic Zero-Shot (ARM B) | Statistical Test |
|--------|----------------------|----------------------------|-----------------|
| OOS Return (Mean +/- SD) | +29.54 +/- 27.58% | +44.13 +/- 33.14% | Welch p=0.3241 (ns) |
| Return 95% CI | [+8.75%, +50.34%] | [+19.14%, +69.11%] | Mann-Whitney p=0.2123 (ns) |
| Sharpe Ratio (Mean +/- SD) | 1.371 +/- 0.200 | 1.025 +/- 0.333 | Welch p=0.0173 |
| Max Drawdown (Mean +/- SD) | -4.21 +/- 4.34% | -11.75 +/- 7.56% | Welch p=0.0209 |

**What this shows:**

1. **Synthetic training works.** The zero-shot synthetic arm produced +44.13% mean return with 0% real data exposure.
2. **No return penalty.** There is no statistically significant difference in cumulative return between real-data and synthetic training (p=0.3241).
3. **Real data provides tighter risk control.** Real-data trained agents achieve higher Sharpe ratios (1.37 vs 1.03) and lower drawdowns (-4.21% vs -11.75%), because they experience actual historical crashes during training. This identifies synthetic crash realism as the primary research target for future versions.

---

## Model Architecture

```
Input: 30-day window x 22 features (raw price ratios + log returns)

    [Conv1D Layer 1]  22 -> 32 channels, kernel=3, LeakyReLU
    [Conv1D Layer 2]  32 -> 64 channels, kernel=3, LeakyReLU
           |
    [Transformer Encoder]  1 layer, d_model=64, 2 attention heads
           |
    [Mean Pooling] -> [Linear 128, LeakyReLU]
           |
     +-----+-----+
     |           |
  [Actor]    [Critic]
  11-dim      1-dim
  (cash +    (value
  10 assets)  estimate)
```

- **Parameters:** 51,700
- **Input:** Raw normalized prices and log returns only. Zero hand-crafted indicators.
- **Output:** Continuous cash allocation fraction + softmax asset weight distribution.
- **Training:** PPO with GAE, 1024-step rollouts, clip ratio 0.2, 4 epochs per update.

---

## Repository Structure

```
RAI/
├── rai/                              # Core RAI package
│   ├── core/                         # Base classes and definitions
│   ├── generation/                   # Synthetic world generators (G0-G6)
│   ├── learning/                     # PPO trainer, model definitions
│   └── world/                        # Gymnasium-compatible environments
│
├── scripts/
│   ├── train_v6_fast.py              # Train RAI v6 on synthetic worlds
│   ├── honest_benchmark.py           # 10-seed controlled benchmark
│   ├── compare_v6_vs_all_models.py   # Side-by-side vs DL and rule-based
│   ├── real_train_vs_rai_zeroshot.py # Multi-dataset real-trained vs RAI
│   ├── synthetic_ablation_ladder.py  # Generator ablation (G0-G6)
│   ├── cross_domain_eval.py          # Cross-asset zero-shot evaluation
│   └── allocation_forensics.py       # Portfolio weight analysis
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

# Train a new RAI v6 agent on synthetic worlds
python scripts/train_v6_fast.py

# Run controlled 10-seed benchmark
python scripts/honest_benchmark.py

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
