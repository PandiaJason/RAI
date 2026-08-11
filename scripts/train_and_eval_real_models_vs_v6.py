"""
Train Standard AI Models on Real 2010-2019 Market Data & Test Alongside Zero-Shot RAI v6
Models Trained on Real Data:
1. Supervised Deep Learning LSTM Predictor
2. Supervised XGBoost Gradient Boosted Trees
3. Supervised Deep MLP Predictor
Zero-Shot Model (0% Real Data):
4. Zero-Shot RAI v6 (End-to-End Transformer)
Test Dataset: Unseen Out-of-Sample Real Market Data (2020-2024)
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_v6_fast import DeepEndToEndTradingNet
from scripts.eval_vs_standard_ai import (
    LSTMPredictor, train_lstm_model, build_xgb_features,
    evaluate_lstm_strategy, evaluate_xgb_strategy, compute_metrics
)


# ═══════════════════════════════════════════════════════════════════
#  SUPERVISED DEEP MLP PREDICTOR (Trained on Real Data)
# ═══════════════════════════════════════════════════════════════════

class SupervisedMLPPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)


def train_real_mlp(train_df, lookback=20, epochs=100, lr=1e-3):
    X, y_reg = build_xgb_features(train_df, lookback)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y_reg).unsqueeze(1)

    model = SupervisedMLPPredictor(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def evaluate_real_mlp(model, test_df, initial_cash=10000.0, lookback=20):
    prices = test_df.values
    T, N = prices.shape
    equity = [initial_cash]
    cash = initial_cash
    shares = np.zeros(N)

    for t in range(lookback + 1, T):
        window = prices[t - lookback:t]
        rets = (window[1:] - window[:-1]) / np.maximum(1e-6, window[:-1])
        feat = np.concatenate([
            np.mean(rets, axis=0), np.std(rets, axis=0),
            (prices[t-1] / prices[t - lookback] - 1.0),
            (prices[t-1] / np.mean(window[-5:], axis=0) - 1.0),
            np.max(rets, axis=0) - np.min(rets, axis=0),
        ]).astype(np.float32)

        with torch.no_grad():
            x = torch.FloatTensor(feat).unsqueeze(0)
            pred_ret = model(x).item()

        wealth = cash + np.sum(shares * prices[t-1])
        if pred_ret > 0:
            shares = (wealth / N) / np.maximum(1e-6, prices[t-1])
            cash = 0.0
        else:
            cash = wealth
            shares = np.zeros(N)

        equity.append(cash + np.sum(shares * prices[t]))

    return equity


# ═══════════════════════════════════════════════════════════════════
#  EVALUATION BRIDGE FOR RAI v6
# ═══════════════════════════════════════════════════════════════════

def eval_v6(model, test_df):
    prices_raw = test_df.values[:, :10]
    T, N = prices_raw.shape
    cash = 5000.0
    init_p = prices_raw[30]
    shares = (5000.0 / N) / init_p
    peak = 10000.0
    wealth_hist = [10000.0]

    obs_history = []
    for t in range(30):
        p = prices_raw[t]; p_prev = prices_raw[max(0, t-1)]
        obs_history.append(np.concatenate([p / prices_raw[30], np.log(p / np.maximum(1e-4, p_prev)), [0.5, 0.0]]).astype(np.float32))

    for t in range(30, T):
        flat_obs = np.concatenate(obs_history).astype(np.float32)
        act = model.get_action(flat_obs, deterministic=True)

        cl = np.clip(act[0], -5, 5)
        target_cash = 1.0 / (1.0 + np.exp(-cl))
        target_stock = 1.0 - target_cash

        ea = np.exp(act[1:] - np.max(act[1:]))
        target_aw = (ea / np.sum(ea)) * target_stock

        p = prices_raw[t]
        w = max(1e-4, cash + np.sum(shares * p))
        caw = (shares * p) / w
        ccf = cash / w

        drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))
        if drift > 0.03:
            tv = abs(cash - w*target_cash) + np.sum(np.abs(shares*p - w*target_aw))
            net = max(1e-4, w - tv * 0.001)
            cash = net * target_cash
            shares = (net * target_aw) / np.maximum(1e-4, p)

        nw = cash + np.sum(shares * p)
        peak = max(peak, nw)
        wealth_hist.append(nw)

        p_prev = prices_raw[t-1]
        obs_history.pop(0)
        obs_history.append(np.concatenate([p / prices_raw[30], np.log(p / np.maximum(1e-4, p_prev)), [cash/nw, np.clip((nw-peak)/peak, -1, 0)]]).astype(np.float32))

    return wealth_hist


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def main():
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)

    print("=" * 115, flush=True)
    print("  TRAINING MODELS ON REAL 2010-2019 DATA & EVALUATING ON UNSEEN 2020-2024 DATA", flush=True)
    print("=" * 115, flush=True)

    # 1. Train LSTM Predictor on Real Data
    print("\n  [1/3] Training Supervised LSTM Predictor on 2010-2019 Real Market Data...", end=" ", flush=True)
    t0 = time.time()
    lstm_model = train_lstm_model(train_df, lookback=20, epochs=100)
    print(f"done in {time.time()-t0:.1f}s", flush=True)

    # 2. Train XGBoost Classifier on Real Data
    print("  [2/3] Training Supervised XGBoost Classifier on 2010-2019 Real Market Data...", end=" ", flush=True)
    t0 = time.time()
    X_tr, y_tr = build_xgb_features(train_df, lookback=20)
    xgb_clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_clf.fit(X_tr, y_tr)
    print(f"done in {time.time()-t0:.1f}s", flush=True)

    # 3. Train Supervised Deep MLP on Real Data
    print("  [3/3] Training Supervised Deep MLP Predictor on 2010-2019 Real Market Data...", end=" ", flush=True)
    t0 = time.time()
    mlp_model = train_real_mlp(train_df, lookback=20, epochs=100)
    print(f"done in {time.time()-t0:.1f}s", flush=True)

    # 4. Load Zero-Shot RAI v6 (0% Real Data)
    v6_model = DeepEndToEndTradingNet(history_len=30, features_per_step=22, action_dim=11)
    v6_path = "./data/v0.6_rl_checkpoints/rai_v6_fast.pt"
    if os.path.exists(v6_path):
        v6_model.load_state_dict(torch.load(v6_path))
        v6_model.eval()

    # ═══════════════════════════════════════════════════════════════════
    #  EVALUATION ON UNSEEN 2020-2024 DATA ($10,000 STARTING CAPITAL)
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*115}", flush=True)
    print(f"  EVALUATION ON UNSEEN OUT-OF-SAMPLE REAL MARKET DATA: 2020-2024 ({len(test_df)} trading days)", flush=True)
    print(f"  Initial Capital: $10,000.00", flush=True)
    print(f"{'='*115}", flush=True)
    print(f"  {'Model / Strategy':<42} | {'Final Value':>12} | {'Net Profit ($)':>14} | {'Sharpe':>7} | {'Max DD (%)':>10} | Real Data Trained?", flush=True)
    print(f"  {'-'*112}", flush=True)

    # SPY
    spy = test_df['SPY'].values
    eq_spy = 10000.0 * (spy / spy[0])
    m_spy = compute_metrics(eq_spy)
    print(f"  {'SPY Buy & Hold (S&P 500 Index)':<42} | ${m_spy['final']:>11,.2f} | ${m_spy['final']-10000:>+13,.2f} | {m_spy['sharpe']:>7.2f} | {m_spy['max_dd_pct']:>9.2f}% | Passive")

    # RAI v6
    eq_v6 = eval_v6(v6_model, test_df)
    m_v6 = compute_metrics(eq_v6)
    print(f"  {'🏆 Zero-Shot RAI v6 (End-to-End Transformer)':<42} | ${m_v6['final']:>11,.2f} | ${m_v6['final']-10000:>+13,.2f} | {m_v6['sharpe']:>7.2f} | {m_v6['max_dd_pct']:>9.2f}% | ❌ 0% Real Data")

    # LSTM
    eq_lstm = evaluate_lstm_strategy(lstm_model, test_df)
    m_lstm = compute_metrics(eq_lstm)
    print(f"  {'LSTM Return Predictor (Deep Learning)':<42} | ${m_lstm['final']:>11,.2f} | ${m_lstm['final']-10000:>+13,.2f} | {m_lstm['sharpe']:>7.2f} | {m_lstm['max_dd_pct']:>9.2f}% | ✅ Trained on 10 Yrs Real")

    # XGBoost
    eq_xgb = evaluate_xgb_strategy(xgb_clf, test_df)
    m_xgb = compute_metrics(eq_xgb)
    print(f"  {'XGBoost Classifier (Machine Learning)':<42} | ${m_xgb['final']:>11,.2f} | ${m_xgb['final']-10000:>+13,.2f} | {m_xgb['sharpe']:>7.2f} | {m_xgb['max_dd_pct']:>9.2f}% | ✅ Trained on 10 Yrs Real")

    # Supervised MLP
    eq_mlp = evaluate_real_mlp(mlp_model, test_df)
    m_mlp = compute_metrics(eq_mlp)
    print(f"  {'Supervised Deep MLP (Deep Learning)':<42} | ${m_mlp['final']:>11,.2f} | ${m_mlp['final']-10000:>+13,.2f} | {m_mlp['sharpe']:>7.2f} | {m_mlp['max_dd_pct']:>9.2f}% | ✅ Trained on 10 Yrs Real")

    print(f"  {'-'*112}", flush=True)

if __name__ == "__main__":
    main()
