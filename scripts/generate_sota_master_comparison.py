"""
Master SOTA Comparison Report: RAI v6 ALPHA vs Industry Best Models
Starting Capital: $10,000.00
"""
import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_v6_fast import DeepEndToEndTradingNet
from scripts.eval_vs_standard_ai import (
    train_lstm_model, evaluate_lstm_strategy,
    build_xgb_features, evaluate_xgb_strategy,
    evaluate_risk_parity, evaluate_momentum, evaluate_sma_crossover,
    GradientBoostingClassifier, compute_metrics
)

def build_master_comparison():
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)

    # 1. Load RAI v6 ALPHA
    v6_alpha_model = DeepEndToEndTradingNet(history_len=30, features_per_step=22, action_dim=11)
    v6_alpha_path = "./data/v0.6_rl_checkpoints/rai_v6_alpha.pt"
    if os.path.exists(v6_alpha_path):
        v6_alpha_model.load_state_dict(torch.load(v6_alpha_path))
        v6_alpha_model.eval()

    # 2. Train SOTA Supervised AI Models on Real Data (2010-2019)
    lstm_model = train_lstm_model(train_df, lookback=20, epochs=100)
    X_tr, y_tr = build_xgb_features(train_df, lookback=20)
    xgb_clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_clf.fit(X_tr, y_tr)

    def eval_v6_alpha(df):
        prices_raw = df.values[:, :10]
        T, N = prices_raw.shape
        cash = 500.0
        init_p = prices_raw[30]
        shares = (9500.0 / N) / init_p
        peak = 10000.0
        wealth_hist = [10000.0]

        obs_history = []
        for t in range(30):
            p = prices_raw[t]; p_prev = prices_raw[max(0, t-1)]
            obs_history.append(np.concatenate([p / prices_raw[30], np.log(p / np.maximum(1e-4, p_prev)), [0.05, 0.0]]).astype(np.float32))

        for t in range(30, T):
            flat_obs = np.concatenate(obs_history).astype(np.float32)
            act = v6_alpha_model.get_action(flat_obs, deterministic=True)
            cl = np.clip(act[0] - 2.5, -8.0, 3.0)
            target_cash = 1.0 / (1.0 + np.exp(-cl))
            target_stock = 1.0 - target_cash
            ea = np.exp(act[1:] - np.max(act[1:])); target_aw = (ea / np.sum(ea)) * target_stock

            p = prices_raw[t]; w = max(1e-4, cash + np.sum(shares * p))
            caw = (shares * p) / w; ccf = cash / w
            drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))
            if drift > 0.03:
                tv = abs(cash - w*target_cash) + np.sum(np.abs(shares*p - w*target_aw))
                net = max(1e-4, w - tv * 0.001)
                cash = net * target_cash; shares = (net * target_aw) / np.maximum(1e-4, p)

            nw = cash + np.sum(shares * p)
            peak = max(peak, nw)
            wealth_hist.append(nw)

            p_prev = prices_raw[t-1]
            obs_history.pop(0)
            obs_history.append(np.concatenate([p / prices_raw[30], np.log(p / np.maximum(1e-4, p_prev)), [cash/nw, np.clip((nw-peak)/peak, -1, 0)]]).astype(np.float32))

        return compute_metrics(wealth_hist)

    m_v6a = eval_v6_alpha(test_df)
    m_spy = compute_metrics(10000.0 * (test_df['SPY'].values / test_df['SPY'].values[0]))
    m_lstm = compute_metrics(evaluate_lstm_strategy(lstm_model, test_df))
    m_xgb = compute_metrics(evaluate_xgb_strategy(xgb_clf, test_df))
    m_rp = compute_metrics(evaluate_risk_parity(test_df))
    m_mom = compute_metrics(evaluate_momentum(test_df, top_k=3))
    m_sma = compute_metrics(evaluate_sma_crossover(test_df))

    print("Master SOTA Comparison Complete.", flush=True)

if __name__ == "__main__":
    build_master_comparison()
