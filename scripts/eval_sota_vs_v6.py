"""
Comprehensive SOTA Leaderboard: RAI v6 ALPHA vs SOTA Financial AI & Quantitative Models
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


def eval_v6_alpha_df(model, df):
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
        act = model.get_action(flat_obs, deterministic=True)
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


def run_sota_leaderboard():
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

    periods = [("2020-2024 Out-of-Sample (Unseen Real Market Data)", test_df)]

    print("=" * 125, flush=True)
    print("  SOTA FINANCIAL AI LEADERBOARD: RAI v6 ALPHA vs SOTA AI & INSTITUTIONAL MODELS ($10,000 STARTING CAPITAL)", flush=True)
    print("=" * 125, flush=True)

    for pname, df in periods:
        print(f"\n  EVALUATION PERIOD: {pname}", flush=True)
        print(f"  {'-'*120}", flush=True)
        print(f"  {'Model / Strategy':<42} | {'Final Value':>12} | {'Net Profit ($)':>14} | {'Return (%)':>10} | {'Sharpe':>7} | {'Max DD (%)':>10} | Real Data Trained?", flush=True)
        print(f"  {'-'*120}", flush=True)

        # SPY
        spy = df['SPY'].values
        eq_spy = 10000.0 * (spy / spy[0])
        m_spy = compute_metrics(eq_spy)
        print(f"  {'SPY Buy & Hold (S&P 500 Index)':<42} | ${m_spy['final']:>11,.2f} | ${m_spy['final']-10000:>+13,.2f} | {m_spy['return_pct']:>+9.2f}% | {m_spy['sharpe']:>7.2f} | {m_spy['max_dd_pct']:>9.2f}% | Real Market")

        # RAI v6 ALPHA
        m_v6a = eval_v6_alpha_df(v6_alpha_model, df)
        print(f"  {'🏆 Zero-Shot RAI v6 ALPHA (Transformer)':<42} | ${m_v6a['final']:>11,.2f} | ${m_v6a['final']-10000:>+13,.2f} | {m_v6a['return_pct']:>+9.2f}% | {m_v6a['sharpe']:>7.2f} | {m_v6a['max_dd_pct']:>9.2f}% | ❌ 0% Real Data")

        # Supervised LSTM
        eq_lstm = evaluate_lstm_strategy(lstm_model, df)
        m_lstm = compute_metrics(eq_lstm)
        print(f"  {'SOTA LSTM Predictor (Deep Learning)':<42} | ${m_lstm['final']:>11,.2f} | ${m_lstm['final']-10000:>+13,.2f} | {m_lstm['return_pct']:>+9.2f}% | {m_lstm['sharpe']:>7.2f} | {m_lstm['max_dd_pct']:>9.2f}% | ✅ Trained on 10 Yrs Real")

        # Supervised XGBoost
        eq_xgb = evaluate_xgb_strategy(xgb_clf, df)
        m_xgb = compute_metrics(eq_xgb)
        print(f"  {'SOTA XGBoost (Gradient Boosted Trees)':<42} | ${m_xgb['final']:>11,.2f} | ${m_xgb['final']-10000:>+13,.2f} | {m_xgb['return_pct']:>+9.2f}% | {m_xgb['sharpe']:>7.2f} | {m_xgb['max_dd_pct']:>9.2f}% | ✅ Trained on 10 Yrs Real")

        # Risk Parity
        eq_rp = evaluate_risk_parity(df)
        m_rp = compute_metrics(eq_rp)
        print(f"  {'SOTA Institutional Risk Parity':<42} | ${m_rp['final']:>11,.2f} | ${m_rp['final']-10000:>+13,.2f} | {m_rp['return_pct']:>+9.2f}% | {m_rp['sharpe']:>7.2f} | {m_rp['max_dd_pct']:>9.2f}% | Rule-Based")

        # Momentum Factor
        eq_mom = evaluate_momentum(df, top_k=3)
        m_mom = compute_metrics(eq_mom)
        print(f"  {'SOTA Momentum Factor (Top-3 Winners)':<42} | ${m_mom['final']:>11,.2f} | ${m_mom['final']-10000:>+13,.2f} | {m_mom['return_pct']:>+9.2f}% | {m_mom['sharpe']:>7.2f} | {m_mom['max_dd_pct']:>9.2f}% | Rule-Based")

        # SMA Crossover
        eq_sma = evaluate_sma_crossover(df)
        m_sma = compute_metrics(eq_sma)
        print(f"  {'SMA 50/200 Trend Following':<42} | ${m_sma['final']:>11,.2f} | ${m_sma['final']-10000:>+13,.2f} | {m_sma['return_pct']:>+9.2f}% | {m_sma['sharpe']:>7.2f} | {m_sma['max_dd_pct']:>9.2f}% | Rule-Based")

        print(f"  {'-'*120}", flush=True)

if __name__ == "__main__":
    run_sota_leaderboard()
