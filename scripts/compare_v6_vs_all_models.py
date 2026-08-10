"""
Comprehensive Side-by-Side Comparison: RAI v6 vs All Baseline Models & AI Algorithms
Starting Capital: $10,000.00
"""
import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_v6_fast import DeepEndToEndTradingNet
from scripts.train_v5_dual_head import DualHeadGatedPolicy, RealMarketV5Env, metrics
from scripts.eval_vs_standard_ai import (
    train_lstm_model, evaluate_lstm_strategy,
    build_xgb_features, evaluate_xgb_strategy,
    evaluate_risk_parity, evaluate_momentum, evaluate_sma_crossover,
    GradientBoostingClassifier
)

def run_all_models_comparison():
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)

    # 1. Load RAI v6
    v6_model = DeepEndToEndTradingNet(history_len=30, features_per_step=22, action_dim=11)
    v6_path = "./data/v0.6_rl_checkpoints/rai_v6_fast.pt"
    if os.path.exists(v6_path):
        v6_model.load_state_dict(torch.load(v6_path))
        v6_model.eval()

    # 2. Load RAI v5
    v5_policy = DualHeadGatedPolicy(obs_dim=384, action_dim=11)
    v5_path = "./data/v0.5_rl_checkpoints/rai_v5_dual_head.pt"
    if os.path.exists(v5_path):
        v5_policy.load_state_dict(torch.load(v5_path))
        v5_policy.eval()

    # 3. Train Standard AI Models (LSTM & XGBoost) on 2010-2019
    lstm_model = train_lstm_model(train_df, lookback=20, epochs=100)
    X_tr, y_tr = build_xgb_features(train_df, lookback=20)
    xgb_clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_clf.fit(X_tr, y_tr)

    def eval_v6_df(df):
        prices_raw = df.values[:, :10]
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
            act = v6_model.get_action(flat_obs, deterministic=True)
            cl = np.clip(act[0], -5, 5)
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

        return wealth_hist

    def eval_v5_df(df):
        env = RealMarketV5Env(price_df=df, max_assets=10)
        obs, _ = env.reset()
        done = False
        while not done:
            act, probs = v5_policy.get_action(obs, deterministic=True)
            obs, _, done, _, _ = env.step((act, probs))
        return [10000.0] + env.log_wealth

    periods = [("2020-2024 Out-of-Sample", test_df), ("2010-2019 Historical", train_df)]

    print("=" * 120, flush=True)
    print("  COMPREHENSIVE BENCHMARK: RAI v6 vs ALL AI MODELS & BASELINES ($10,000 STARTING CAPITAL)", flush=True)
    print("=" * 120, flush=True)

    for pname, df in periods:
        print(f"\n  PERIOD: {pname}", flush=True)
        print(f"  {'-'*115}", flush=True)
        print(f"  {'Model / Strategy':<40} | {'Final Value':>12} | {'Net Profit ($)':>14} | {'Sharpe':>7} | {'Max DD (%)':>10} | Raw Prices Only?", flush=True)
        print(f"  {'-'*115}", flush=True)

        # 1. RAI v6
        eq_v6 = eval_v6_df(df)
        m_v6 = metrics(eq_v6)
        print(f"  {'🏆 Zero-Shot RAI v6 (Transformer)':<40} | ${m_v6['final']:>11,.2f} | ${m_v6['final']-10000:>+13,.2f} | {m_v6['sharpe']:>7.2f} | {m_v6['max_dd']:>9.2f}% | ✅ YES (End-to-End)")

        # 2. RAI v5
        eq_v5 = eval_v5_df(df)
        m_v5 = metrics(eq_v5)
        print(f"  {'🤖 Zero-Shot RAI v5 (Dual-Head Gated)':<40} | ${m_v5['final']:>11,.2f} | ${m_v5['final']-10000:>+13,.2f} | {m_v5['sharpe']:>7.2f} | {m_v5['max_dd']:>9.2f}% | ❌ Uses SMAs")

        # 3. SPY
        spy = df['SPY'].values
        eq_spy = 10000.0 * (spy / spy[0])
        m_spy = metrics(eq_spy)
        print(f"  {'SPY Buy & Hold (S&P 500)':<40} | ${m_spy['final']:>11,.2f} | ${m_spy['final']-10000:>+13,.2f} | {m_spy['sharpe']:>7.2f} | {m_spy['max_dd']:>9.2f}% | Real Market")

        # 4. LSTM
        eq_lstm = evaluate_lstm_strategy(lstm_model, df)
        m_lstm = metrics(eq_lstm)
        tag_lstm = "Trained on Real" if "2020" in pname else "⚠️ In-Sample"
        print(f"  {'LSTM Return Predictor (Deep Learning)':<40} | ${m_lstm['final']:>11,.2f} | ${m_lstm['final']-10000:>+13,.2f} | {m_lstm['sharpe']:>7.2f} | {m_lstm['max_dd']:>9.2f}% | {tag_lstm}")

        # 5. XGBoost
        eq_xgb = evaluate_xgb_strategy(xgb_clf, df)
        m_xgb = metrics(eq_xgb)
        tag_xgb = "Trained on Real" if "2020" in pname else "⚠️ Overfit"
        print(f"  {'XGBoost Classifier (Machine Learning)':<40} | ${m_xgb['final']:>11,.2f} | ${m_xgb['final']-10000:>+13,.2f} | {m_xgb['sharpe']:>7.2f} | {m_xgb['max_dd']:>9.2f}% | {tag_xgb}")

        # 6. Risk Parity
        eq_rp = evaluate_risk_parity(df)
        m_rp = metrics(eq_rp)
        print(f"  {'Risk Parity (Inverse Volatility)':<40} | ${m_rp['final']:>11,.2f} | ${m_rp['final']-10000:>+13,.2f} | {m_rp['sharpe']:>7.2f} | {m_rp['max_dd']:>9.2f}% | ❌ Uses Vol")

        # 7. Momentum
        eq_mom = evaluate_momentum(df, top_k=3)
        m_mom = metrics(eq_mom)
        print(f"  {'Momentum Factor (Top-3 Winners)':<40} | ${m_mom['final']:>11,.2f} | ${m_mom['final']-10000:>+13,.2f} | {m_mom['sharpe']:>7.2f} | {m_mom['max_dd']:>9.2f}% | ❌ Uses Returns")

        # 8. SMA Crossover
        eq_sma = evaluate_sma_crossover(df)
        m_sma = metrics(eq_sma)
        print(f"  {'SMA 50/200 Trend Following':<40} | ${m_sma['final']:>11,.2f} | ${m_sma['final']-10000:>+13,.2f} | {m_sma['sharpe']:>7.2f} | {m_sma['max_dd']:>9.2f}% | ❌ Uses SMAs")

        # 9. 60/40
        if 'TLT' in df.columns:
            tlt = df['TLT'].values
            eq_6040 = 10000.0 * (0.60 * (spy / spy[0]) + 0.40 * (tlt / tlt[0]))
            m_6040 = metrics(eq_6040)
            print(f"  {'60/40 Portfolio (SPY / TLT)':<40} | ${m_6040['final']:>11,.2f} | ${m_6040['final']-10000:>+13,.2f} | {m_6040['sharpe']:>7.2f} | {m_6040['max_dd']:>9.2f}% | Passive")

        print(f"  {'-'*115}", flush=True)

if __name__ == "__main__":
    run_all_models_comparison()
