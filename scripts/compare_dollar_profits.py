"""
Dollar Profit Comparison: RAI v5 vs Real Market vs Standard AI / ML Models
Starting Capital: $10,000.00
"""
import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_v5_dual_head import DualHeadGatedPolicy, RealMarketV5Env, metrics
from scripts.eval_vs_standard_ai import (
    train_lstm_model, evaluate_lstm_strategy,
    build_xgb_features, evaluate_xgb_strategy,
    evaluate_risk_parity, evaluate_momentum, evaluate_sma_crossover,
    GradientBoostingClassifier
)

def get_dollar_results():
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)

    # 1. RAI v5
    obs_dim = 384; action_dim = 11
    policy = DualHeadGatedPolicy(obs_dim=obs_dim, action_dim=action_dim)
    policy.load_state_dict(torch.load("./data/v0.5_rl_checkpoints/rai_v5_dual_head.pt"))
    policy.eval()

    def eval_v5_df(df):
        env = RealMarketV5Env(price_df=df, max_assets=10)
        obs, _ = env.reset()
        done = False
        while not done:
            act, probs = policy.get_action(obs, deterministic=True)
            obs, _, done, _, _ = env.step((act, probs))
        eq = [10000.0] + env.log_wealth
        return eq

    # Train LSTM & XGBoost on 2010-2019
    lstm_model = train_lstm_model(train_df, lookback=20, epochs=100)
    X_tr, y_tr = build_xgb_features(train_df, lookback=20)
    xgb_clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_clf.fit(X_tr, y_tr)

    periods = [("2020-2024 (Out-of-Sample)", test_df), ("2010-2019 (Historical)", train_df)]

    print("=" * 115, flush=True)
    print("  COMPLETE DOLLAR PROFIT COMPARISON: $10,000 INITIAL CAPITAL", flush=True)
    print("=" * 115, flush=True)

    for pname, df in periods:
        print(f"\n  PERIOD: {pname}", flush=True)
        print(f"  {'-'*110}", flush=True)
        print(f"  {'Model / Strategy':<38} | {'Final Value':>12} | {'Net Profit ($)':>14} | {'Return (%)':>10} | {'Max DD (%)':>10} | Data Used", flush=True)
        print(f"  {'-'*110}", flush=True)

        # SPY
        spy = df['SPY'].values
        eq_spy = 10000.0 * (spy / spy[0])
        m_spy = metrics(eq_spy)
        print(f"  {'SPY Buy & Hold (S&P 500)':<38} | ${m_spy['final']:>11,.2f} | ${m_spy['final']-10000:>+13,.2f} | {m_spy['return']:>+9.2f}% | {m_spy['max_dd']:>9.2f}% | Real Market")

        # RAI v5
        eq_v5 = eval_v5_df(df)
        m_v5 = metrics(eq_v5)
        print(f"  {'🤖 Zero-Shot RAI v5 (0% Real Data)':<38} | ${m_v5['final']:>11,.2f} | ${m_v5['final']-10000:>+13,.2f} | {m_v5['return']:>+9.2f}% | {m_v5['max_dd']:>9.2f}% | 0% Real Data")

        # XGBoost
        eq_xgb = evaluate_xgb_strategy(xgb_clf, df)
        m_xgb = metrics(eq_xgb)
        tag_xgb = "Trained on Real" if "2020" in pname else "⚠️ In-Sample Overfit"
        print(f"  {'XGBoost Direction Classifier (ML)':<38} | ${m_xgb['final']:>11,.2f} | ${m_xgb['final']-10000:>+13,.2f} | {m_xgb['return']:>+9.2f}% | {m_xgb['max_dd']:>9.2f}% | {tag_xgb}")

        # LSTM
        eq_lstm = evaluate_lstm_strategy(lstm_model, df)
        m_lstm = metrics(eq_lstm)
        tag_lstm = "Trained on Real" if "2020" in pname else "⚠️ In-Sample"
        print(f"  {'LSTM Return Predictor (Deep Learning)':<38} | ${m_lstm['final']:>11,.2f} | ${m_lstm['final']-10000:>+13,.2f} | {m_lstm['return']:>+9.2f}% | {m_lstm['max_dd']:>9.2f}% | {tag_lstm}")

        # Risk Parity
        eq_rp = evaluate_risk_parity(df)
        m_rp = metrics(eq_rp)
        print(f"  {'Risk Parity (Inverse Volatility)':<38} | ${m_rp['final']:>11,.2f} | ${m_rp['final']-10000:>+13,.2f} | {m_rp['return']:>+9.2f}% | {m_rp['max_dd']:>9.2f}% | Rule-Based")

        # Momentum
        eq_mom = evaluate_momentum(df, top_k=3)
        m_mom = metrics(eq_mom)
        print(f"  {'Momentum Factor (Top-3 Winners)':<38} | ${m_mom['final']:>11,.2f} | ${m_mom['final']-10000:>+13,.2f} | {m_mom['return']:>+9.2f}% | {m_mom['max_dd']:>9.2f}% | Rule-Based")

        # SMA
        eq_sma = evaluate_sma_crossover(df)
        m_sma = metrics(eq_sma)
        print(f"  {'SMA 50/200 Trend Following':<38} | ${m_sma['final']:>11,.2f} | ${m_sma['final']-10000:>+13,.2f} | {m_sma['return']:>+9.2f}% | {m_sma['max_dd']:>9.2f}% | Rule-Based")

        # Equal-Weight
        eq_ew = 10000.0 * np.mean(df.values / df.values[0], axis=1)
        m_ew = metrics(eq_ew)
        print(f"  {'Equal-Weight (1/N)':<38} | ${m_ew['final']:>11,.2f} | ${m_ew['final']-10000:>+13,.2f} | {m_ew['return']:>+9.2f}% | {m_ew['max_dd']:>9.2f}% | Passive")

        # 60/40
        if 'TLT' in df.columns:
            tlt = df['TLT'].values
            eq_6040 = 10000.0 * (0.60 * (spy / spy[0]) + 0.40 * (tlt / tlt[0]))
            m_6040 = metrics(eq_6040)
            print(f"  {'60/40 Portfolio (SPY / TLT)':<38} | ${m_6040['final']:>11,.2f} | ${m_6040['final']-10000:>+13,.2f} | {m_6040['return']:>+9.2f}% | {m_6040['max_dd']:>9.2f}% | Passive")

        print(f"  {'-'*110}", flush=True)

if __name__ == "__main__":
    get_dollar_results()
