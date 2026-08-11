"""
RAI v5 Deep Diagnostic & Truth Test
=====================================
Performs the 7 rigorous tests on RAI v5 (Dual-Head Gated Net) on real market data.
"""
import os, sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_v5_dual_head import DualHeadGatedPolicy, RealMarketV5Env, metrics

def main():
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)

    obs_dim = 384
    action_dim = 11
    policy = DualHeadGatedPolicy(obs_dim=obs_dim, action_dim=action_dim)
    policy.load_state_dict(torch.load("./data/v0.5_rl_checkpoints/rai_v5_dual_head.pt"))
    policy.eval()

    print("=" * 80, flush=True)
    print("  RAI v5 DEEP DIAGNOSTIC & TRUTH TEST", flush=True)
    print("=" * 80, flush=True)

    for label, df in [("2020-2024 (Out-of-Sample)", test_df), ("2010-2019 (Historical)", train_df)]:
        print(f"\n{'-'*80}", flush=True)
        print(f"  PERIOD: {label}", flush=True)
        print(f"{'-'*80}", flush=True)

        env = RealMarketV5Env(price_df=df, max_assets=10)
        obs, _ = env.reset()
        done = False

        actions, probs_list = [], []
        while not done:
            act, probs = policy.get_action(obs, deterministic=True)
            actions.append(act)
            probs_list.append(probs)
            obs, _, done, _, _ = env.step((act, probs))

        cf = np.array(env.log_cash_frac)
        wealth = np.array(env.log_wealth)
        probs_arr = np.array(probs_list)
        acts_arr = np.array(actions)

        # Metric 1: Cash Fraction Range & Std
        cf_range = np.max(cf) - np.min(cf)
        cf_std = np.std(cf)
        act_std = np.mean(np.std(acts_arr, axis=0))

        print(f"  [1] Action Output Variation:   std = {act_std:.4f}  {'✅ DIVERSE' if act_std > 0.05 else '❌ STATIC'}")
        print(f"  [2] Cash Weight Range:         min = {np.min(cf):.4f}, max = {np.max(cf):.4f} (Range: {cf_range*100:.1f}%)  {'✅ REAL SWING' if cf_range > 0.3 else '❌ MINIMAL'}")
        print(f"  [3] Cash Weight Std:           std = {cf_std:.4f}")

        # Metric 2: Regime Probabilities
        p_bull = np.mean(probs_arr[:, 0]) * 100
        p_bear = np.mean(probs_arr[:, 1]) * 100
        p_side = np.mean(probs_arr[:, 2]) * 100
        print(f"  [4] Regime Classification:    Bull={p_bull:.1f}%, Bear={p_bear:.1f}%, Sideways={p_side:.1f}%")

        # Metric 3: COVID Crash Defense (if 2020-2024)
        if "2020" in label:
            dates = df.index
            offset = env.history_len + 20
            pre_idx = [i-offset for i, d in enumerate(dates) if str(d)[:7] == '2020-01' and i>=offset and i-offset<len(cf)]
            crash_idx = [i-offset for i, d in enumerate(dates) if str(d)[:7] in ['2020-02', '2020-03'] and i>=offset and i-offset<len(cf)]
            post_idx = [i-offset for i, d in enumerate(dates) if str(d)[:7] in ['2020-05', '2020-06'] and i>=offset and i-offset<len(cf)]

            if pre_idx and crash_idx:
                pre_c = np.mean(cf[pre_idx])
                crash_c = np.mean(cf[crash_idx])
                post_c = np.mean(cf[post_idx]) if post_idx else 0.0

                print(f"  [5] COVID Crash Adaptation:")
                print(f"      Pre-Crash (Jan 2020):     {pre_c*100:.1f}% Cash")
                print(f"      During Crash (Feb-Mar):  {crash_c*100:.1f}% Cash")
                print(f"      Post-Crash (May-Jun):     {post_c*100:.1f}% Cash")
                
                if crash_c > pre_c + 0.10:
                    print(f"      VERDICT: ✅ STRONGLY DEFENSIVE (+{(crash_c-pre_c)*100:.1f}% cash during crash!)")
                elif crash_c > pre_c + 0.02:
                    print(f"      VERDICT: ⚠️ MILDLY DEFENSIVE")
                else:
                    print(f"      VERDICT: ❌ NOT DEFENSIVE")

        # Metric 4: Market Correlation
        spy = df['SPY'].values[env.history_len+20:-1]
        spy_rets = (spy[1:] - spy[:-1]) / spy[:-1]
        min_l = min(len(spy_rets), len(cf)-1)
        if min_l > 50:
            spy_20d = pd.Series(spy_rets[:min_l]).rolling(20).mean().dropna().values
            c_align = cf[20:20+len(spy_20d)]
            if len(spy_20d) == len(c_align):
                r_val, p_val = pearsonr(spy_20d, c_align)
                print(f"  [6] Market Return Correlation: r = {r_val:.4f} (p = {p_val:.4f})")
                if r_val < -0.1 and p_val < 0.05:
                    print(f"      VERDICT: ✅ ADAPTIVE (Increases cash when market returns fall)")
                else:
                    print(f"      VERDICT: ℹ️ Neutral or complex correlation")

        # Metric 5: Performance vs SPY
        eq = [10000.0] + list(wealth)
        m = metrics(eq)
        spy_eq = 10000.0 * (df['SPY'].values / df['SPY'].values[0])
        m_spy = metrics(spy_eq)
        print(f"  [7] Max Drawdown Comparison:")
        print(f"      RAI v5 Max DD:  {m['max_dd']:.2f}%")
        print(f"      SPY Max DD:     {m_spy['max_dd']:.2f}%")
        print(f"      Drawdown Reduction: {abs(m_spy['max_dd']) - abs(m['max_dd']):.2f}% less pain! 🛡️")

if __name__ == "__main__":
    main()
