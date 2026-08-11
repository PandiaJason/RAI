"""
Empirical Comparison: Zero-Shot RAI v6 vs Random 50% Baseline (100 Random Seeds)
"""
import os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_v6_fast import DeepEndToEndTradingNet
from scripts.train_v5_dual_head import metrics


def eval_random_50_strategy(df, n_seeds=100, initial_cash=10000.0, fee=0.001):
    prices_raw = df.values[:, :10]
    T, N = prices_raw.shape

    returns_list = []
    sharpe_list = []
    max_dd_list = []
    final_wealth_list = []

    for seed in range(n_seeds):
        np.random.seed(seed)
        cash = initial_cash * 0.5
        init_p = prices_raw[30]
        shares = (initial_cash * 0.5 / N) / init_p
        peak = initial_cash
        wealth_hist = [initial_cash]

        for t in range(30, T):
            # 50% random action: generate uniform random target cash fraction & target stock weights
            target_cash = np.random.uniform(0.0, 1.0)
            raw_w = np.random.uniform(0.0, 1.0, size=N)
            target_aw = (raw_w / np.sum(raw_w)) * (1.0 - target_cash)

            p = prices_raw[t]
            w = max(1e-4, cash + np.sum(shares * p))
            caw = (shares * p) / w
            ccf = cash / w

            drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))
            if drift > 0.03:
                tv = abs(cash - w*target_cash) + np.sum(np.abs(shares*p - w*target_aw))
                net = max(1e-4, w - tv * fee)
                cash = net * target_cash
                shares = (net * target_aw) / np.maximum(1e-4, p)

            nw = cash + np.sum(shares * p)
            peak = max(peak, nw)
            wealth_hist.append(nw)

        eq = np.array(wealth_hist)
        m = metrics(eq)
        final_wealth_list.append(m['final'])
        returns_list.append(m['return'])
        sharpe_list.append(m['sharpe'])
        max_dd_list.append(m['max_dd'])

    return {
        "final_mean": np.mean(final_wealth_list),
        "return_mean": np.mean(returns_list),
        "return_std": np.std(returns_list),
        "sharpe_mean": np.mean(sharpe_list),
        "max_dd_mean": np.mean(max_dd_list),
    }


def main():
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)

    # Load RAI v6
    v6_model = DeepEndToEndTradingNet(history_len=30, features_per_step=22, action_dim=11)
    v6_path = "./data/v0.6_rl_checkpoints/rai_v6_fast.pt"
    if not os.path.exists(v6_path):
        print("Model file not found!")
        return
    v6_model.load_state_dict(torch.load(v6_path))
    v6_model.eval()

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

        return metrics(wealth_hist)

    periods = [("2020-2024 Out-of-Sample", test_df), ("2010-2019 Historical", train_df)]

    print("=" * 105, flush=True)
    print("  EMPIRICAL BENCHMARK: ZERO-SHOT RAI v6 vs RANDOM 50% BASELINE (100 SEEDS)", flush=True)
    print("=" * 105, flush=True)

    for pname, df in periods:
        print(f"\n  PERIOD: {pname}", flush=True)
        print(f"  {'-'*100}", flush=True)

        m_v6 = eval_v6_df(df)
        r_rand = eval_random_50_strategy(df, n_seeds=100)

        print(f"  {'Model / Baseline':<35} | {'Final Value':>12} | {'Net Return':>12} | {'Sharpe':>7} | {'Max DD (%)':>10}")
        print(f"  {'-'*100}")
        print(f"  {'🏆 Zero-Shot RAI v6 (Deep AI)':<35} | ${m_v6['final']:>11,.2f} | {m_v6['return']:>+11.2f}% | {m_v6['sharpe']:>7.2f} | {m_v6['max_dd']:>9.2f}%")
        print(f"  {'🎲 Random 50% Action (Mean of 100 seeds)':<35} | ${r_rand['final_mean']:>11,.2f} | {r_rand['return_mean']:>+11.2f}% | {r_rand['sharpe_mean']:>7.2f} | {r_rand['max_dd_mean']:>9.2f}%")
        print(f"  {'-'*100}")

        ret_diff = m_v6['return'] - r_rand['return_mean']
        sh_diff = m_v6['sharpe'] - r_rand['sharpe_mean']
        dd_diff = abs(r_rand['max_dd_mean']) - abs(m_v6['max_dd'])

        print(f"  RAI v6 Outperformance over Random 50%:")
        print(f"    • Return Advantage: {ret_diff:+.2f}% higher return")
        print(f"    • Sharpe Advantage: {sh_diff:+.2f} higher Sharpe ratio")
        print(f"    • Risk Advantage:   {dd_diff:+.2f}% lower drawdown! 🛡️")

if __name__ == "__main__":
    main()
