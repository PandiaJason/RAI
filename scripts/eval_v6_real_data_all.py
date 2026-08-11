"""
RAI v6 Full Real-Data Evaluation Across Multiple Asset Classes & Periods
(Raw Prices Only - 0% Hand-Crafted Features)
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_v6_fast import DeepEndToEndTradingNet


def download_data(tickers, start, end):
    print(f"  Downloading real data for {tickers} ({start} to {end})...", flush=True)
    df = yf.download(tickers, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    df = df.dropna()
    return df


def eval_v6_on_real_df(name, df, model):
    prices_raw = df.values
    T, N = prices_raw.shape
    num_assets = min(N, 10)
    prices_sub = prices_raw[:, :num_assets]

    # Set up portfolio
    cash = 5000.0
    init_p = prices_sub[30]
    shares = (5000.0 / num_assets) / init_p
    wealth_hist = [10000.0]
    cash_hist = []

    # Build initial 30-day raw history
    obs_history = []
    for t in range(30):
        p = prices_sub[t]
        p_prev = prices_sub[max(0, t-1)]
        norm_p = p / prices_sub[30]
        log_r = np.log(p / np.maximum(1e-4, p_prev))

        # Pad to 10 assets if num_assets < 10
        if num_assets < 10:
            norm_p = np.pad(norm_p, (0, 10 - num_assets), constant_values=1.0)
            log_r = np.pad(log_r, (0, 10 - num_assets), constant_values=0.0)

        step_obs = np.concatenate([norm_p, log_r, [0.5, 0.0]]).astype(np.float32)
        obs_history.append(step_obs)

    peak = 10000.0
    for t in range(30, T):
        flat_obs = np.concatenate(obs_history).astype(np.float32)
        act = model.get_action(flat_obs, deterministic=True)

        cl = np.clip(act[0], -5, 5)
        target_cash = 1.0 / (1.0 + np.exp(-cl))
        target_stock = 1.0 - target_cash

        # Allocation among real assets
        al = act[1:1+num_assets]
        ea = np.exp(al - np.max(al))
        target_aw = (ea / np.sum(ea)) * target_stock

        p = prices_sub[t]
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
        cash_hist.append(target_cash)

        # Update obs history
        p_prev = prices_sub[t-1]
        norm_p = p / prices_sub[30]
        log_r = np.log(p / np.maximum(1e-4, p_prev))
        if num_assets < 10:
            norm_p = np.pad(norm_p, (0, 10 - num_assets), constant_values=1.0)
            log_r = np.pad(log_r, (0, 10 - num_assets), constant_values=0.0)

        dd = np.clip((nw - peak) / max(1e-4, peak), -1, 0)
        step_obs = np.concatenate([norm_p, log_r, [cash / nw, dd]]).astype(np.float32)

        obs_history.pop(0)
        obs_history.append(step_obs)

    # Compute metrics
    eq = np.array(wealth_hist)
    cf = np.array(cash_hist)
    rets = (eq[1:] - eq[:-1]) / np.maximum(1e-8, eq[:-1])

    tot_ret = (eq[-1] / eq[0] - 1) * 100
    profit = eq[-1] - eq[0]
    vol = np.std(rets) * np.sqrt(252) * 100
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 1e-8 else 0.0
    pk = np.maximum.accumulate(eq)
    mdd = np.min((eq - pk) / pk) * 100

    # Baseline comparison (Equal-weight buy & hold)
    ew_eq = 10000.0 * np.mean(prices_sub / prices_sub[30], axis=1)
    ew_rets = (ew_eq[1:] - ew_eq[:-1]) / np.maximum(1e-8, ew_eq[:-1])
    ew_ret = (ew_eq[-1] / ew_eq[0] - 1) * 100
    ew_pk = np.maximum.accumulate(ew_eq)
    ew_mdd = np.min((ew_eq - ew_pk) / ew_pk) * 100

    print(f"\n{'='*85}", flush=True)
    print(f"  REAL MARKET EVALUATION: {name} ({len(df)} trading days)", flush=True)
    print(f"{'='*85}", flush=True)
    print(f"  Initial Capital:  $10,000.00", flush=True)
    print(f"  Final Portfolio:  ${eq[-1]:,.2f}", flush=True)
    print(f"  Net Profit ($):   ${profit:+,.2f}", flush=True)
    print(f"  Total Return (%): {tot_ret:+.2f}%", flush=True)
    print(f"  Volatility (%):   {vol:.2f}%", flush=True)
    print(f"  Sharpe Ratio:     {sharpe:.2f}", flush=True)
    print(f"  Max Drawdown (%): {mdd:.2f}%", flush=True)
    print(f"  Cash Min / Max:   {np.min(cf)*100:.1f}% / {np.max(cf)*100:.1f}% (Range: {(np.max(cf)-np.min(cf))*100:.1f}%)", flush=True)
    print(f"  ---------------------------------------------------------", flush=True)
    print(f"  Baseline Buy & Hold Return:   {ew_ret:+.2f}%", flush=True)
    print(f"  Baseline Max Drawdown:        {ew_mdd:.2f}%", flush=True)
    print(f"  Drawdown Shielding:           {abs(ew_mdd) - abs(mdd):+.2f}% less drawdown pain! 🛡️", flush=True)


def main():
    model = DeepEndToEndTradingNet(history_len=30, features_per_step=22, action_dim=11)
    path = "./data/v0.6_rl_checkpoints/rai_v6_fast.pt"
    if not os.path.exists(path):
        print(f"Model {path} not found!")
        return
    model.load_state_dict(torch.load(path))
    model.eval()

    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)

    print("=" * 85, flush=True)
    print("  RAI v6 REAL DATA COMPREHENSIVE EVALUATION (100% RAW PRICES)", flush=True)
    print("=" * 85, flush=True)

    # 1. Out-of-sample US Market
    eval_v6_on_real_df("US Equities Basket (2020-2024 Out-of-Sample)", test_df, model)

    # 2. Historical US Market
    eval_v6_on_real_df("US Equities Basket (2010-2019 Historical)", train_df, model)

    # 3. Live Downloaded Baskets
    baskets = [
        ("Tech Giants (AAPL, MSFT, NVDA, AMZN, GOOGL)", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"], "2015-01-01", "2024-01-01"),
        ("Crypto Assets (BTC-USD, ETH-USD)", ["BTC-USD", "ETH-USD"], "2018-01-01", "2024-01-01"),
        ("2007-2009 Global Financial Crisis (SPY, QQQ, GLD)", ["SPY", "QQQ", "GLD"], "2007-01-01", "2010-01-01"),
    ]

    for name, tickers, start, end in baskets:
        try:
            df = download_data(tickers, start, end)
            if len(df) > 50:
                eval_v6_on_real_df(name, df, model)
        except Exception as e:
            print(f"Error loading {name}: {e}")

if __name__ == "__main__":
    main()
