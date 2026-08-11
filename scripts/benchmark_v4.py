"""RAI v4 Full Benchmark vs All Models"""
import os, sys, numpy as np, pandas as pd
from stable_baselines3 import PPO
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.eval_v4_real import RealMarketV4Env, compute_metrics

def eval_v4(model, df):
    env = RealMarketV4Env(price_df=df, max_assets=10)
    obs, _ = env.reset()
    eq = [10000.0]; done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        eq.append(info["portfolio_value"])
    return eq

def eval_risk_parity(df, lookback=60):
    prices = df.values; T, N = prices.shape
    eq = [10000.0]
    for t in range(lookback+1, T):
        w = prices[t-lookback:t]; r = (w[1:]-w[:-1])/np.maximum(1e-6,w[:-1])
        v = np.std(r, axis=0); iv = 1.0/np.maximum(0.001, v); wt = iv/np.sum(iv)
        s = (eq[-1]*wt)/np.maximum(1e-6, prices[t-1])
        eq.append(np.sum(s*prices[t]))
    return eq

def eval_momentum(df, lookback=60, top_k=3):
    prices = df.values; T, N = prices.shape; top_k = min(top_k, N)
    eq = [10000.0]
    for t in range(lookback+1, T):
        mom = prices[t-1]/prices[t-lookback]-1.0
        top = np.argsort(mom)[-top_k:]
        s = np.zeros(N); s[top] = (eq[-1]/top_k)/np.maximum(1e-6, prices[t-1, top])
        eq.append(np.sum(s*prices[t]))
    return eq

def eval_sma(df, fast=50, slow=200):
    prices = df.values; T, N = prices.shape
    eq = [10000.0]; cash = 10000.0; shares = np.zeros(N)
    for t in range(slow+1, T):
        sf = np.mean(prices[t-fast:t], axis=0); ss = np.mean(prices[t-slow:t], axis=0)
        bull = sf > ss; nb = max(1, np.sum(bull))
        w = cash + np.sum(shares*prices[t-1])
        if np.sum(bull) > N/2:
            shares = np.zeros(N); shares[bull] = (w/nb)/np.maximum(1e-6, prices[t-1,bull]); cash = 0
        else:
            cash = w; shares = np.zeros(N)
        eq.append(cash + np.sum(shares*prices[t]))
    return eq

def row(name, m, tag=""):
    print(f"  {name:<40} | ${m['final']:<10,.2f} | {m['return_pct']:>+8.2f}% | {m['vol_pct']:>6.2f}% | {m['sharpe']:>5.2f} | {m['max_dd_pct']:>7.2f}% | {tag}", flush=True)

def run_period(label, df, v4_model):
    print(f"\n{'='*110}", flush=True)
    print(f"  {label} ({len(df)} trading days)", flush=True)
    print(f"{'='*110}", flush=True)
    print(f"  {'Model':<40} | {'Final':>10} | {'Return':>9} | {'Vol':>7} | {'Sharpe':>5} | {'Max DD':>8} | Notes", flush=True)
    print(f"  {'-'*105}", flush=True)

    spy = df['SPY'].values; eq_spy = 10000*(spy/spy[0])
    row("SPY Buy & Hold", compute_metrics(eq_spy), "Passive")

    eq_v4 = eval_v4(v4_model, df)
    row("🤖 RAI v4 (Zero-Shot, 0% Real Data)", compute_metrics(eq_v4), "0% Real Data")

    eq_rp = eval_risk_parity(df)
    row("Risk Parity (Inverse Vol)", compute_metrics(eq_rp), "Rule-Based")

    eq_mom = eval_momentum(df, top_k=3)
    row("Momentum Factor (Top-3)", compute_metrics(eq_mom), "Rule-Based")

    eq_sma = eval_sma(df)
    row("SMA 50/200 Trend Following", compute_metrics(eq_sma), "Rule-Based")

    eq_ew = 10000*np.mean(df.values/df.values[0], axis=1)
    row("Equal-Weight (1/N)", compute_metrics(eq_ew), "Passive")

    if 'TLT' in df.columns:
        tlt = df['TLT'].values
        eq_6040 = 10000*(0.60*(spy/spy[0]) + 0.40*(tlt/tlt[0]))
        row("60/40 Portfolio", compute_metrics(eq_6040), "Passive")

    print(f"  {'-'*105}", flush=True)

def main():
    print("="*110, flush=True)
    print("  RAI v4 FULL BENCHMARK", flush=True)
    print("="*110, flush=True)

    v4 = PPO.load("./data/v0.4_rl_checkpoints/rai_v4_fast")
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)

    run_period("OUT-OF-SAMPLE: 2020-2024", test_df, v4)
    run_period("HISTORICAL: 2010-2019", train_df, v4)

if __name__ == "__main__":
    main()
