"""v4 Truth Test: Same 7 tests that exposed v3 as fake."""
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from scripts.eval_v4_real import RealMarketV4Env
from scipy.stats import pearsonr

model = PPO.load("./data/v0.4_rl_checkpoints/rai_v4_fast")
df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)

env = RealMarketV4Env(price_df=df, max_assets=10)
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)

cash_w = np.array(env.log_target_cash)
stock_w = np.array(env.log_target_stock)
raw = np.array(env.log_raw_action)
rebal = np.array(env.log_rebalanced)
wealth = np.array(env.log_wealth)
T = len(cash_w)

print("=" * 70, flush=True)
print("  RAI v4: REAL AI OR NOT? (Same 7 Tests That Exposed v3)", flush=True)
print("=" * 70, flush=True)

# TEST 1: Action variance
std = np.mean(np.std(raw, axis=0))
print(f"\n  TEST 1: Action std = {std:.6f}", flush=True)
print(f"    v3 was: 0.000002  |  v4 is: {std:.6f}  |  {'✅ Better' if std > 0.001 else '❌ Same'}", flush=True)

# TEST 2: Cash weight range
print(f"\n  TEST 2: Cash weight range", flush=True)
print(f"    Min: {np.min(cash_w):.4f}  Max: {np.max(cash_w):.4f}  Range: {np.max(cash_w)-np.min(cash_w):.4f}", flush=True)
print(f"    v3 was: 0.0000 range  |  v4 is: {np.max(cash_w)-np.min(cash_w):.4f} range", flush=True)
if np.max(cash_w) - np.min(cash_w) > 0.05:
    print(f"    ✅ Meaningful variation (>5%)", flush=True)
elif np.max(cash_w) - np.min(cash_w) > 0.01:
    print(f"    ⚠️ Small variation (1-5%) — adapting but barely", flush=True)
else:
    print(f"    ❌ Negligible variation (<1%) — effectively constant", flush=True)

# TEST 3: Rebalancing
print(f"\n  TEST 3: Trading frequency", flush=True)
print(f"    Rebalance days: {np.sum(rebal)}/{T} ({np.sum(rebal)/T*100:.1f}%)", flush=True)

# TEST 4: COVID crash behavior
print(f"\n  TEST 4: COVID Crash Behavior (Feb-Mar 2020)", flush=True)
dates = df.index
crash_mask = [(str(d)[:7] in ['2020-02', '2020-03']) for d in dates]
pre_mask = [(str(d)[:7] in ['2020-01']) for d in dates]
post_mask = [(str(d)[:7] in ['2020-04', '2020-05']) for d in dates]

offset = env.history_len + 20
crash_idx = [i - offset for i in range(len(dates)) if crash_mask[i] and i >= offset and i - offset < T]
pre_idx = [i - offset for i in range(len(dates)) if pre_mask[i] and i >= offset and i - offset < T]
post_idx = [i - offset for i in range(len(dates)) if post_mask[i] and i >= offset and i - offset < T]

if crash_idx and pre_idx:
    pre_cash = np.mean([cash_w[i] for i in pre_idx if i < T])
    crash_cash = np.mean([cash_w[i] for i in crash_idx if i < T])
    post_cash = np.mean([cash_w[i] for i in post_idx if i < T]) if post_idx else 0
    print(f"    Before crash: {pre_cash:.4f} cash", flush=True)
    print(f"    During crash: {crash_cash:.4f} cash", flush=True)
    print(f"    After crash:  {post_cash:.4f} cash", flush=True)
    if crash_cash > pre_cash + 0.02:
        print(f"    ✅ Increased cash during crash (defensive)", flush=True)
    elif crash_cash > pre_cash + 0.005:
        print(f"    ⚠️ Slightly increased cash (weak defense)", flush=True)
    else:
        print(f"    ❌ Did NOT increase cash during crash", flush=True)

# TEST 5: Compare to buy-and-hold
bh_cash = 5000.0
prices = df.values[:, :10]
init_p = prices[offset]
bh_shares = (5000.0 / 10) / init_p
bh_final = bh_cash + np.sum(bh_shares * prices[-1])
v4_final = wealth[-1]
diff = (v4_final / 10000 - 1)*100 - (bh_final / 10000 - 1)*100
print(f"\n  TEST 5: vs Buy-and-Hold (same starting allocation)", flush=True)
print(f"    v4 return:       {(v4_final/10000-1)*100:+.2f}%", flush=True)
print(f"    Buy-hold return: {(bh_final/10000-1)*100:+.2f}%", flush=True)
print(f"    Difference:      {diff:+.2f}%", flush=True)

# TEST 6: Correlation with market
print(f"\n  TEST 6: Do actions respond to market conditions?", flush=True)
spy = df['SPY'].values[offset:-1]
spy_rets = (spy[1:] - spy[:-1]) / spy[:-1]
min_len = min(len(spy_rets), T-1)
if min_len > 100:
    spy_roll = pd.Series(spy_rets[:min_len]).rolling(20).mean().dropna().values
    cash_aligned = cash_w[20:20+len(spy_roll)]
    if len(spy_roll) == len(cash_aligned):
        corr, pval = pearsonr(spy_roll, cash_aligned)
        print(f"    Corr(20d SPY return, cash weight): {corr:.4f}  p={pval:.4f}", flush=True)
        if corr < -0.1 and pval < 0.05:
            print(f"    ✅ Goes defensive when market drops (smart)", flush=True)
        elif corr > 0.1 and pval < 0.05:
            print(f"    ⚠️ Increases cash when market rises (wrong)", flush=True)
        else:
            print(f"    ❌ No correlation — ignores market", flush=True)

# FINAL VERDICT
print(f"\n{'='*70}", flush=True)
print(f"  FINAL VERDICT: Is RAI v4 Real AI?", flush=True)
print(f"{'='*70}", flush=True)

is_diverse = std > 0.001
cash_varies = (np.max(cash_w) - np.min(cash_w)) > 0.01
crash_defensive = crash_idx and pre_idx and crash_cash > pre_cash + 0.005
adds_value = diff > 3.0

score = sum([is_diverse, cash_varies, crash_defensive, adds_value])

print(f"\n  Scorecard:", flush=True)
print(f"    Actions non-constant:      {'✅' if is_diverse else '❌'} (std={std:.6f})", flush=True)
print(f"    Cash weight varies >1%:    {'✅' if cash_varies else '❌'} (range={np.max(cash_w)-np.min(cash_w):.4f})", flush=True)
print(f"    Defensive in crash:        {'✅' if crash_defensive else '❌'}", flush=True)
print(f"    Adds value vs buy-hold:    {'✅' if adds_value else '❌'} ({diff:+.2f}%)", flush=True)
print(f"\n  Score: {score}/4", flush=True)

if score >= 3:
    print(f"  ✅ YES — RAI v4 is functioning as real AI.", flush=True)
elif score >= 2:
    print(f"  ⚠️ PARTIALLY — v4 shows signs of AI but adaptation is weak.", flush=True)
else:
    print(f"  ❌ NO — v4 is not meaningfully different from a static rule.", flush=True)
