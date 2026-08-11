"""
DIAGNOSTIC: Is the MLP Actually Doing Anything?
=================================================
This script answers:
1. Does the MLP change its weights over time, or stay static?
2. Is it just mimicking buy-and-hold?
3. What is it actually outputting?
4. Would a "do nothing" strategy produce the same result?
"""
import os, sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


class DiagnosticEnv(gym.Env):
    """Same as v3 eval env but records all actions and weights."""
    def __init__(self, price_df, initial_cash=10000.0, history_len=32,
                 max_resources=20, transaction_fee=0.001, rebalance_threshold=0.03):
        super().__init__()
        self.price_df = price_df.copy()
        self.prices_matrix = self.price_df.values
        self.num_steps, self.num_resources = self.prices_matrix.shape
        self.max_resources = max_resources
        self.initial_cash = initial_cash
        self.history_len = history_len
        self.transaction_fee = transaction_fee
        self.rebalance_threshold = rebalance_threshold
        T, N = self.prices_matrix.shape
        self.sma50 = np.zeros_like(self.prices_matrix)
        self.sma200 = np.zeros_like(self.prices_matrix)
        self.vol20 = np.zeros_like(self.prices_matrix)
        for t in range(T):
            self.sma50[t] = np.mean(self.prices_matrix[max(0,t-50):t+1], axis=0)
            self.sma200[t] = np.mean(self.prices_matrix[max(0,t-200):t+1], axis=0)
            if t > 1:
                sub_p = self.prices_matrix[max(0,t-20):t+1]
                r = (sub_p[1:] - sub_p[:-1]) / np.maximum(1e-4, sub_p[:-1])
                self.vol20[t] = np.std(r, axis=0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(max_resources+1,), dtype=np.float32)
        self.single_obs_dim = 1 + max_resources * 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(history_len * self.single_obs_dim,), dtype=np.float32)
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.history_len
        self.cash = self.initial_cash * 0.50
        prices = self.prices_matrix[self.current_step]
        self.shares = (self.initial_cash * 0.50 / self.num_resources) / prices
        self.obs_history = [self._get_single_obs() for _ in range(self.history_len)]
        self.rebalance_count = 0
        # DIAGNOSTIC LOGS
        self.log_cash_weight = []
        self.log_target_cash = []
        self.log_target_stock = []
        self.log_rebalanced = []
        self.log_wealth = []
        self.log_raw_action = []
        return self._get_obs(), {}
    
    def _get_portfolio_value(self):
        return self.cash + np.sum(self.shares * self.prices_matrix[self.current_step])
    
    def _get_single_obs(self):
        prices = self.prices_matrix[self.current_step]
        wealth = max(1e-4, self._get_portfolio_value())
        w_cash = np.array([self.cash / wealth], dtype=np.float32)
        w_a = np.zeros(self.max_resources, dtype=np.float32)
        w_a[:self.num_resources] = (self.shares * prices) / wealth
        np_ = np.ones(self.max_resources, dtype=np.float32)
        np_[:self.num_resources] = prices / 100.0
        tr = np.ones(self.max_resources, dtype=np.float32)
        tr[:self.num_resources] = self.sma50[self.current_step] / np.maximum(1e-4, self.sma200[self.current_step])
        v = np.zeros(self.max_resources, dtype=np.float32)
        v[:self.num_resources] = self.vol20[self.current_step]
        return np.concatenate([w_cash, w_a, np_, tr, v]).astype(np.float32)
    
    def _get_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)
    
    def step(self, action):
        # Log raw action
        self.log_raw_action.append(action.copy())
        
        ea = np.exp(action - np.max(action))
        tw = ea / np.sum(ea)
        tc = tw[0]; ra = tw[1:1+self.num_resources]
        tot = tc + np.sum(ra); tc /= tot; ta = ra / tot
        
        # Log target weights
        self.log_target_cash.append(tc)
        self.log_target_stock.append(np.sum(ta))
        
        prices = self.prices_matrix[self.current_step]
        cw = max(1e-4, self._get_portfolio_value())
        ca = (self.shares * prices) / cw
        actual_cash_w = self.cash / cw
        
        self.log_cash_weight.append(actual_cash_w)
        
        did_rebalance = False
        if np.sum(np.abs(ca - ta)) > self.rebalance_threshold:
            self.rebalance_count += 1
            did_rebalance = True
            rv = np.sum(np.abs((self.shares * prices) - cw * ta))
            nw = max(1e-4, cw - rv * self.transaction_fee)
            self.cash = nw * tc
            self.shares = (nw * ta) / np.maximum(1e-4, prices)
        
        self.log_rebalanced.append(did_rebalance)
        
        self.current_step += 1
        done = self.current_step >= self.num_steps - 1
        self.obs_history.pop(0); self.obs_history.append(self._get_single_obs())
        nw = self._get_portfolio_value()
        self.log_wealth.append(nw)
        return self._get_obs(), 0.0, done, False, {"portfolio_value": nw, "rebalances": self.rebalance_count}


def main():
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    model = PPO.load("./data/v0.3_rl_checkpoints/rai_v3_multiregime.zip")
    
    env = DiagnosticEnv(price_df=test_df)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
    
    # ═══════════════════════════════════════════════
    #  ANALYSIS
    # ═══════════════════════════════════════════════
    cash_w = np.array(env.log_cash_weight)
    target_cash = np.array(env.log_target_cash)
    target_stock = np.array(env.log_target_stock)
    rebalanced = np.array(env.log_rebalanced)
    wealth = np.array(env.log_wealth)
    raw_actions = np.array(env.log_raw_action)
    
    total_days = len(cash_w)
    rebalance_days = np.sum(rebalanced)
    
    print("=" * 80)
    print("  DIAGNOSTIC: Is the MLP Actually Doing Anything?")
    print("=" * 80)
    
    # TEST 1: Does the MLP change its outputs?
    print(f"\n  TEST 1: Does the MLP change its action outputs over time?")
    print(f"  {'-'*60}")
    action_std = np.std(raw_actions, axis=0)
    action_mean = np.mean(raw_actions, axis=0)
    print(f"  Action dim 0 (cash logit):  mean={action_mean[0]:.4f}  std={action_std[0]:.4f}")
    for i in range(1, min(6, len(action_mean))):
        print(f"  Action dim {i} (asset {i} logit): mean={action_mean[i]:.4f}  std={action_std[i]:.4f}")
    print(f"  ...")
    total_action_std = np.mean(action_std)
    print(f"  Average std across all dims: {total_action_std:.6f}")
    if total_action_std < 0.001:
        print(f"  ⚠️  VERDICT: Actions are NEARLY CONSTANT. MLP is NOT adapting.")
    elif total_action_std < 0.01:
        print(f"  ⚠️  VERDICT: Actions change VERY SLIGHTLY. MLP is barely adapting.")
    else:
        print(f"  ✅  VERDICT: Actions change meaningfully. MLP IS adapting.")
    
    # TEST 2: Cash allocation over time
    print(f"\n  TEST 2: Cash Weight Over Time")
    print(f"  {'-'*60}")
    print(f"  Actual cash weight: min={np.min(cash_w):.4f}  max={np.max(cash_w):.4f}  mean={np.mean(cash_w):.4f}  std={np.std(cash_w):.4f}")
    print(f"  Target cash weight: min={np.min(target_cash):.4f}  max={np.max(target_cash):.4f}  mean={np.mean(target_cash):.4f}  std={np.std(target_cash):.4f}")
    print(f"  Target stock weight: min={np.min(target_stock):.4f}  max={np.max(target_stock):.4f}  mean={np.mean(target_stock):.4f}")
    if np.std(target_cash) < 0.001:
        print(f"  ⚠️  VERDICT: Target cash weight is CONSTANT = buy-and-hold")
    else:
        print(f"  ✅  VERDICT: Target cash weight varies. MLP is making allocation decisions.")
    
    # TEST 3: How often does it trade?
    print(f"\n  TEST 3: Trading Frequency")
    print(f"  {'-'*60}")
    print(f"  Total trading days:    {total_days}")
    print(f"  Days with rebalance:   {rebalance_days} ({rebalance_days/total_days*100:.1f}%)")
    print(f"  Days without rebalance:{total_days - rebalance_days} ({(total_days-rebalance_days)/total_days*100:.1f}%)")
    if rebalance_days < 5:
        print(f"  ⚠️  VERDICT: Almost NEVER trades. This is effectively buy-and-hold.")
    elif rebalance_days < total_days * 0.05:
        print(f"  ⚠️  VERDICT: Trades very rarely (<5% of days). Mostly buy-and-hold.")
    else:
        print(f"  ✅  VERDICT: Trades on {rebalance_days/total_days*100:.1f}% of days.")
    
    # TEST 4: Compare to pure buy-and-hold of initial allocation
    print(f"\n  TEST 4: RAI v3 vs Pure Buy-and-Hold (same initial allocation)")
    print(f"  {'-'*60}")
    prices = test_df.values
    initial_prices = prices[32]  # history_len offset
    # Buy-and-hold: 50% cash, 50% equal-weight stocks from day 0
    bh_cash = 10000.0 * 0.50
    bh_shares = (10000.0 * 0.50 / test_df.shape[1]) / initial_prices
    bh_equity = bh_cash + np.sum(bh_shares * prices[32:], axis=1)
    
    rai_final = wealth[-1]
    bh_final = bh_equity[-1]
    rai_ret = (rai_final / 10000.0 - 1) * 100
    bh_ret = (bh_final / 10000.0 - 1) * 100
    
    print(f"  RAI v3 Final:       ${rai_final:,.2f}  ({rai_ret:+.2f}%)")
    print(f"  Buy-Hold Final:     ${bh_final:,.2f}  ({bh_ret:+.2f}%)")
    print(f"  Difference:         ${rai_final - bh_final:+,.2f}  ({rai_ret - bh_ret:+.2f}%)")
    
    if abs(rai_ret - bh_ret) < 2.0:
        print(f"  ⚠️  VERDICT: RAI v3 is within 2% of buy-and-hold. MLP adds NOTHING.")
    elif rai_ret > bh_ret:
        print(f"  ✅  VERDICT: RAI v3 outperforms buy-and-hold by {rai_ret - bh_ret:.2f}%")
    else:
        print(f"  ❌  VERDICT: RAI v3 UNDERPERFORMS buy-and-hold by {bh_ret - rai_ret:.2f}%")
    
    # TEST 5: Does it reduce exposure during crashes?
    print(f"\n  TEST 5: Behavior During COVID Crash (Feb-Mar 2020)")
    print(f"  {'-'*60}")
    # Find COVID crash period in the data
    dates = test_df.index
    crash_start_idx = None
    crash_end_idx = None
    for i, d in enumerate(dates):
        if str(d)[:7] == '2020-02' and crash_start_idx is None:
            crash_start_idx = max(0, i - 32)  # offset for history
        if str(d)[:7] == '2020-04' and crash_end_idx is None:
            crash_end_idx = i - 32
    
    if crash_start_idx is not None and crash_end_idx is not None:
        crash_cash = target_cash[crash_start_idx:crash_end_idx]
        pre_crash_cash = target_cash[max(0,crash_start_idx-40):crash_start_idx]
        post_crash_cash = target_cash[crash_end_idx:crash_end_idx+40]
        
        print(f"  Target cash before crash: {np.mean(pre_crash_cash):.4f}")
        print(f"  Target cash during crash: {np.mean(crash_cash):.4f}")
        print(f"  Target cash after crash:  {np.mean(post_crash_cash):.4f}")
        
        if np.mean(crash_cash) > np.mean(pre_crash_cash) + 0.02:
            print(f"  ✅  MLP increased cash during crash (defensive behavior)")
        else:
            print(f"  ❌  MLP did NOT increase cash during crash (no defensive behavior)")
    
    # TEST 6: Action entropy
    print(f"\n  TEST 6: Action Distribution Analysis")
    print(f"  {'-'*60}")
    # Check if all actions are basically the same
    action_range = np.max(raw_actions, axis=0) - np.min(raw_actions, axis=0)
    print(f"  Action range (max-min) per dim:")
    print(f"    Cash logit range:  {action_range[0]:.4f}")
    print(f"    Mean asset range:  {np.mean(action_range[1:]):.4f}")
    print(f"    Max asset range:   {np.max(action_range[1:]):.4f}")
    
    # Check correlation between actions and trend features
    print(f"\n  TEST 7: Do actions correlate with market conditions?")
    print(f"  {'-'*60}")
    spy_prices = test_df['SPY'].values[32:-1]
    spy_rets = (spy_prices[1:] - spy_prices[:-1]) / spy_prices[:-1]
    min_len = min(len(spy_rets), len(target_cash)-1)
    
    if min_len > 100:
        # Rolling 20-day SPY return vs target cash weight
        from scipy.stats import pearsonr
        spy_rolling = pd.Series(spy_rets[:min_len]).rolling(20).mean().dropna().values
        cash_aligned = target_cash[20:20+len(spy_rolling)]
        if len(spy_rolling) == len(cash_aligned):
            corr, pval = pearsonr(spy_rolling, cash_aligned)
            print(f"  Correlation(20d SPY return, target cash weight): {corr:.4f}  p={pval:.4f}")
            if corr < -0.1 and pval < 0.05:
                print(f"  ✅  Negative correlation: MLP increases cash when returns drop (smart)")
            elif corr > 0.1 and pval < 0.05:
                print(f"  ⚠️  Positive correlation: MLP increases cash when returns rise (chasing)")
            else:
                print(f"  ❌  No significant correlation: MLP ignores market conditions")
    
    # FINAL VERDICT
    print(f"\n{'='*80}")
    print(f"  FINAL VERDICT")
    print(f"{'='*80}")
    
    is_static = total_action_std < 0.01
    is_buy_hold = rebalance_days < total_days * 0.03
    adds_value = abs(rai_ret - bh_ret) > 2.0
    
    if is_static and is_buy_hold:
        print(f"\n  ❌ THE MLP IS NOT REALLY WORKING.")
        print(f"     It outputs near-constant actions and almost never trades.")
        print(f"     Performance comes from the initial 50/50 allocation, not the model.")
    elif is_buy_hold and not adds_value:
        print(f"\n  ⚠️ THE MLP IS MOSTLY BUY-AND-HOLD.")
        print(f"     It rarely rebalances, and its performance matches a static portfolio.")
        print(f"     The model adds minimal value beyond the initial allocation.")
    elif adds_value:
        print(f"\n  ✅ THE MLP IS ADDING VALUE.")
        print(f"     It makes active allocation decisions that differ from buy-and-hold.")
        print(f"     Performance gap: {rai_ret - bh_ret:+.2f}% vs static allocation.")
    else:
        print(f"\n  ⚠️ INCONCLUSIVE. More analysis needed.")


if __name__ == "__main__":
    main()
