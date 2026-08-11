"""
RAI v4: Real Market Evaluation + Full Diagnostic
==================================================
Evaluates v4 on real market data (2010-2019, 2020-2024)
with the v4-compatible observation bridge.
"""
import os, sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


class RealMarketV4Env(gym.Env):
    """Real market eval env matching v4's observation structure."""
    def __init__(self, price_df, initial_cash=10000.0, history_len=16,
                 max_assets=10, transaction_fee=0.001, rebalance_threshold=0.03):
        super().__init__()
        self.price_df = price_df.copy()
        self.prices_matrix = self.price_df.values
        self.num_steps, self.num_assets_real = self.prices_matrix.shape
        self.max_assets = max_assets
        self.num_assets = min(self.num_assets_real, max_assets)
        self.initial_cash = initial_cash
        self.history_len = history_len
        self.transaction_fee = transaction_fee
        self.rebalance_threshold = rebalance_threshold

        # Precompute indicators
        T = self.num_steps
        N = self.num_assets
        prices = self.prices_matrix[:, :N]

        self.sma20 = np.zeros((T, N))
        self.sma50 = np.zeros((T, N))
        self.vol10 = np.zeros((T, N))
        self.returns_5d = np.zeros((T, N))

        for t in range(T):
            self.sma20[t] = np.mean(prices[max(0,t-20):t+1], axis=0)
            self.sma50[t] = np.mean(prices[max(0,t-50):t+1], axis=0)
            if t >= 5:
                self.returns_5d[t] = (prices[t] - prices[t-5]) / np.maximum(1e-4, prices[t-5])
            if t >= 10:
                sub = prices[max(0,t-10):t+1]
                r = (sub[1:] - sub[:-1]) / np.maximum(1e-4, sub[:-1])
                self.vol10[t] = np.std(r, axis=0)

        # Obs: cash_w(1) + asset_w(N) + returns_5d(N) + trend(N) + vol(N) + drawdown(1)
        self.single_obs_dim = 1 + self.max_assets * 4 + 1
        self.obs_dim = history_len * self.single_obs_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(max_assets + 1,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.history_len + 20
        prices = self.prices_matrix[self.current_step, :self.num_assets]
        self.cash = self.initial_cash * 0.50
        per_asset = (self.initial_cash * 0.50) / self.num_assets
        self.shares = per_asset / prices
        self.peak_wealth = self.initial_cash
        self.rebalance_count = 0
        self.obs_history = [self._get_single_obs() for _ in range(self.history_len)]
        # Diagnostic logs
        self.log_target_cash = []
        self.log_target_stock = []
        self.log_wealth = []
        self.log_rebalanced = []
        self.log_raw_action = []
        return self._get_obs(), {}

    def _get_portfolio_value(self):
        prices = self.prices_matrix[self.current_step, :self.num_assets]
        return self.cash + np.sum(self.shares * prices)

    def _get_single_obs(self):
        prices = self.prices_matrix[self.current_step, :self.num_assets]
        wealth = max(1e-4, self._get_portfolio_value())

        w_cash = self.cash / wealth
        w_assets = np.zeros(self.max_assets, dtype=np.float32)
        w_assets[:self.num_assets] = (self.shares * prices) / wealth

        ret5 = np.zeros(self.max_assets, dtype=np.float32)
        ret5[:self.num_assets] = self.returns_5d[self.current_step]

        trend = np.ones(self.max_assets, dtype=np.float32)
        trend[:self.num_assets] = self.sma20[self.current_step] / np.maximum(1e-4, self.sma50[self.current_step])

        vol = np.zeros(self.max_assets, dtype=np.float32)
        vol[:self.num_assets] = self.vol10[self.current_step]

        dd = (wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
        dd = np.clip(dd, -1.0, 0.0)

        return np.concatenate([[w_cash], w_assets, ret5, trend, vol, [dd]]).astype(np.float32)

    def _get_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        self.log_raw_action.append(action.copy())

        exp_act = np.exp(action - np.max(action))
        tw = exp_act / np.sum(exp_act)
        tc = tw[0]
        ra = tw[1:1+self.num_assets]
        tot = tc + np.sum(ra)
        tc /= tot
        ta = ra / tot

        self.log_target_cash.append(tc)
        self.log_target_stock.append(np.sum(ta))

        prices = self.prices_matrix[self.current_step, :self.num_assets]
        cw = max(1e-4, self._get_portfolio_value())
        ca = (self.shares * prices) / cw

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

        new_wealth = self._get_portfolio_value()
        self.peak_wealth = max(self.peak_wealth, new_wealth)
        self.log_wealth.append(new_wealth)

        self.obs_history.pop(0)
        self.obs_history.append(self._get_single_obs())

        return self._get_obs(), 0.0, done, False, {
            "portfolio_value": new_wealth,
            "rebalances": self.rebalance_count
        }


def compute_metrics(eq):
    eq = np.array(eq, dtype=np.float64)
    if len(eq) < 2:
        return {"final": eq[-1], "return_pct": 0, "vol_pct": 0, "sharpe": 0, "max_dd_pct": 0}
    rets = (eq[1:] - eq[:-1]) / np.maximum(1e-8, eq[:-1])
    ret_pct = ((eq[-1] - eq[0]) / eq[0]) * 100.0
    vol = np.std(rets) * np.sqrt(252) * 100.0
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 1e-8 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = np.min(dd) * 100.0
    return {"final": eq[-1], "return_pct": ret_pct, "vol_pct": vol, "sharpe": sharpe, "max_dd_pct": max_dd}


def main():
    model = PPO.load("./data/v0.4_rl_checkpoints/rai_v4_fast")

    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)

    for label, df in [("2020-2024 (Out-of-Sample)", test_df), ("2010-2019 (Historical)", train_df)]:
        print(f"\n{'='*80}", flush=True)
        print(f"  {label}: {len(df)} trading days", flush=True)
        print(f"{'='*80}", flush=True)

        env = RealMarketV4Env(price_df=df, max_assets=10)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)

        # ── Metrics ──
        eq = [10000.0] + env.log_wealth
        m = compute_metrics(eq)

        target_cash = np.array(env.log_target_cash)
        target_stock = np.array(env.log_target_stock)
        raw_actions = np.array(env.log_raw_action)
        rebalanced = np.array(env.log_rebalanced)

        action_std = np.mean(np.std(raw_actions, axis=0))
        cash_std = np.std(target_cash)
        rebal_pct = np.sum(rebalanced) / len(rebalanced) * 100

        print(f"\n  PERFORMANCE:", flush=True)
        print(f"    Final:    ${m['final']:,.2f}", flush=True)
        print(f"    Return:   {m['return_pct']:+.2f}%", flush=True)
        print(f"    Vol:      {m['vol_pct']:.2f}%", flush=True)
        print(f"    Sharpe:   {m['sharpe']:.2f}", flush=True)
        print(f"    Max DD:   {m['max_dd_pct']:.2f}%", flush=True)

        print(f"\n  DIAGNOSTIC (Is it actually adapting?):", flush=True)
        print(f"    Action std:        {action_std:.6f}  {'✅' if action_std > 0.001 else '❌'}", flush=True)
        print(f"    Cash weight std:   {cash_std:.6f}  {'✅' if cash_std > 0.001 else '❌'}", flush=True)
        print(f"    Cash weight range: {np.min(target_cash):.4f} — {np.max(target_cash):.4f}", flush=True)
        print(f"    Stock weight range:{np.min(target_stock):.4f} — {np.max(target_stock):.4f}", flush=True)
        print(f"    Rebalance days:    {np.sum(rebalanced)}/{len(rebalanced)} ({rebal_pct:.1f}%)", flush=True)

        # SPY comparison
        spy = df['SPY'].values
        eq_spy = 10000.0 * (spy / spy[0])
        m_spy = compute_metrics(eq_spy)

        # Buy-hold same initial
        prices = df.values[:, :10]
        init_p = prices[env.history_len + 20]
        bh_shares = (5000.0 / 10) / init_p
        bh_eq = 5000.0 + np.sum(bh_shares * prices[env.history_len+20:], axis=1)
        m_bh = compute_metrics([10000.0] + list(bh_eq))

        print(f"\n  COMPARISON:", flush=True)
        print(f"    {'Model':<30} {'Return':>10} {'Sharpe':>8} {'Max DD':>8}", flush=True)
        print(f"    {'-'*58}", flush=True)
        print(f"    {'RAI v4':<30} {m['return_pct']:>+9.2f}% {m['sharpe']:>7.2f} {m['max_dd_pct']:>7.2f}%", flush=True)
        print(f"    {'SPY Buy & Hold':<30} {m_spy['return_pct']:>+9.2f}% {m_spy['sharpe']:>7.2f} {m_spy['max_dd_pct']:>7.2f}%", flush=True)
        print(f"    {'50/50 Buy-Hold (same init)':<30} {m_bh['return_pct']:>+9.2f}% {m_bh['sharpe']:>7.2f} {m_bh['max_dd_pct']:>7.2f}%", flush=True)

    print(f"\n  ✅ Evaluation complete!", flush=True)


if __name__ == "__main__":
    main()
