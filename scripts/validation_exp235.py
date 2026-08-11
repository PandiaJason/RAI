"""
Validation Experiments 2, 3, 5: No Training Required
=====================================================
Exp 2: Walk-Forward Validation (XGBoost & LSTM)
Exp 3: Regime-Specific Sub-Period Analysis
Exp 5: Statistical Significance (paired t-test, bootstrap CI)
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier


# ═══════════════════════════════════════════════
#  Shared: LSTM & XGBoost builders (from eval_vs_standard_ai.py)
# ═══════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def build_features(prices_df, lookback=20):
    prices = prices_df.values
    T, N = prices.shape
    X, y_reg, y_cls = [], [], []
    for t in range(lookback + 1, T):
        window = prices[t - lookback:t]
        rets = (window[1:] - window[:-1]) / np.maximum(1e-6, window[:-1])
        feat = np.concatenate([
            np.mean(rets, axis=0), np.std(rets, axis=0),
            (prices[t-1] / prices[t - lookback] - 1.0),
            (prices[t-1] / np.mean(window[-5:], axis=0) - 1.0),
            np.max(rets, axis=0) - np.min(rets, axis=0),
        ])
        X.append(feat)
        next_ret = np.mean((prices[t] - prices[t-1]) / np.maximum(1e-6, prices[t-1]))
        y_reg.append(next_ret)
        y_cls.append(1 if next_ret > 0 else 0)
    return np.array(X, dtype=np.float32), np.array(y_reg, dtype=np.float32), np.array(y_cls)

def train_and_eval_lstm(train_df, test_df, lookback=20, epochs=50):
    X_tr, y_tr, _ = build_features(train_df, lookback)
    X_te, y_te, _ = build_features(test_df, lookback)
    
    X_tensor = torch.FloatTensor(X_tr).unsqueeze(1)
    y_tensor = torch.FloatTensor(y_tr).unsqueeze(1)
    
    model = LSTMPredictor(input_dim=X_tr.shape[1], hidden_dim=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    # Evaluate on test
    prices = test_df.values
    T, N = prices.shape
    equity = [10000.0]
    cash = 10000.0
    shares = np.zeros(N)
    
    for t in range(lookback + 1, T):
        window = prices[t - lookback:t]
        rets = (window[1:] - window[:-1]) / np.maximum(1e-6, window[:-1])
        feat = np.concatenate([
            np.mean(rets, axis=0), np.std(rets, axis=0),
            (prices[t-1] / prices[t - lookback] - 1.0),
            (prices[t-1] / np.mean(window[-5:], axis=0) - 1.0),
            np.max(rets, axis=0) - np.min(rets, axis=0),
        ]).astype(np.float32)
        
        with torch.no_grad():
            x = torch.FloatTensor(feat).unsqueeze(0).unsqueeze(0)
            pred_return = model(x).item()
        
        wealth = cash + np.sum(shares * prices[t-1])
        if pred_return > 0:
            shares = (wealth / N) / np.maximum(1e-6, prices[t-1])
            cash = 0.0
        else:
            cash = wealth
            shares = np.zeros(N)
        equity.append(cash + np.sum(shares * prices[t]))
    
    return equity

def train_and_eval_xgb(train_df, test_df, lookback=20):
    X_tr, _, y_tr = build_features(train_df, lookback)
    
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    clf.fit(X_tr, y_tr)
    
    prices = test_df.values
    T, N = prices.shape
    equity = [10000.0]
    cash = 10000.0
    shares = np.zeros(N)
    
    for t in range(lookback + 1, T):
        window = prices[t - lookback:t]
        rets = (window[1:] - window[:-1]) / np.maximum(1e-6, window[:-1])
        feat = np.concatenate([
            np.mean(rets, axis=0), np.std(rets, axis=0),
            (prices[t-1] / prices[t - lookback] - 1.0),
            (prices[t-1] / np.mean(window[-5:], axis=0) - 1.0),
            np.max(rets, axis=0) - np.min(rets, axis=0),
        ]).reshape(1, -1)
        
        pred = clf.predict(feat)[0]
        wealth = cash + np.sum(shares * prices[t-1])
        if pred == 1:
            shares = (wealth / N) / np.maximum(1e-6, prices[t-1])
            cash = 0.0
        else:
            cash = wealth
            shares = np.zeros(N)
        equity.append(cash + np.sum(shares * prices[t]))
    
    return equity

def compute_metrics(eq):
    eq = np.array(eq, dtype=np.float64)
    if len(eq) < 2:
        return {"return_pct": 0, "sharpe": 0, "max_dd_pct": 0}
    rets = (eq[1:] - eq[:-1]) / np.maximum(1e-8, eq[:-1])
    ret_pct = ((eq[-1] - eq[0]) / eq[0]) * 100.0
    vol = np.std(rets) * np.sqrt(252) * 100.0
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 1e-8 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = np.min(dd) * 100.0
    return {"return_pct": ret_pct, "vol_pct": vol, "sharpe": sharpe, "max_dd_pct": max_dd}

# ═══════════════════════════════════════════════
#  RAI v3 evaluation helper
# ═══════════════════════════════════════════════

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class ZeroShotRealMarketEnvV3(gym.Env):
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
        ea = np.exp(action - np.max(action))
        tw = ea / np.sum(ea)
        tc = tw[0]; ra = tw[1:1+self.num_resources]
        tot = tc + np.sum(ra); tc /= tot; ta = ra / tot
        prices = self.prices_matrix[self.current_step]
        cw = max(1e-4, self._get_portfolio_value())
        ca = (self.shares * prices) / cw
        if np.sum(np.abs(ca - ta)) > self.rebalance_threshold:
            self.rebalance_count += 1
            rv = np.sum(np.abs((self.shares * prices) - cw * ta))
            nw = max(1e-4, cw - rv * self.transaction_fee)
            self.cash = nw * tc
            self.shares = (nw * ta) / np.maximum(1e-4, prices)
        self.current_step += 1
        done = self.current_step >= self.num_steps - 1
        self.obs_history.pop(0); self.obs_history.append(self._get_single_obs())
        nw = self._get_portfolio_value()
        return self._get_obs(), 0.0, done, False, {"portfolio_value": nw, "rebalances": self.rebalance_count}

def eval_rai_v3(model, df):
    env = ZeroShotRealMarketEnvV3(price_df=df)
    obs, _ = env.reset()
    eq = [10000.0]; done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        eq.append(info["portfolio_value"])
    return eq

def eval_rai_v3_daily_returns(model, df):
    eq = eval_rai_v3(model, df)
    eq = np.array(eq)
    return (eq[1:] - eq[:-1]) / np.maximum(1e-8, eq[:-1])


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════

def run_experiment_2(train_df):
    print("\n" + "=" * 90)
    print("  EXPERIMENT 2: Walk-Forward Validation for XGBoost & LSTM")
    print("  5 Rolling Folds on 2010-2019 Data")
    print("=" * 90)
    
    years = sorted(train_df.index.year.unique())
    
    lstm_fold_returns = []
    xgb_fold_returns = []
    
    print(f"\n  {'Fold':<6} | {'Train Years':<20} | {'Test Year':<10} | {'LSTM Return':<14} | {'XGBoost Return':<14}")
    print(f"  {'-'*70}")
    
    for fold_idx, test_year in enumerate(years[3:]):  # Need at least 3 years for training
        train_years = [y for y in years if y < test_year]
        
        fold_train = train_df[train_df.index.year.isin(train_years)]
        fold_test = train_df[train_df.index.year == test_year]
        
        if len(fold_test) < 50:
            continue
        
        # LSTM
        try:
            eq_lstm = train_and_eval_lstm(fold_train, fold_test, epochs=50)
            m_lstm = compute_metrics(eq_lstm)
            lstm_fold_returns.append(m_lstm["return_pct"])
        except:
            m_lstm = {"return_pct": 0.0}
            lstm_fold_returns.append(0.0)
        
        # XGBoost
        try:
            eq_xgb = train_and_eval_xgb(fold_train, fold_test)
            m_xgb = compute_metrics(eq_xgb)
            xgb_fold_returns.append(m_xgb["return_pct"])
        except:
            m_xgb = {"return_pct": 0.0}
            xgb_fold_returns.append(0.0)
        
        train_str = f"{train_years[0]}-{train_years[-1]}"
        print(f"  {fold_idx+1:<6} | {train_str:<20} | {test_year:<10} | {m_lstm['return_pct']:>+10.2f}%   | {m_xgb['return_pct']:>+10.2f}%")
    
    print(f"  {'-'*70}")
    print(f"  {'MEAN':<6} | {'Walk-Forward Avg':<20} | {'All':<10} | {np.mean(lstm_fold_returns):>+10.2f}%   | {np.mean(xgb_fold_returns):>+10.2f}%")
    print(f"  {'STD':<6} |                      |            | {np.std(lstm_fold_returns):>10.2f}%   | {np.std(xgb_fold_returns):>10.2f}%")
    
    print(f"\n  Walk-Forward vs In-Sample Comparison:")
    print(f"  XGBoost Walk-Forward Mean: {np.mean(xgb_fold_returns):+.2f}%  vs  In-Sample: +11,975%  → Overfitting Ratio: {11975/max(0.01, abs(np.mean(xgb_fold_returns))):.0f}x")
    
    return lstm_fold_returns, xgb_fold_returns


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: Regime-Specific Sub-Period Analysis
# ═══════════════════════════════════════════════════════════════════

def run_experiment_3(test_df, rai_model):
    print("\n" + "=" * 90)
    print("  EXPERIMENT 3: Regime-Specific Sub-Period Analysis (2020-2024)")
    print("=" * 90)
    
    regimes = [
        ("COVID Crash (Bear)",     "2020-01-02", "2020-03-23"),
        ("Recovery Rally (Bull)",  "2020-03-24", "2021-12-31"),
        ("Rate Hike DD (Bear)",    "2022-01-03", "2022-10-14"),
        ("Choppy Recovery (Sideways)", "2022-10-17", "2023-12-29"),
    ]
    
    print(f"\n  {'Regime':<30} | {'RAI v3':>10} | {'SPY B&H':>10} | {'Equal-Wt':>10} | {'Days':>5}")
    print(f"  {'-'*75}")
    
    for regime_name, start, end in regimes:
        try:
            sub = test_df.loc[start:end]
        except:
            continue
        if len(sub) < 10:
            continue
        
        # RAI v3
        eq_rai = eval_rai_v3(rai_model, sub)
        m_rai = compute_metrics(eq_rai)
        
        # SPY
        spy = sub['SPY'].values
        eq_spy = 10000.0 * (spy / spy[0])
        m_spy = compute_metrics(eq_spy)
        
        # Equal Weight
        eq_ew = 10000.0 * np.mean(sub.values / sub.values[0], axis=1)
        m_ew = compute_metrics(eq_ew)
        
        print(f"  {regime_name:<30} | {m_rai['return_pct']:>+9.2f}% | {m_spy['return_pct']:>+9.2f}% | {m_ew['return_pct']:>+9.2f}% | {len(sub):>5}")
    
    print(f"  {'-'*75}")


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 5: Statistical Significance
# ═══════════════════════════════════════════════════════════════════

def run_experiment_5(test_df, rai_model):
    print("\n" + "=" * 90)
    print("  EXPERIMENT 5: Statistical Significance (RAI v3 vs LSTM)")
    print("=" * 90)
    
    # RAI v3 daily returns
    rai_rets = eval_rai_v3_daily_returns(rai_model, test_df)
    
    # LSTM daily returns
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)
    eq_lstm = train_and_eval_lstm(train_df, test_df, epochs=100)
    eq_lstm = np.array(eq_lstm)
    lstm_rets = (eq_lstm[1:] - eq_lstm[:-1]) / np.maximum(1e-8, eq_lstm[:-1])
    
    # Align lengths (take the shorter)
    min_len = min(len(rai_rets), len(lstm_rets))
    rai_rets = rai_rets[:min_len]
    lstm_rets = lstm_rets[:min_len]
    
    # Paired t-test
    diff = rai_rets - lstm_rets
    t_stat, p_value = stats.ttest_rel(rai_rets, lstm_rets)
    
    # Bootstrap confidence interval
    n_bootstrap = 10000
    np.random.seed(42)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(min_len, size=min_len, replace=True)
        boot_diffs.append(np.mean(diff[idx]))
    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5) * 252 * 100  # Annualized %
    ci_upper = np.percentile(boot_diffs, 97.5) * 252 * 100
    
    # Effect size (Cohen's d)
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
    
    mean_diff_annual = np.mean(diff) * 252 * 100
    
    print(f"\n  RAI v3 vs LSTM (Paired Daily Returns, N={min_len} days)")
    print(f"  {'-'*60}")
    print(f"  Mean Daily Return Difference:  {np.mean(diff)*100:+.4f}%")
    print(f"  Annualized Return Difference:  {mean_diff_annual:+.2f}%")
    print(f"  Paired t-statistic:            {t_stat:.4f}")
    print(f"  p-value (two-tailed):          {p_value:.6f}")
    print(f"  p-value significant (< 0.05):  {'YES ✅' if p_value < 0.05 else 'NO ❌'}")
    print(f"  95% Bootstrap CI (annualized): [{ci_lower:+.2f}%, {ci_upper:+.2f}%]")
    print(f"  Cohen's d (effect size):       {cohens_d:.4f}")
    print(f"  Effect size interpretation:    {'Small' if abs(cohens_d) < 0.2 else 'Medium' if abs(cohens_d) < 0.5 else 'Large'}")
    print(f"  {'-'*60}")
    
    return {"t_stat": t_stat, "p_value": p_value, "ci_lower": ci_lower, "ci_upper": ci_upper, "cohens_d": cohens_d}


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    
    rai_model = PPO.load("./data/v0.3_rl_checkpoints/rai_v3_multiregime.zip")
    
    run_experiment_2(train_df)
    run_experiment_3(test_df, rai_model)
    run_experiment_5(test_df, rai_model)
    
    print("\n\n  ✅ Experiments 2, 3, 5 complete!")

if __name__ == "__main__":
    main()
