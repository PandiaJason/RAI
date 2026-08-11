"""
RAI v3 vs Standard AI Trading Models
======================================
Compare Zero-Shot RAI v3 (0% real data) against standard AI/ML trading models
that ARE trained on real historical market data:

1. LSTM Return Predictor - Deep learning supervised model (trained on real data)
2. XGBoost Direction Classifier - Gradient boosted trees (trained on real data)
3. Risk Parity Portfolio - Inverse-volatility institutional model
4. Momentum Factor Strategy - Cross-sectional momentum (buy winners, sell losers)
5. SMA Trend Following - Moving average crossover strategy

NOTE: Standard AI models are trained on 2010-2019 data (they see real data).
      RAI v3 sees 0% real data — trained only on synthetic procedural worlds.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from stable_baselines3 import PPO


# ═══════════════════════════════════════════════════════════════════
#  1. LSTM RETURN PREDICTOR (Supervised Deep Learning)
# ═══════════════════════════════════════════════════════════════════

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_lstm_features(prices_df, lookback=20):
    """Build features for LSTM: returns, volatility, momentum, RSI-like."""
    prices = prices_df.values
    T, N = prices.shape
    
    features_list = []
    targets_list = []
    
    for t in range(lookback + 1, T):
        window = prices[t - lookback:t]
        rets = (window[1:] - window[:-1]) / np.maximum(1e-6, window[:-1])
        
        feat = np.concatenate([
            rets.flatten(),                                    # raw returns (lookback-1) * N
            np.mean(rets, axis=0),                              # mean return per asset (N)
            np.std(rets, axis=0),                               # volatility per asset (N)
            (prices[t-1] / prices[t - lookback] - 1.0),        # momentum (N)
        ])
        features_list.append(feat)
        
        # Target: equal-weight portfolio return next day
        next_ret = np.mean((prices[t] - prices[t-1]) / np.maximum(1e-6, prices[t-1]))
        targets_list.append(next_ret)
    
    return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)


def train_lstm_model(train_prices_df, lookback=20, epochs=50, lr=1e-3):
    """Train LSTM on historical price data to predict next-day returns."""
    X, y = build_lstm_features(train_prices_df, lookback)
    
    # Reshape X for LSTM: (samples, seq_len=1, features)
    X_tensor = torch.FloatTensor(X).unsqueeze(1)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    model = LSTMPredictor(input_dim=X.shape[1], hidden_dim=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model


def evaluate_lstm_strategy(model, test_prices_df, initial_cash=10000.0, lookback=20):
    """Use LSTM predictions to allocate: if predicted return > 0, go long; else hold cash."""
    prices = test_prices_df.values
    T, N = prices.shape
    
    equity = [initial_cash]
    cash = initial_cash
    shares = np.zeros(N)
    
    for t in range(lookback + 1, T):
        window = prices[t - lookback:t]
        rets = (window[1:] - window[:-1]) / np.maximum(1e-6, window[:-1])
        
        feat = np.concatenate([
            rets.flatten(),
            np.mean(rets, axis=0),
            np.std(rets, axis=0),
            (prices[t-1] / prices[t - lookback] - 1.0),
        ]).astype(np.float32)
        
        with torch.no_grad():
            x = torch.FloatTensor(feat).unsqueeze(0).unsqueeze(0)
            pred_return = model(x).item()
        
        wealth = cash + np.sum(shares * prices[t-1])
        
        if pred_return > 0:
            # Allocate equally across all assets
            target_per_asset = wealth / N
            shares = target_per_asset / np.maximum(1e-6, prices[t-1])
            cash = 0.0
        else:
            # Move to cash
            cash = wealth
            shares = np.zeros(N)
        
        new_wealth = cash + np.sum(shares * prices[t])
        equity.append(new_wealth)
    
    return equity


# ═══════════════════════════════════════════════════════════════════
#  2. XGBOOST DIRECTION CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

def build_xgb_features(prices_df, lookback=20):
    """Build features for XGBoost: technical indicators."""
    prices = prices_df.values
    T, N = prices.shape
    
    X_list = []
    y_list = []
    
    for t in range(lookback + 1, T):
        window = prices[t - lookback:t]
        rets = (window[1:] - window[:-1]) / np.maximum(1e-6, window[:-1])
        
        feat = np.concatenate([
            np.mean(rets, axis=0),                              # mean return
            np.std(rets, axis=0),                               # volatility
            (prices[t-1] / prices[t - lookback] - 1.0),        # momentum
            (prices[t-1] / np.mean(window[-5:], axis=0) - 1.0), # 5-day mean reversion
            np.max(rets, axis=0) - np.min(rets, axis=0),       # range
        ])
        X_list.append(feat)
        
        # Target: 1 if market goes up, 0 if down
        next_ret = np.mean((prices[t] - prices[t-1]) / np.maximum(1e-6, prices[t-1]))
        y_list.append(1 if next_ret > 0 else 0)
    
    return np.array(X_list), np.array(y_list)


def evaluate_xgb_strategy(clf, test_prices_df, initial_cash=10000.0, lookback=20):
    """Use XGBoost predictions to go long (predicted up) or cash (predicted down)."""
    prices = test_prices_df.values
    T, N = prices.shape
    
    equity = [initial_cash]
    cash = initial_cash
    shares = np.zeros(N)
    
    for t in range(lookback + 1, T):
        window = prices[t - lookback:t]
        rets = (window[1:] - window[:-1]) / np.maximum(1e-6, window[:-1])
        
        feat = np.concatenate([
            np.mean(rets, axis=0),
            np.std(rets, axis=0),
            (prices[t-1] / prices[t - lookback] - 1.0),
            (prices[t-1] / np.mean(window[-5:], axis=0) - 1.0),
            np.max(rets, axis=0) - np.min(rets, axis=0),
        ]).reshape(1, -1)
        
        pred = clf.predict(feat)[0]
        wealth = cash + np.sum(shares * prices[t-1])
        
        if pred == 1:
            target_per_asset = wealth / N
            shares = target_per_asset / np.maximum(1e-6, prices[t-1])
            cash = 0.0
        else:
            cash = wealth
            shares = np.zeros(N)
        
        new_wealth = cash + np.sum(shares * prices[t])
        equity.append(new_wealth)
    
    return equity


# ═══════════════════════════════════════════════════════════════════
#  3. RISK PARITY PORTFOLIO (Inverse Volatility Weighting)
# ═══════════════════════════════════════════════════════════════════

def evaluate_risk_parity(prices_df, initial_cash=10000.0, lookback=60):
    """Allocate inversely proportional to each asset's rolling volatility."""
    prices = prices_df.values
    T, N = prices.shape
    
    equity = [initial_cash]
    
    for t in range(lookback + 1, T):
        window = prices[t - lookback:t]
        rets = (window[1:] - window[:-1]) / np.maximum(1e-6, window[:-1])
        vols = np.std(rets, axis=0)
        inv_vols = 1.0 / np.maximum(0.001, vols)
        weights = inv_vols / np.sum(inv_vols)
        
        if t == lookback + 1:
            prev_wealth = initial_cash
        else:
            prev_wealth = equity[-1]
        
        shares = (prev_wealth * weights) / np.maximum(1e-6, prices[t-1])
        new_wealth = np.sum(shares * prices[t])
        equity.append(new_wealth)
    
    return equity


# ═══════════════════════════════════════════════════════════════════
#  4. MOMENTUM FACTOR STRATEGY (Cross-Sectional)
# ═══════════════════════════════════════════════════════════════════

def evaluate_momentum(prices_df, initial_cash=10000.0, lookback=60, top_k=3):
    """Buy the top-K performing assets over the lookback period."""
    prices = prices_df.values
    T, N = prices.shape
    top_k = min(top_k, N)
    
    equity = [initial_cash]
    
    for t in range(lookback + 1, T):
        momentum = prices[t-1] / prices[t - lookback] - 1.0
        top_indices = np.argsort(momentum)[-top_k:]
        
        if t == lookback + 1:
            prev_wealth = initial_cash
        else:
            prev_wealth = equity[-1]
        
        per_asset = prev_wealth / top_k
        shares = np.zeros(N)
        shares[top_indices] = per_asset / np.maximum(1e-6, prices[t-1, top_indices])
        
        new_wealth = np.sum(shares * prices[t])
        equity.append(new_wealth)
    
    return equity


# ═══════════════════════════════════════════════════════════════════
#  5. SMA TREND FOLLOWING (Moving Average Crossover)
# ═══════════════════════════════════════════════════════════════════

def evaluate_sma_crossover(prices_df, initial_cash=10000.0, fast=50, slow=200):
    """Go long equal-weight when SMA(fast) > SMA(slow) for majority of assets, else cash."""
    prices = prices_df.values
    T, N = prices.shape
    
    equity = [initial_cash]
    cash = initial_cash
    shares = np.zeros(N)
    
    for t in range(slow + 1, T):
        sma_fast = np.mean(prices[t - fast:t], axis=0)
        sma_slow = np.mean(prices[t - slow:t], axis=0)
        
        # Count how many assets are in uptrend
        bullish_count = np.sum(sma_fast > sma_slow)
        
        wealth = cash + np.sum(shares * prices[t-1])
        
        if bullish_count > N / 2:
            # Majority uptrend: go long
            bullish_mask = sma_fast > sma_slow
            n_bull = max(1, np.sum(bullish_mask))
            per_asset = wealth / n_bull
            shares = np.zeros(N)
            shares[bullish_mask] = per_asset / np.maximum(1e-6, prices[t-1, bullish_mask])
            cash = 0.0
        else:
            # Majority downtrend: go to cash
            cash = wealth
            shares = np.zeros(N)
        
        new_wealth = cash + np.sum(shares * prices[t])
        equity.append(new_wealth)
    
    return equity


# ═══════════════════════════════════════════════════════════════════
#  RAI v3 EVALUATION (Same as eval_v3_benchmark.py)
# ═══════════════════════════════════════════════════════════════════

import gymnasium as gym
from gymnasium import spaces


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
            t50_start = max(0, t - 50)
            t200_start = max(0, t - 200)
            self.sma50[t] = np.mean(self.prices_matrix[t50_start:t+1], axis=0)
            self.sma200[t] = np.mean(self.prices_matrix[t200_start:t+1], axis=0)
            if t > 1:
                t20_start = max(0, t - 20)
                sub_p = self.prices_matrix[t20_start:t+1]
                rets = (sub_p[1:] - sub_p[:-1]) / np.maximum(1e-4, sub_p[:-1])
                self.vol20[t] = np.std(rets, axis=0)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(max_resources + 1,), dtype=np.float32)
        self.single_obs_dim = 1 + max_resources * 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.history_len * self.single_obs_dim,), dtype=np.float32)
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.history_len
        self.cash = self.initial_cash * 0.50
        prices = self.prices_matrix[self.current_step]
        per_asset_cash = (self.initial_cash * 0.50) / self.num_resources
        self.shares = per_asset_cash / prices
        self.obs_history = [self._get_single_obs() for _ in range(self.history_len)]
        self.last_wealth = self.initial_cash
        self.rebalance_count = 0
        return self._get_obs(), {}
    
    def _get_portfolio_value(self):
        return self.cash + np.sum(self.shares * self.prices_matrix[self.current_step])
    
    def _get_single_obs(self):
        prices = self.prices_matrix[self.current_step]
        wealth = max(1e-4, self._get_portfolio_value())
        w_cash = np.array([self.cash / wealth], dtype=np.float32)
        w_assets = np.zeros(self.max_resources, dtype=np.float32)
        w_assets[:self.num_resources] = (self.shares * prices) / wealth
        norm_prices = np.ones(self.max_resources, dtype=np.float32)
        norm_prices[:self.num_resources] = prices / 100.0
        trend_ratio = np.ones(self.max_resources, dtype=np.float32)
        trend_ratio[:self.num_resources] = self.sma50[self.current_step] / np.maximum(1e-4, self.sma200[self.current_step])
        v20 = np.zeros(self.max_resources, dtype=np.float32)
        v20[:self.num_resources] = self.vol20[self.current_step]
        return np.concatenate([w_cash, w_assets, norm_prices, trend_ratio, v20]).astype(np.float32)
    
    def _get_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)
    
    def step(self, action):
        exp_act = np.exp(action - np.max(action))
        target_weights = exp_act / np.sum(exp_act)
        target_cash_w = target_weights[0]
        raw_asset_w = target_weights[1:1 + self.num_resources]
        total_active_w = target_cash_w + np.sum(raw_asset_w)
        target_cash_w = target_cash_w / total_active_w
        target_asset_w = raw_asset_w / total_active_w
        
        prices = self.prices_matrix[self.current_step]
        current_wealth = max(1e-4, self._get_portfolio_value())
        current_asset_w = (self.shares * prices) / current_wealth
        
        if np.sum(np.abs(current_asset_w - target_asset_w)) > self.rebalance_threshold:
            self.rebalance_count += 1
            target_asset_vals = current_wealth * target_asset_w
            rebalance_vol = np.sum(np.abs((self.shares * prices) - target_asset_vals))
            fee = rebalance_vol * self.transaction_fee
            net_wealth = max(1e-4, current_wealth - fee)
            self.cash = net_wealth * target_cash_w
            self.shares = (net_wealth * target_asset_w) / np.maximum(1e-4, prices)
        
        self.current_step += 1
        done = self.current_step >= self.num_steps - 1
        self.obs_history.pop(0)
        self.obs_history.append(self._get_single_obs())
        new_wealth = self._get_portfolio_value()
        return self._get_obs(), 0.0, done, False, {"portfolio_value": new_wealth, "rebalances": self.rebalance_count}


def evaluate_rai_v3(model, test_prices_df, initial_cash=10000.0):
    env = ZeroShotRealMarketEnvV3(price_df=test_prices_df, initial_cash=initial_cash)
    obs, _ = env.reset()
    equity = [initial_cash]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        equity.append(info["portfolio_value"])
    return equity


# ═══════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(equity_curve, rf_daily=0.0):
    eq = np.array(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return {"final": eq[-1], "return_pct": 0, "vol_pct": 0, "sharpe": 0, "max_dd_pct": 0}
    returns = (eq[1:] - eq[:-1]) / np.maximum(1e-8, eq[:-1])
    total_return_pct = ((eq[-1] - eq[0]) / eq[0]) * 100.0
    vol_annual_pct = np.std(returns) * np.sqrt(252) * 100.0
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = (mean_ret - rf_daily) / std_ret * np.sqrt(252) if std_ret > 1e-8 else 0.0
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    max_dd_pct = np.min(drawdown) * 100.0
    return {"final": eq[-1], "return_pct": total_return_pct, "vol_pct": vol_annual_pct,
            "sharpe": sharpe, "max_dd_pct": max_dd_pct}


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def print_row(name, m, tag=""):
    print(f"  {name:<45} | ${m['final']:<10.2f} | {m['return_pct']:<9.2f}% | {m['vol_pct']:<7.2f}% | {m['sharpe']:<6.2f} | {m['max_dd_pct']:<9.2f}% | {tag}")


def main():
    print("=" * 110)
    print("  RAI v3 vs STANDARD AI TRADING MODELS: Comprehensive Benchmark")
    print("  NOTE: Standard AI models are TRAINED on real 2010-2019 data.")
    print("        RAI v3 uses 0% real data (pure synthetic training).")
    print("=" * 110)
    
    train_csv = "./data/real_market_checkpoints/train_prices.csv"
    test_csv = "./data/real_market_checkpoints/test_prices.csv"
    
    train_df = pd.read_csv(train_csv, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    
    # ─── Train Standard AI Models on 2010-2019 Real Data ───
    print("\n  Training LSTM Return Predictor on 2010-2019 real data...")
    lstm_model = train_lstm_model(train_df, lookback=20, epochs=100)
    
    print("  Training XGBoost Direction Classifier on 2010-2019 real data...")
    X_train, y_train = build_xgb_features(train_df, lookback=20)
    xgb_clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_clf.fit(X_train, y_train)
    train_acc = xgb_clf.score(X_train, y_train)
    print(f"  XGBoost Training Accuracy: {train_acc:.2%}")
    
    # ─── Load RAI v3 ───
    v3_path = "./data/v0.3_rl_checkpoints/rai_v3_multiregime.zip"
    rai_v3 = PPO.load(v3_path) if os.path.exists(v3_path) else None
    
    # ═══════════════════════════════════════════════════════
    #  EVALUATE ON 2020-2024 (OUT-OF-SAMPLE)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  OUT-OF-SAMPLE EVALUATION: 2020-2024 ({len(test_df)} trading days)")
    print(f"  Standard AI models trained on 2010-2019 real data. RAI v3 trained on 0% real data.")
    print(f"{'='*110}")
    print(f"  {'Model / Strategy':<45} | {'Final ($)':<10} | {'Return (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<6} | {'Max DD (%)':<10} | Data Used")
    print(f"  {'-'*108}")
    
    # SPY
    spy = test_df['SPY'].values
    eq_spy = 10000.0 * (spy / spy[0])
    print_row("SPY Buy & Hold (S&P 500)", compute_metrics(eq_spy), "Passive")
    
    # RAI v3
    if rai_v3:
        eq_v3 = evaluate_rai_v3(rai_v3, test_df)
        print_row("🏆 Zero-Shot RAI v3 (0% Real Data)", compute_metrics(eq_v3), "0% Real Data")
    
    # LSTM
    eq_lstm = evaluate_lstm_strategy(lstm_model, test_df)
    print_row("LSTM Return Predictor (Deep Learning)", compute_metrics(eq_lstm), "Trained on Real")
    
    # XGBoost
    eq_xgb = evaluate_xgb_strategy(xgb_clf, test_df)
    print_row("XGBoost Direction Classifier (ML)", compute_metrics(eq_xgb), "Trained on Real")
    
    # Risk Parity
    eq_rp = evaluate_risk_parity(test_df)
    print_row("Risk Parity (Inverse Volatility)", compute_metrics(eq_rp), "Rule-Based")
    
    # Momentum
    eq_mom = evaluate_momentum(test_df, top_k=3)
    print_row("Momentum Factor (Top-3 Winners)", compute_metrics(eq_mom), "Rule-Based")
    
    # SMA Crossover
    eq_sma = evaluate_sma_crossover(test_df)
    print_row("SMA 50/200 Trend Following", compute_metrics(eq_sma), "Rule-Based")
    
    # Equal Weight
    eq_ew = 10000.0 * np.mean(test_df.values / test_df.values[0], axis=1)
    print_row("Equal-Weight (1/N)", compute_metrics(eq_ew), "Passive")
    
    # 60/40
    if 'TLT' in test_df.columns:
        tlt = test_df['TLT'].values
        eq_6040 = 10000.0 * (0.60 * (spy / spy[0]) + 0.40 * (tlt / tlt[0]))
        print_row("60/40 Portfolio (SPY / TLT)", compute_metrics(eq_6040), "Passive")
    
    print(f"  {'-'*108}")
    
    # ═══════════════════════════════════════════════════════
    #  EVALUATE ON 2010-2019 (HISTORICAL)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  HISTORICAL EVALUATION: 2010-2019 ({len(train_df)} trading days)")
    print(f"  NOTE: LSTM & XGBoost have IN-SAMPLE advantage on this period (trained on same data).")
    print(f"        RAI v3 still uses 0% real data.")
    print(f"{'='*110}")
    print(f"  {'Model / Strategy':<45} | {'Final ($)':<10} | {'Return (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<6} | {'Max DD (%)':<10} | Data Used")
    print(f"  {'-'*108}")
    
    # SPY
    spy_h = train_df['SPY'].values
    eq_spy_h = 10000.0 * (spy_h / spy_h[0])
    print_row("SPY Buy & Hold (S&P 500)", compute_metrics(eq_spy_h), "Passive")
    
    # RAI v3
    if rai_v3:
        eq_v3_h = evaluate_rai_v3(rai_v3, train_df)
        print_row("🏆 Zero-Shot RAI v3 (0% Real Data)", compute_metrics(eq_v3_h), "0% Real Data")
    
    # LSTM (in-sample)
    eq_lstm_h = evaluate_lstm_strategy(lstm_model, train_df)
    print_row("LSTM Return Predictor (IN-SAMPLE)", compute_metrics(eq_lstm_h), "⚠️ In-Sample")
    
    # XGBoost (in-sample)
    eq_xgb_h = evaluate_xgb_strategy(xgb_clf, train_df)
    print_row("XGBoost Direction Classifier (IN-SAMPLE)", compute_metrics(eq_xgb_h), "⚠️ In-Sample")
    
    # Risk Parity
    eq_rp_h = evaluate_risk_parity(train_df)
    print_row("Risk Parity (Inverse Volatility)", compute_metrics(eq_rp_h), "Rule-Based")
    
    # Momentum
    eq_mom_h = evaluate_momentum(train_df, top_k=3)
    print_row("Momentum Factor (Top-3 Winners)", compute_metrics(eq_mom_h), "Rule-Based")
    
    # SMA Crossover
    eq_sma_h = evaluate_sma_crossover(train_df)
    print_row("SMA 50/200 Trend Following", compute_metrics(eq_sma_h), "Rule-Based")
    
    # Equal Weight
    eq_ew_h = 10000.0 * np.mean(train_df.values / train_df.values[0], axis=1)
    print_row("Equal-Weight (1/N)", compute_metrics(eq_ew_h), "Passive")
    
    # 60/40
    if 'TLT' in train_df.columns:
        tlt_h = train_df['TLT'].values
        eq_6040_h = 10000.0 * (0.60 * (spy_h / spy_h[0]) + 0.40 * (tlt_h / tlt_h[0]))
        print_row("60/40 Portfolio (SPY / TLT)", compute_metrics(eq_6040_h), "Passive")
    
    print(f"  {'-'*108}")


if __name__ == "__main__":
    main()
