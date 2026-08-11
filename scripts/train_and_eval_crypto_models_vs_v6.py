"""
Train Supervised AI Models on Real Crypto Data (2018-2021) and Compare Out-of-Sample (2022-2024) vs Frozen Zero-Shot RAI v6 ALPHA
Starting Capital: $10,000.00
"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_v6_fast import DeepEndToEndTradingNet, RawPriceSyntheticEnv
from scripts.eval_vs_standard_ai import (
    train_lstm_model, evaluate_lstm_strategy,
    build_xgb_features, evaluate_xgb_strategy,
    evaluate_risk_parity, evaluate_momentum,
    GradientBoostingClassifier, compute_metrics
)


# ═══════════════════════════════════════════════════════════════════
#  REAL-CRYPTO PPO TRAINER
# ═══════════════════════════════════════════════════════════════════

class RealCryptoGymEnv(RawPriceSyntheticEnv):
    def __init__(self, price_df, history_len=30, fee=0.001):
        self.prices_df = price_df.values
        self.T_max, self.num_assets = self.prices_df.shape
        super().__init__(num_assets=self.num_assets, history_len=history_len, episode_len=self.T_max-history_len-5, fee=fee)

    def _generate_raw_prices(self):
        return self.prices_df

    def _get_obs(self):
        obs_history = []
        for i in range(self.history_len):
            t_idx = self.current_step - self.history_len + 1 + i
            p = self.prices[t_idx]
            p_prev = self.prices[max(0, t_idx - 1)]
            norm_p = np.pad(p / np.maximum(1e-4, self.prices[0]), (0, 10 - self.num_assets), constant_values=1.0)
            log_r = np.pad(np.log(p / np.maximum(1e-4, p_prev)), (0, 10 - self.num_assets), constant_values=0.0)
            dd = (self.portfolio_value - self.peak_value) / max(1e-4, self.peak_value)
            obs_history.append(np.concatenate([norm_p, log_r, [self.cash / max(1e-4, self.portfolio_value), dd]]).astype(np.float32))
        return np.concatenate(obs_history).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = self.history_len
        self.prices = self._generate_raw_prices()
        self.cash = 10000.0
        self.shares = np.zeros(self.num_assets)
        self.portfolio_value = 10000.0
        self.peak_value = 10000.0
        return self._get_obs(), {}

    def step(self, action):
        cl = np.clip(action[0] - 2.5, -8.0, 3.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cl))
        target_stock_frac = 1.0 - target_cash_frac

        al = action[1:1+self.num_assets]
        exp_a = np.exp(al - np.max(al))
        target_asset_w = (exp_a / np.sum(exp_a)) * target_stock_frac

        p_t = self.prices[self.current_step]
        cur_w = max(1e-4, self.cash + np.sum(self.shares * p_t))

        cur_asset_w = (self.shares * p_t) / cur_w
        cur_cash_frac = self.cash / cur_w

        drift = abs(cur_cash_frac - target_cash_frac) + np.sum(np.abs(cur_asset_w - target_asset_w))
        if drift > 0.03:
            trade_val = abs(self.cash - cur_w * target_cash_frac) + np.sum(np.abs(self.shares * p_t - cur_w * target_asset_w))
            fee = trade_val * self.fee
            net_w = max(1e-4, cur_w - fee)
            self.cash = net_w * target_cash_frac
            self.shares = (net_w * target_asset_w) / np.maximum(1e-4, p_t)

        self.current_step += 1
        p_next = self.prices[self.current_step]
        new_w = self.cash + np.sum(self.shares * p_next)

        r_t = (new_w - cur_w) / cur_w
        self.portfolio_value = new_w
        self.peak_value = max(self.peak_value, new_w)

        dd = (new_w - self.peak_value) / max(1e-4, self.peak_value)
        penalty = 0.0
        if dd < -0.20:
            penalty = abs(dd + 0.20) * 2.0

        reward = r_t * 20.0 - penalty
        done = self.current_step >= (self.history_len + self.episode_len - 1)
        return self._get_obs(), float(reward), done, False, {}


def train_crypto_ppo(train_df, history_len=30, total_steps=60000):
    env = RealCryptoGymEnv(train_df, history_len=history_len)
    model = DeepEndToEndTradingNet(history_len=history_len, features_per_step=22, action_dim=11)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    BATCH = 64; ROLLOUT = 1024; EPOCHS = 4
    obs, _ = env.reset(seed=42)
    step = 0

    while step < total_steps:
        obs_b, act_b, rew_b, val_b, logp_b = [], [], [], [], []
        for _ in range(ROLLOUT):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean, val = model(obs_t)
                dist = Normal(mean, torch.exp(model.log_std))
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            act_np = action.squeeze(0).numpy()
            nobs, rew, done, _, _ = env.step(act_np)
            obs_b.append(obs); act_b.append(act_np); rew_b.append(rew); val_b.append(val.item()); logp_b.append(logp.item())
            obs = nobs; step += 1
            if done: obs, _ = env.reset()

        with torch.no_grad():
            _, nval = model(torch.FloatTensor(obs).unsqueeze(0))
            nval = nval.item()

        r = np.array(rew_b); v = np.array(val_b + [nval])
        d = r + 0.99 * v[1:] - v[:-1]
        adv = np.zeros_like(r); gae = 0.0
        for t in reversed(range(len(r))):
            gae = d[t] + 0.99 * 0.95 * gae
            adv[t] = gae
        ret = adv + v[:-1]

        o_t, a_t, adv_t, ret_t, old_t = torch.FloatTensor(np.array(obs_b)), torch.FloatTensor(np.array(act_b)), torch.FloatTensor(adv), torch.FloatTensor(ret), torch.FloatTensor(np.array(logp_b))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(EPOCHS):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), BATCH):
                b_idx = idx[s:s+BATCH]
                mean, val = model(o_t[b_idx])
                dist = Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_t[b_idx])
                surr1 = ratio * adv_t[b_idx]; surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b_idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(val.squeeze(-1), ret_t[b_idx])
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); optimizer.step()

    return model


def eval_v6_alpha_crypto(model, df):
    prices_raw = df.values[:, :2]
    T, N = prices_raw.shape
    cash = 500.0
    init_p = prices_raw[30]
    shares = (9500.0 / N) / init_p
    peak = 10000.0
    wealth_hist = [10000.0]

    obs_history = []
    for t in range(30):
        p = prices_raw[t]; p_prev = prices_raw[max(0, t-1)]
        norm_p = np.pad(p / prices_raw[30], (0, 8), constant_values=1.0)
        log_r = np.pad(np.log(p / np.maximum(1e-4, p_prev)), (0, 8), constant_values=0.0)
        obs_history.append(np.concatenate([norm_p, log_r, [0.05, 0.0]]).astype(np.float32))

    for t in range(30, T):
        flat_obs = np.concatenate(obs_history).astype(np.float32)
        act = model.get_action(flat_obs, deterministic=True)
        cl = np.clip(act[0] - 2.5, -8.0, 3.0)
        target_cash = 1.0 / (1.0 + np.exp(-cl))
        target_stock = 1.0 - target_cash

        al = act[1:3]
        ea = np.exp(al - np.max(al))
        target_aw = (ea / np.sum(ea)) * target_stock

        p = prices_raw[t]; w = max(1e-4, cash + np.sum(shares * p))
        caw = (shares * p) / w; ccf = cash / w

        drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))
        if drift > 0.03:
            tv = abs(cash - w*target_cash) + np.sum(np.abs(shares*p - w*target_aw))
            net = max(1e-4, w - tv * 0.001)
            cash = net * target_cash
            shares = (net * target_aw) / np.maximum(1e-4, p)

        nw = cash + np.sum(shares * p)
        peak = max(peak, nw)
        wealth_hist.append(nw)

        p_prev = prices_raw[t-1]
        norm_p = np.pad(p / prices_raw[30], (0, 8), constant_values=1.0)
        log_r = np.pad(np.log(p / np.maximum(1e-4, p_prev)), (0, 8), constant_values=0.0)
        dd = np.clip((nw - peak) / max(1e-4, peak), -1, 0)
        obs_history.pop(0)
        obs_history.append(np.concatenate([norm_p, log_r, [cash / nw, dd]]).astype(np.float32))

    return compute_metrics(wealth_hist)


# ═══════════════════════════════════════════════════════════════════
#  MAIN CRYPTO TRAINING & EVALUATION CONTROLLER
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 125, flush=True)
    print("  TRAINING SUPERVISED AI MODELS ON REAL CRYPTO DATA (2018-2021) VS ZERO-SHOT RAI v6 ALPHA", flush=True)
    print("=" * 125, flush=True)

    # 1. Download Crypto Data (2018-2021 Train, 2022-2024 Test)
    print("  Downloading Real BTC-USD and ETH-USD Market Data...", flush=True)
    crypto_data = yf.download(["BTC-USD", "ETH-USD"], start="2018-01-01", end="2024-01-01", progress=False)
    if isinstance(crypto_data.columns, pd.MultiIndex):
        crypto_data = crypto_data['Close']
    crypto_data = crypto_data.dropna()

    train_df = crypto_data.loc['2018-01-01':'2021-12-31']
    test_df = crypto_data.loc['2022-01-01':'2024-01-01']

    print(f"  Training Set (2018-2021): {len(train_df)} days | Test Set (2022-2024): {len(test_df)} days", flush=True)

    # 2. Train AI Models on Real 2018-2021 Crypto Data
    print("\n  [1/3] Training Supervised Crypto LSTM on Real 2018-2021 Data...", end=" ", flush=True)
    t0 = time.time()
    lstm_model = train_lstm_model(train_df, lookback=20, epochs=100)
    print(f"done in {time.time()-t0:.1f}s", flush=True)

    print("  [2/3] Training Supervised Crypto XGBoost on Real 2018-2021 Data...", end=" ", flush=True)
    t0 = time.time()
    X_tr, y_tr = build_xgb_features(train_df, lookback=20)
    xgb_clf = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_clf.fit(X_tr, y_tr)
    print(f"done in {time.time()-t0:.1f}s", flush=True)

    print("  [3/3] Training Real-Crypto PPO on Real 2018-2021 Data...", end=" ", flush=True)
    t0 = time.time()
    crypto_ppo_model = train_crypto_ppo(train_df, history_len=30, total_steps=50000)
    print(f"done in {time.time()-t0:.1f}s", flush=True)

    # 3. Load Frozen Zero-Shot RAI v6 ALPHA
    v6_alpha_model = DeepEndToEndTradingNet(history_len=30, features_per_step=22, action_dim=11)
    v6_alpha_path = "./data/v0.6_rl_checkpoints/rai_v6_alpha.pt"
    if os.path.exists(v6_alpha_path):
        v6_alpha_model.load_state_dict(torch.load(v6_alpha_path))
        v6_alpha_model.eval()

    # 4. Out-of-Sample Evaluation on Unseen 2022-2024 Crypto Market (Crypto Winter + Recovery)
    print(f"\n{'='*125}", flush=True)
    print(f"  UNSEEN OUT-OF-SAMPLE CRYPTO EVALUATION (2022-2024 | $10,000 STARTING CAPITAL)", flush=True)
    print(f"  Includes 2022 Crypto Winter Crash (-70% BTC Drawdown) + 2023 Recovery", flush=True)
    print(f"{'='*125}", flush=True)
    print(f"  {'Model / Strategy':<45} | {'Final Value ($)':>14} | {'Net Return (%)':>14} | {'Sharpe':>7} | {'Max DD (%)':>10} | Real Crypto Trained?", flush=True)
    print(f"  {'-'*122}", flush=True)

    # Equal Weight Crypto Basket
    ew_eq = 10000.0 * np.mean(test_df.values / test_df.values[0], axis=1)
    m_ew = compute_metrics(ew_eq)
    print(f"  {'Equal-Weight Crypto Basket (BTC + ETH)':<45} | ${m_ew['final']:>14,.2f} | {m_ew['return_pct']:>+13.2f}% | {m_ew['sharpe']:>7.2f} | {m_ew['max_dd_pct']:>9.2f}% | Passive Market")

    # Supervised Crypto LSTM
    m_lstm = compute_metrics(evaluate_lstm_strategy(lstm_model, test_df))
    print(f"  {'Supervised Crypto LSTM':<45} | ${m_lstm['final']:>14,.2f} | {m_lstm['return_pct']:>+13.2f}% | {m_lstm['sharpe']:>7.2f} | {m_lstm['max_dd_pct']:>9.2f}% | ✅ Real Crypto 2018-2021")

    # Supervised Crypto XGBoost
    m_xgb = compute_metrics(evaluate_xgb_strategy(xgb_clf, test_df))
    print(f"  {'Supervised Crypto XGBoost':<45} | ${m_xgb['final']:>14,.2f} | {m_xgb['return_pct']:>+13.2f}% | {m_xgb['sharpe']:>7.2f} | {m_xgb['max_dd_pct']:>9.2f}% | ✅ Real Crypto 2018-2021")

    # Real-Crypto PPO
    m_cppo = eval_v6_alpha_crypto(crypto_ppo_model, test_df)
    print(f"  {'Real-Crypto PPO (Trained on 2018-2021 Crypto)':<45} | ${m_cppo['final']:>14,.2f} | {m_cppo['return_pct']:>+13.2f}% | {m_cppo['sharpe']:>7.2f} | {m_cppo['max_dd_pct']:>9.2f}% | ✅ Real Crypto 2018-2021")

    # Zero-Shot RAI v6 ALPHA
    m_v6a = eval_v6_alpha_crypto(v6_alpha_model, test_df)
    print(f"  {'🏆 Zero-Shot RAI v6 ALPHA (OUR MODEL)':<45} | ${m_v6a['final']:>14,.2f} | {m_v6a['return_pct']:>+13.2f}% | {m_v6a['sharpe']:>7.2f} | {m_v6a['max_dd_pct']:>9.2f}% | ❌ 0% Real Data (Zero-Shot)")

    print(f"  {'-'*122}", flush=True)

if __name__ == "__main__":
    main()
