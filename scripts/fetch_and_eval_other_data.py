"""
Fetch & Evaluate RAI v5 on Completely New Datasets & Asset Classes
(With zero-padding to 10 assets for variable asset basket sizes)
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_v5_dual_head import DualHeadGatedPolicy, metrics


class MultiAssetPaddedV5Env(gym.Env):
    """Bridge environment that zero-pads observation features up to max_assets=10."""
    def __init__(self, price_df, initial_cash=10000.0, history_len=16, max_assets=10, fee=0.001):
        super().__init__()
        self.prices_raw = price_df.values.copy()
        self.T, self.num_assets_real = self.prices_raw.shape
        self.max_assets = max_assets
        self.initial_cash = initial_cash
        self.history_len = history_len
        self.fee = fee

        # Pad prices matrix to max_assets if smaller
        if self.num_assets_real < max_assets:
            pad = np.ones((self.T, max_assets - self.num_assets_real))
            self.prices = np.hstack([self.prices_raw, pad])
        else:
            self.prices = self.prices_raw[:, :max_assets]

        self.single_obs_dim = 4 + 2 * max_assets
        self.obs_dim = history_len * self.single_obs_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (max_assets + 1,), np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = self.history_len + 20
        self.cash = self.initial_cash * 0.5
        p = self.prices[self.step_idx, :self.num_assets_real]
        self.shares = np.zeros(self.max_assets)
        self.shares[:self.num_assets_real] = (self.initial_cash * 0.5 / self.num_assets_real) / p
        self.peak = self.initial_cash
        self.obs_hist = [self._obs() for _ in range(self.history_len)]
        self.log_cash_frac = []; self.log_wealth = []; self.log_rebal = []
        self.log_regime_probs = []
        return self._flat_obs(), {}

    def _w(self):
        p = self.prices[self.step_idx, :self.num_assets_real]
        return self.cash + np.sum(self.shares[:self.num_assets_real] * p)

    def _obs(self):
        p_real = self.prices[self.step_idx, :self.num_assets_real]
        t = self.step_idx
        w = max(1e-4, self._w())
        cw = self.cash / w
        dd = np.clip((w - self.peak) / max(1e-4, self.peak), -1, 0)
        
        r5 = np.mean((p_real - self.prices[max(0,t-5), :self.num_assets_real]) / np.maximum(1e-4, self.prices[max(0,t-5), :self.num_assets_real])) if t >= 5 else 0
        if t >= 10:
            sub = self.prices[t-10:t+1, :self.num_assets_real]
            r = (sub[1:]-sub[:-1])/np.maximum(1e-4,sub[:-1])
            vol = np.mean(np.std(r, axis=0))
        else:
            vol = 0

        # Padded asset weights
        aw = np.zeros(self.max_assets, dtype=np.float32)
        aw[:self.num_assets_real] = (self.shares[:self.num_assets_real] * p_real) / w

        # Padded trend
        trend = np.zeros(self.max_assets, dtype=np.float32)
        if t >= 50:
            s20 = np.mean(self.prices[t-20:t, :self.num_assets_real], axis=0)
            s50 = np.mean(self.prices[t-50:t, :self.num_assets_real], axis=0)
            trend[:self.num_assets_real] = s20 / np.maximum(1e-4, s50) - 1.0

        return np.concatenate([[cw, dd, r5, vol], aw, trend]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_hist).astype(np.float32)

    def step(self, action_tuple):
        action, regime_prob = action_tuple
        cl = np.clip(action[0], -10, 10)
        target_cash = 1.0 / (1.0 + np.exp(-cl))
        target_stock = 1.0 - target_cash

        # Only allocate among real assets
        al = action[1:1+self.num_assets_real]
        ea = np.exp(al - np.max(al)); rw = ea / np.sum(ea)
        target_aw = rw * target_stock

        p_real = self.prices[self.step_idx, :self.num_assets_real]
        w = max(1e-4, self._w())
        caw = (self.shares[:self.num_assets_real] * p_real) / w; ccf = self.cash / w
        drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))

        did_rebal = False
        if drift > 0.03:
            did_rebal = True
            tv = abs(self.cash - w*target_cash) + np.sum(np.abs(self.shares[:self.num_assets_real]*p_real - w*target_aw))
            net = max(1e-4, w - tv * self.fee)
            self.cash = net * target_cash
            self.shares[:self.num_assets_real] = (net * target_aw) / np.maximum(1e-4, p_real)

        self.log_cash_frac.append(target_cash)
        self.log_regime_probs.append(regime_prob)
        self.log_rebal.append(did_rebal)

        self.step_idx += 1
        done = self.step_idx >= self.T - 1
        nw = self._w()
        self.peak = max(self.peak, nw)
        self.log_wealth.append(nw)
        self.obs_hist.pop(0); self.obs_hist.append(self._obs())
        return self._flat_obs(), 0.0, done, False, {"portfolio_value": nw}


def download_asset_basket(tickers, start_date="2015-01-01", end_date="2024-01-01"):
    print(f"  Downloading data for tickers: {tickers} from {start_date} to {end_date}...", flush=True)
    df = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    df = df.dropna()
    return df


def eval_basket(name, df, policy):
    print(f"\n{'='*85}", flush=True)
    print(f"  DATASET: {name} ({len(df)} trading days, {df.shape[1]} assets)", flush=True)
    print(f"{'='*85}", flush=True)

    env = MultiAssetPaddedV5Env(price_df=df, max_assets=10)
    obs, _ = env.reset()
    done = False

    actions, probs_list = [], []
    while not done:
        act, probs = policy.get_action(obs, deterministic=True)
        actions.append(act)
        probs_list.append(probs)
        obs, _, done, _, _ = env.step((act, probs))

    eq = [10000.0] + env.log_wealth
    m = metrics(eq)
    cf = np.array(env.log_cash_frac)
    rp = np.array(probs_list)

    print(f"  Final Wealth:     ${m['final']:,.2f}", flush=True)
    print(f"  Total Return:     {m['return']:+.2f}%", flush=True)
    print(f"  Volatility:       {m['vol']:.2f}%", flush=True)
    print(f"  Sharpe Ratio:     {m['sharpe']:.2f}", flush=True)
    print(f"  Max Drawdown:     {m['max_dd']:.2f}%", flush=True)
    print(f"  Cash Min / Max:   {np.min(cf)*100:.1f}% / {np.max(cf)*100:.1f}% (Range: {(np.max(cf)-np.min(cf))*100:.1f}%)", flush=True)
    print(f"  Predicted Regimes: Bull={np.mean(rp[:,0])*100:.1f}%, Bear={np.mean(rp[:,1])*100:.1f}%, Sideways={np.mean(rp[:,2])*100:.1f}%", flush=True)

    # Equal-weight baseline for comparison
    ew_eq = 10000.0 * np.mean(df.values / df.values[0], axis=1)
    m_ew = metrics(ew_eq)
    print(f"\n  vs Equal-Weight Basket: {m_ew['return']:+.2f}% return, {m_ew['sharpe']:.2f} Sharpe, {m_ew['max_dd']:.2f}% maxDD", flush=True)
    print(f"  Drawdown Shielding: {abs(m_ew['max_dd']) - abs(m['max_dd']):+.2f}% drawdown reduction!", flush=True)


def main():
    obs_dim = 384
    action_dim = 11
    policy = DualHeadGatedPolicy(obs_dim=obs_dim, action_dim=action_dim)
    model_path = "./data/v0.5_rl_checkpoints/rai_v5_dual_head.pt"

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return

    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    baskets = {
        "2007-2009 Global Financial Crisis (SPY, QQQ, EEM, GLD)": (["SPY", "QQQ", "EEM", "GLD"], "2007-01-01", "2010-01-01"),
        "Tech Mega-Caps (AAPL, MSFT, NVDA, AMZN, GOOGL)": (["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"], "2015-01-01", "2024-01-01"),
        "International & Emerging Equities (EFA, EEM, FXI, EWJ)": (["EFA", "EEM", "FXI", "EWJ"], "2015-01-01", "2024-01-01"),
        "Commodities & Hard Assets (GLD, SLV, USO, DBC)": (["GLD", "SLV", "USO", "DBC"], "2015-01-01", "2024-01-01"),
        "Crypto Assets (BTC-USD, ETH-USD)": (["BTC-USD", "ETH-USD"], "2018-01-01", "2024-01-01"),
    }

    for name, (tickers, start, end) in baskets.items():
        try:
            df = download_asset_basket(tickers, start_date=start, end_date=end)
            if len(df) > 50:
                eval_basket(name, df, policy)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

if __name__ == "__main__":
    main()
