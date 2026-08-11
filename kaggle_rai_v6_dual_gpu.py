"""
====================================================================================================
🏆 RAI v6: ZERO-SHOT SIM-TO-REAL PORTFOLIO MANAGEMENT (DUAL GPU T4 x2 OPTIMIZED + ROBUST NETWORK)
====================================================================================================
Kaggle Notebook / Standalone Script

Note on Kaggle Internet Access:
  Ensure 'Internet On' is toggled ON in the Kaggle right sidebar panel under Settings -> Internet!
====================================================================================================
"""

# Install required packages
!pip install -q yfinance gymnasium torch pandas numpy matplotlib

import os, sys, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import yfinance as yf

warnings.filterwarnings("ignore")

# Automatic Dual GPU Device Selection
N_GPUS = torch.cuda.device_count()
PRIMARY_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"✓ Dual GPU Engine Initialized | GPUs: {N_GPUS} | Primary Device: {PRIMARY_DEVICE}")


# ==================================================================================================
# SECTION 1: SYNTHETIC MARKET WORLD GENERATOR (0% Real Historical Data Intake)
# ==================================================================================================
class RawPriceSyntheticEnv:
    """Procedurally generates virtual synthetic market worlds with multi-regime dynamics."""
    REGIMES = {
        'bull':     {'drift': (0.15, 0.45),  'vol': (0.10, 0.20)},
        'bear':     {'drift': (-0.50, -0.15), 'vol': (0.25, 0.60)},
        'sideways': {'drift': (-0.05, 0.05),  'vol': (0.10, 0.20)},
    }

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001):
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee
        self.features_per_step = 2 * num_assets + 2
        self.reset()

    def _generate_raw_prices(self):
        total_T = self.episode_len + self.history_len + 10
        n_seg = np.random.randint(3, 7)
        keys = list(self.REGIMES.keys())
        seq = [keys[np.random.randint(len(keys))] for _ in range(n_seg)]

        dur = np.random.dirichlet(np.ones(n_seg) * 2.0) * total_T
        dur = np.maximum(dur.astype(int), 30)
        dur[-1] = total_T - sum(dur[:-1])

        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)

        for asset in range(self.num_assets):
            p = np.random.uniform(20.0, 300.0)
            series = [p]
            day = 0
            for reg, d in zip(seq, dur):
                params = self.REGIMES[reg]
                drift = np.random.uniform(*params['drift']) + np.random.uniform(-0.04, 0.04)
                vol = np.random.uniform(*params['vol']) * np.random.uniform(0.85, 1.15)
                mu = drift / 252.0; sigma = vol / np.sqrt(252.0)
                for _ in range(max(0, min(d, total_T - day - 1))):
                    p = max(0.01, p * np.exp((mu - 0.5*sigma**2) + sigma * np.random.standard_normal()))
                    series.append(p)
                    day += 1
                    if day >= total_T: break
                if day >= total_T: break
            while len(series) < total_T: series.append(series[-1])
            prices[:, asset] = series[:total_T]

        return prices

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.prices = self._generate_raw_prices()
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.5
        init_p = self.prices[self.current_step]
        self.shares = (self.initial_cash * 0.5 / self.num_assets) / init_p
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs()

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p = self.prices[t]; p_prev = self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        norm_prices = p / self.prices[self.start]
        log_rets = np.log(p / np.maximum(1e-4, p_prev))
        cash_ratio = self.cash / w
        dd = np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1.0, 0.0)
        return np.concatenate([norm_prices, log_rets, [cash_ratio, dd]]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        cash_logit = np.clip(action[0], -5.0, 5.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash_frac
        asset_logits = action[1:]
        exp_a = np.exp(asset_logits - np.max(asset_logits))
        target_asset_w = (exp_a / np.sum(exp_a)) * stock_portion

        prices = self.prices[self.current_step]
        wealth = max(1e-4, self._wealth())
        cur_asset_w = (self.shares * prices) / wealth
        cur_cash_frac = self.cash / wealth

        drift = abs(cur_cash_frac - target_cash_frac) + np.sum(np.abs(cur_asset_w - target_asset_w))

        if drift > 0.03:
            t_vol = abs(self.cash - wealth * target_cash_frac) + np.sum(np.abs(self.shares * prices - wealth * target_asset_w))
            net = max(1e-4, wealth - t_vol * self.fee)
            self.cash = net * target_cash_frac
            self.shares = (net * target_asset_w) / np.maximum(1e-4, prices)

        self.current_step += 1
        self.steps_done += 1
        new_wealth = self._wealth()
        self.peak_wealth = max(self.peak_wealth, new_wealth)

        daily_ret = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        reward = daily_ret * 5.0
        if daily_ret < 0: reward *= 2.0
        drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
        if drawdown < -0.10: reward += drawdown * 2.0

        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))

        return self._flat_obs(), reward, done, {"portfolio_value": new_wealth}


# ==================================================================================================
# SECTION 2: DUAL GPU NEURAL ARCHITECTURE
# ==================================================================================================
class DeepEndToEndTradingNet(nn.Module):
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64, nhead=2):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step

        self.conv1d = nn.Sequential(
            nn.Conv1d(features_per_step, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=128,
            dropout=0.05, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * history_len, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
        )
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)

        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step)
        x_trans = x.transpose(1, 2)
        conv_out = self.conv1d(x_trans).transpose(1, 2)
        trans_out = self.transformer(conv_out)
        flat = trans_out.reshape(b, -1)
        feat = self.fc(flat)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).to(PRIMARY_DEVICE).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs).to(PRIMARY_DEVICE)
            mean, _ = self.forward(flat_obs)
            if deterministic:
                return mean.cpu().numpy().squeeze(0)
            dist = Normal(mean, torch.exp(self.log_std))
            return dist.sample().cpu().numpy().squeeze(0)


# ==================================================================================================
# SECTION 3: DUAL GPU ACCELERATED PPO TRAINING (DataParallel)
# ==================================================================================================
def train_rai_v6_dual_gpu(total_steps=50_000, seed=42):
    print("=" * 100)
    print(f"  ⚡ TRAINING RAI v6 ON DUAL KAGGLE GPUs: Detected {N_GPUS} x NVIDIA T4 GPUs ({PRIMARY_DEVICE})")
    print("=" * 100)
    t0 = time.time()

    env = RawPriceSyntheticEnv(num_assets=10, episode_len=504)
    model = DeepEndToEndTradingNet().to(PRIMARY_DEVICE)

    if N_GPUS > 1:
        print(f"  ✓ DataParallel Activated across {N_GPUS} NVIDIA T4 GPUs!")
        parallel_model = nn.DataParallel(model)
    else:
        parallel_model = model

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    obs = env.reset(seed=seed)
    step = 0

    while step < total_steps:
        obs_b, act_b, rew_b, val_b, logp_b = [], [], [], [], []
        
        # Fast single-GPU rollout collection
        for _ in range(1024):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(PRIMARY_DEVICE)
            with torch.no_grad():
                mean, val = model(obs_t)
                dist = Normal(mean, torch.exp(model.log_std))
                action = dist.sample()
                logp = dist.log_prob(action).sum(dim=-1)
            act_np = action.squeeze(0).cpu().numpy()
            nobs, rew, done, _ = env.step(act_np)
            obs_b.append(obs); act_b.append(act_np); rew_b.append(rew); val_b.append(val.item()); logp_b.append(logp.item())
            obs = nobs
            step += 1
            if done: obs = env.reset()

        with torch.no_grad():
            _, nval = model(torch.FloatTensor(obs).unsqueeze(0).to(PRIMARY_DEVICE))
            nval = nval.item()

        r = np.array(rew_b)
        v = np.array(val_b + [nval])
        delta = r + 0.99 * v[1:] - v[:-1]
        adv = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            gae = delta[t] + 0.99 * 0.95 * gae
            adv[t] = gae
        ret = adv + v[:-1]

        o_t = torch.FloatTensor(np.array(obs_b)).to(PRIMARY_DEVICE)
        a_t = torch.FloatTensor(np.array(act_b)).to(PRIMARY_DEVICE)
        adv_t = torch.FloatTensor(adv).to(PRIMARY_DEVICE)
        ret_t = torch.FloatTensor(ret).to(PRIMARY_DEVICE)
        old_logp_t = torch.FloatTensor(np.array(logp_b)).to(PRIMARY_DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # Dual-GPU DataParallel PPO mini-batch updates
        for _ in range(4):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), 256):
                b_idx = idx[s:s + 256]
                mean, val = parallel_model(o_t[b_idx])
                dist = Normal(mean, torch.exp(model.log_std))
                new_logp = dist.log_prob(a_t[b_idx]).sum(dim=-1)
                ratio = torch.exp(new_logp - old_logp_t[b_idx])
                surr1 = ratio * adv_t[b_idx]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[b_idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(val.squeeze(-1), ret_t[b_idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if step % 10000 == 0:
            print(f"    Step {step:5d} / {total_steps} | Elapsed: {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"  ✅ Dual GPU Training complete in {elapsed:.1f} seconds! Policy frozen for Zero-Shot evaluation.\n")
    model.eval()
    return model


# ==================================================================================================
# SECTION 4: REAL-WORLD ZERO-SHOT EVALUATION (WITH ROBUST NETWORK RETRY)
# ==================================================================================================
def fetch_market_data_robust(tickers):
    """Fetches real market prices with retry logic & friendly Kaggle internet guidance."""
    print(f"  📥 Fetching live market data for {len(tickers)} assets: {', '.join(tickers)}...")
    for attempt in range(1, 4):
        try:
            df = yf.download(tickers, period="2y", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            df = df.dropna().ffill().bfill()
            if not df.empty and len(df) > 50:
                print(f"  ✓ Market data loaded successfully! ({len(df)} trading days)")
                return df
        except Exception as e:
            print(f"  ⚠️ Network attempt {attempt}/3 failed: {e}")
            time.sleep(2)

    raise RuntimeError(
        "\n❌ ERROR: Could not connect to Yahoo Finance.\n"
        "👉 KAGGLE INSTRUCTION: Please toggle 'Internet On' in Kaggle Notebook Options!\n"
        "   (Right sidebar -> Notebook Options -> Internet -> Toggle ON)"
    )


def evaluate_zero_shot_real(model, tickers):
    print("=" * 100)
    print(f"  📈 ZERO-SHOT SIM-TO-REAL EVALUATION ON REAL MARKET DATA")
    print("=" * 100)

    df = fetch_market_data_robust(tickers)
    prices = df.values
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    T, N = prices.shape

    initial_wealth = 10000.0
    cash = initial_wealth * 0.05
    init_p = prices[30]
    shares = (initial_wealth * 0.95 / N) / init_p
    peak = initial_wealth
    eq = [initial_wealth]

    obs_h = []
    for t in range(30):
        p, pp = prices[t], prices[max(0, t - 1)]
        w = max(1e-4, cash + np.sum(shares * p))
        obs_h.append(np.concatenate([
            p / prices[30],
            np.log(p / np.maximum(1e-4, pp)),
            [cash / w, np.clip((w - peak) / max(1e-4, peak), -1, 0)]
        ]).astype(np.float32))

    for t in range(30, T):
        flat_obs = np.concatenate(obs_h).astype(np.float32)
        act = model.get_action(flat_obs, deterministic=True)

        cash_logit = np.clip(act[0], -5.0, 5.0)
        tc = 1.0 / (1.0 + np.exp(-cash_logit))
        ts = 1.0 - tc
        ea = np.exp(act[1:] - np.max(act[1:]))
        taw = (ea / ea.sum()) * ts

        p = prices[t].copy()
        w = max(1e-4, cash + np.sum(shares * p))
        caw = (shares * p) / w
        ccf = cash / w

        if abs(ccf - tc) + np.sum(np.abs(caw - taw)) > 0.03:
            tv = abs(cash - w * tc) + np.sum(np.abs(shares * p - w * taw))
            net = max(1e-4, w - tv * 0.001)
            cash = net * tc
            shares = (net * taw) / np.maximum(1e-4, p)

        nw = cash + np.sum(shares * prices[t])
        peak = max(peak, nw)
        eq.append(nw)

        pp = prices[t - 1]
        obs_h.pop(0)
        obs_h.append(np.concatenate([
            prices[t] / prices[30],
            np.log(prices[t] / np.maximum(1e-4, pp)),
            [cash / max(1e-4, nw), np.clip((nw - peak) / max(1e-4, peak), -1, 0)]
        ]).astype(np.float32))

    eq_a = np.array(eq)
    r = np.diff(eq_a) / np.maximum(1e-8, eq_a[:-1])
    pk = np.maximum.accumulate(eq_a)
    ret_pct = (eq_a[-1] / eq_a[0] - 1) * 100
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 1e-8 else 0.
    max_dd = float(np.min((eq_a - pk) / pk) * 100)

    print(f"  Out-of-Sample Test Days : {T-30} trading days")
    print(f"  Cumulative OOS Return  : {ret_pct:>+7.2f}%")
    print(f"  Annualized Sharpe Ratio : {sharpe:>7.2f}")
    print(f"  Maximum Drawdown       : {max_dd:>7.2f}%\n")

    # Latest Live Allocation Output
    last_act = model.get_action(np.concatenate(obs_h).astype(np.float32), deterministic=True)
    c_logit = np.clip(last_act[0], -5.0, 5.0)
    c_frac = 1.0 / (1.0 + np.exp(-c_logit))
    s_portion = 1.0 - c_frac
    e_a = np.exp(last_act[1:] - np.max(last_act[1:]))
    a_weights = (e_a / np.sum(e_a)) * s_portion

    print(f"{'═'*100}")
    print(f"  CURRENT REAL MARKET PORTFOLIO ALLOCATION DECISION (Date: {latest_date})")
    print(f"{'═'*100}")
    print(f"  Cash Reserves Allocation : {c_frac*100:>6.2f}%")
    print(f"  Equities Allocation      : {s_portion*100:>6.2f}%\n")

    print(f"  {'Asset Ticker':<15} | {'Asset Weight (%)':<20} | {'Current Market Price ($)':<25}")
    print(f"  {'-'*65}")
    for i, ticker in enumerate(df.columns):
        print(f"  {ticker:<15} | {a_weights[i]*100:>18.2f}% | ${prices[-1, i]:>23.2f}")
    print(f"  {'-'*65}")
    print(f"  Total Portfolio Allocation : {(c_frac + s_portion)*100:.2f}%\n")
    print("=" * 100)


# Execute full script
if __name__ == "__main__":
    ASSET_UNIVERSE = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TLT", "GLD"]
    trained_model = train_rai_v6_dual_gpu(total_steps=50_000, seed=SEED)
    evaluate_zero_shot_real(trained_model, ASSET_UNIVERSE)
