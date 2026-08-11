"""
═══════════════════════════════════════════════════════════════════════════════
  RAI v7: SCENARIO-RANDOMIZED FACTOR-DRIVEN GENERATOR EXPERIMENT
  ════════════════════════════════════════════════════════════════
  Research Question:
    "Is decision-relevant scenario diversity—rather than statistical realism—
     the key to zero-shot policy transfer and regime-adaptive allocation?"

  Core Paradigm:
    - Domain & Parameter Randomization across 100,000+ synthetic worlds
    - 4 Latent Factors: Equity, Bond, Commodity, Dollar/Liquidity
    - 7 Causal Macro Scenarios:
        1. Bull Run
        2. Flight-to-Safety Crisis (Stocks ↓, Bonds/Gold ↑)
        3. Liquidity Crunch (Everything ↓, Cash/USD ↑)
        4. Stagflation (Commodities ↑, Stocks/Bonds ↓)
        5. False Recovery & W-Crash
        6. Flash Crash & V-Rebound
        7. Sideways Chop
    - Randomized factor loadings, vol spikes, jump sizes, and segment lengths

  Experimental Control:
    - Identical RAI v6 hybrid architecture (Conv1D + Transformer)
    - Identical PPO hyperparameters & 100k step budget
    - RAI v6 frozen baseline vs RAI v7 Scenario-Randomized models
    - Evaluated on identical real-market out-of-sample & untouched holdout sets
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, warnings, multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
from scipy import stats

warnings.filterwarnings('ignore')
torch.set_num_threads(1)  # Single-threaded per worker to prevent CPU core thrashing

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "v7_scenarios")
NUM_WORKERS = 8
N_SEEDS = 5
TOTAL_STEPS = 100_000

os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE (Strictly identical RAI v6 Hybrid)
# ═══════════════════════════════════════════════════════════════════════════════

class DeepEndToEndTradingNet(nn.Module):
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, embed_dim=64, nhead=2):
        super().__init__()
        self.history_len = history_len
        self.features_per_step = features_per_step
        self.conv1d = nn.Sequential(
            nn.Conv1d(features_per_step, 32, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1), nn.LeakyReLU(0.1),
        )
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=128, dropout=0.05, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.fc_features = nn.Sequential(nn.Linear(embed_dim, 128), nn.LeakyReLU(0.1))
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, flat_obs):
        b = flat_obs.shape[0]
        x = flat_obs.reshape(b, self.history_len, self.features_per_step)
        x_conv = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_trans = self.transformer(x_conv)
        feat = self.fc_features(x_trans.mean(dim=1))
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            mean, _ = self.forward(flat_obs)
            return mean.cpu().numpy().squeeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  RAI v7 SCENARIO-RANDOMIZED FACTOR GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaScenarioEnvV7(gym.Env):
    """
    Scenario-Randomized Synthetic Environment with Latent Causal Macro Factors.
    """
    SCENARIO_TYPES = [
        'BULL_RUN',
        'FLIGHT_TO_SAFETY',
        'LIQUIDITY_CRUNCH',
        'INFLATION_STAGFLATION',
        'FALSE_RECOVERY',
        'FLASH_CRASH',
        'SIDEWAYS_CHOP',
    ]

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001):
        super().__init__()
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee

        # Observation & Action space identical to v6
        self.features_per_step = 2 * num_assets + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (history_len * self.features_per_step,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (num_assets + 1,), np.float32)
        self.reset()

    def _sample_random_world(self):
        """Generates a complete synthetic world with domain & parameter randomization."""
        T = self.episode_len + self.history_len + 10

        # Domain Randomization: Randomize asset factor loadings per episode
        # 4 Latent Factors: [Equity, Bond, Commodity, USD]
        beta_matrix = np.zeros((self.num_assets, 4))
        for a in range(self.num_assets):
            if a < 4:  # Equities (Assets 0-3)
                beta_matrix[a] = [self.np_random.uniform(0.8, 1.4), self.np_random.uniform(-0.4, 0.1), self.np_random.uniform(-0.2, 0.3), self.np_random.uniform(-0.5, -0.1)]
            elif a < 6:  # Bonds (Assets 4-5)
                beta_matrix[a] = [self.np_random.uniform(-0.3, 0.1), self.np_random.uniform(0.8, 1.3), self.np_random.uniform(-0.3, 0.1), self.np_random.uniform(-0.2, 0.2)]
            elif a < 9:  # Commodities (Assets 6-8)
                beta_matrix[a] = [self.np_random.uniform(0.1, 0.5), self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(0.8, 1.5), self.np_random.uniform(-0.8, -0.3)]
            else:  # USD / Cash Equivalent (Asset 9)
                beta_matrix[a] = [self.np_random.uniform(-0.4, -0.1), self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.6, -0.2), self.np_random.uniform(0.9, 1.4)]

        # Sample sequence of causal scenarios
        n_segments = self.np_random.integers(3, 8)
        scenario_sequence = [self.SCENARIO_TYPES[self.np_random.integers(len(self.SCENARIO_TYPES))] for _ in range(n_segments)]

        # Dirichlet segment lengths with domain noise
        durations = self.np_random.dirichlet(np.ones(n_segments) * self.np_random.uniform(1.5, 3.5)) * T
        durations = np.maximum(durations.astype(int), 30)
        durations[-1] = T - sum(durations[:-1])

        prices = np.zeros((T, self.num_assets), np.float64)
        init_p = self.np_random.uniform(20., 300., self.num_assets)
        prices[0] = init_p

        t = 1
        for sc, d in zip(scenario_sequence, durations):
            if t >= T: break
            seg_len = max(1, min(d, T - t))

            # Macro factor drifts & vols per scenario
            if sc == 'BULL_RUN':
                f_drift = np.array([0.25, 0.02, 0.10, -0.05]) / 252.
                f_vol = np.array([0.12, 0.08, 0.14, 0.06]) / np.sqrt(252.)
            elif sc == 'FLIGHT_TO_SAFETY':
                # Stocks crash (-45%), Bonds & Gold rally (+25%), USD up
                f_drift = np.array([-0.45, 0.25, 0.15, 0.15]) / 252.
                f_vol = np.array([0.35, 0.14, 0.25, 0.12]) / np.sqrt(252.)
            elif sc == 'LIQUIDITY_CRUNCH':
                # Everything crashes, Cash/USD surges (+30%)
                f_drift = np.array([-0.55, -0.20, -0.40, 0.35]) / 252.
                f_vol = np.array([0.45, 0.25, 0.40, 0.20]) / np.sqrt(252.)
            elif sc == 'INFLATION_STAGFLATION':
                # Commodities surge (+45%), Equities & Bonds drop (-20%)
                f_drift = np.array([-0.20, -0.25, 0.45, -0.15]) / 252.
                f_vol = np.array([0.22, 0.18, 0.30, 0.15]) / np.sqrt(252.)
            elif sc == 'FALSE_RECOVERY':
                # First half bull (+30%), second half sharp collapse (-50%)
                f_drift = np.array([0.35, -0.05, 0.10, -0.10]) / 252.
                f_vol = np.array([0.30, 0.15, 0.25, 0.15]) / np.sqrt(252.)
            elif sc == 'FLASH_CRASH':
                # Sudden 2-day 15% shock followed by rapid recovery
                f_drift = np.array([0.10, 0.00, 0.05, 0.00]) / 252.
                f_vol = np.array([0.15, 0.08, 0.15, 0.08]) / np.sqrt(252.)
            else: # SIDEWAYS_CHOP
                f_drift = np.array([0.00, 0.00, 0.00, 0.00]) / 252.
                f_vol = np.array([0.16, 0.10, 0.18, 0.09]) / np.sqrt(252.)

            # Parameter Randomization: Add noise to factor parameters per segment
            f_drift = f_drift + self.np_random.uniform(-0.03, 0.03, 4) / 252.
            f_vol = f_vol * self.np_random.uniform(0.85, 1.2, 4)

            for step in range(seg_len):
                if t >= T: break

                # Handle False Recovery split
                current_f_drift = f_drift.copy()
                if sc == 'FALSE_RECOVERY' and step > seg_len // 2:
                    current_f_drift = np.array([-0.50, 0.15, -0.20, 0.20]) / 252.

                # Sample independent macro factor innovations
                z_factors = self.np_random.standard_normal(4)
                factor_returns = current_f_drift + f_vol * z_factors

                # Compute asset returns from factor loadings + idiosyncratic noise
                idiosyncratic_vol = self.np_random.uniform(0.05, 0.15, self.num_assets) / np.sqrt(252.)
                z_idio = self.np_random.standard_normal(self.num_assets)

                asset_returns = beta_matrix @ factor_returns + idiosyncratic_vol * z_idio

                # Flash crash injection
                if sc == 'FLASH_CRASH' and step == seg_len // 3:
                    asset_returns[:4] -= 0.08  # Equities drop 8% in a day

                prices[t] = np.maximum(0.01, prices[t-1] * np.exp(asset_returns))
                t += 1

        while t < T:
            prices[t] = prices[t-1]
            t += 1

        return prices

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prices = self._sample_random_world()
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.05
        self.shares = (self.initial_cash * 0.95 / self.num_assets) / self.prices[self.current_step]
        self.peak_wealth = self.last_wealth = self.initial_cash
        self.steps_done = 0
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs(), {}

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p, pp = self.prices[t], self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        return np.concatenate([p/self.prices[self.start], np.log(p/np.maximum(1e-4, pp)),
                               [self.cash/w, np.clip((w-self.peak_wealth)/max(1e-4, self.peak_wealth), -1, 0)]]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        cl = np.clip(action[0] - 2.5, -8., 3.)
        tc = 1./(1.+np.exp(-cl)); ts = 1.-tc
        ea = np.exp(action[1:] - np.max(action[1:])); taw = (ea/ea.sum()) * ts
        p, w = self.prices[self.current_step], max(1e-4, self._wealth())
        caw, ccf = (self.shares*p)/w, self.cash/w
        if abs(ccf-tc)+np.sum(np.abs(caw-taw)) > 0.03:
            tv = abs(self.cash-w*tc)+np.sum(np.abs(self.shares*p-w*taw))
            net = max(1e-4, w-tv*self.fee); self.cash = net*tc; self.shares = (net*taw)/np.maximum(1e-4, p)
        self.current_step += 1; self.steps_done += 1
        nw = self._wealth(); self.peak_wealth = max(self.peak_wealth, nw)
        reward = ((nw-self.last_wealth)/max(1e-4, self.last_wealth)) * 20.
        done = self.current_step >= self.prices.shape[0]-1 or self.steps_done >= self.episode_len
        self.last_wealth = nw; self.obs_history.pop(0); self.obs_history.append(self._obs_at(self.current_step))
        return self._flat_obs(), reward, done, False, {}


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO TRAINING WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def train_ppo(model, env, seed, total_steps=100_000):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs, _ = env.reset(seed=seed); step = 0
    while step < total_steps:
        obs_b, act_b, rew_b, val_b, logp_b = [], [], [], [], []
        for _ in range(1024):
            ot = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                m, v = model(ot); d = Normal(m, torch.exp(model.log_std))
                a = d.sample(); lp = d.log_prob(a).sum(-1)
            an = a.squeeze(0).numpy(); no, r, dn, _, _ = env.step(an)
            obs_b.append(obs); act_b.append(an); rew_b.append(r)
            val_b.append(v.item()); logp_b.append(lp.item())
            obs = no; step += 1
            if dn: obs, _ = env.reset()
        with torch.no_grad():
            _, nv = model(torch.FloatTensor(obs).unsqueeze(0)); nv = nv.item()
        r_a = np.array(rew_b); v_a = np.array(val_b+[nv])
        delta = r_a + 0.99*v_a[1:] - v_a[:-1]
        adv = np.zeros_like(r_a); gae = 0.
        for t in reversed(range(len(r_a))): gae = delta[t]+0.99*0.95*gae; adv[t] = gae
        ret = adv + v_a[:-1]
        ot = torch.FloatTensor(np.array(obs_b)); at = torch.FloatTensor(np.array(act_b))
        advt = torch.FloatTensor(adv); rett = torch.FloatTensor(ret); oldt = torch.FloatTensor(np.array(logp_b))
        advt = (advt - advt.mean())/(advt.std()+1e-8)
        for _ in range(4):
            idx = np.random.permutation(len(obs_b))
            for s in range(0, len(obs_b), 64):
                bi = idx[s:s+64]; m2, v2 = model(ot[bi])
                d2 = Normal(m2, torch.exp(model.log_std)); nlp = d2.log_prob(at[bi]).sum(-1)
                ratio = torch.exp(nlp - oldt[bi])
                loss = -torch.min(ratio*advt[bi], torch.clamp(ratio, .8, 1.2)*advt[bi]).mean() + .5*F.mse_loss(v2.squeeze(-1), rett[bi])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), .5); opt.step()
    return model

def _worker_train_v7(args):
    seed, path, total_steps = args
    if os.path.exists(path):
        return (seed, path, "exists", 0.)
    t0 = time.time()
    model = DeepEndToEndTradingNet()
    env = AlphaScenarioEnvV7()
    model = train_ppo(model, env, seed=seed, total_steps=total_steps)
    torch.save(model.state_dict(), path)
    return (seed, path, "trained", time.time()-t0)


# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(eq_list):
    eq = np.array(eq_list, np.float64)
    if len(eq) < 2: return {"final": eq[-1], "return_pct": 0, "vol_pct": 0, "sharpe": 0, "max_dd_pct": 0}
    r = (eq[1:]-eq[:-1])/np.maximum(1e-8, eq[:-1])
    pk = np.maximum.accumulate(eq)
    return {"final": float(eq[-1]), "return_pct": float((eq[-1]/eq[0]-1)*100),
            "vol_pct": float(np.std(r)*np.sqrt(252)*100),
            "sharpe": float(np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r) > 1e-8 else 0.,
            "max_dd_pct": float(np.min((eq-pk)/pk)*100)}

def eval_model_on_prices(model, prices_raw, fee_bps=5, slippage_pct=0.02):
    T, N = prices_raw.shape
    if T < 35: return compute_metrics([10000.])
    fee = fee_bps/10000.; cash = 500.; init_p = prices_raw[30]
    shares = (9500./N)/init_p; peak = 10000.; eq = [10000.]
    obs_h = []
    for t in range(30):
        p, pp = prices_raw[t], prices_raw[max(0, t-1)]
        np_ = np.pad(p/prices_raw[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.append(np.concatenate([np_, lr, [.05, 0.]]).astype(np.float32))
    for t in range(30, T):
        act = model.get_action(np.concatenate(obs_h).astype(np.float32), deterministic=True)
        cl = np.clip(act[0]-2.5, -8., 3.); tc = 1./(1.+np.exp(-cl)); ts = 1.-tc
        n = min(N, 10); ea = np.exp(act[1:1+n]-np.max(act[1:1+n])); taw = (ea/ea.sum())*ts
        p = prices_raw[t].copy()
        w = max(1e-4, cash+np.sum(shares*p)); caw = (shares*p)/w; ccf = cash/w
        if abs(ccf-tc)+np.sum(np.abs(caw-taw)) > 0.03:
            tv = abs(cash-w*tc)+np.sum(np.abs(shares*p-w*taw))
            net = max(1e-4, w-tv*fee); cash = net*tc; shares = (net*taw)/np.maximum(1e-4, p)
        nw = cash+np.sum(shares*p); peak = max(peak, nw); eq.append(nw)
        pp = prices_raw[t-1]
        np_ = np.pad(p/prices_raw[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.pop(0); obs_h.append(np.concatenate([np_, lr, [cash/max(1e-4,nw), np.clip((nw-peak)/max(1e-4,peak),-1,0)]]).astype(np.float32))
    return compute_metrics(eq)

def comp_stats(vals):
    a = np.array(vals); n = len(a); mu = np.mean(a)
    sd = np.std(a, ddof=1) if n > 1 else 0.; se = sd/np.sqrt(n) if n > 1 else 0.
    ci = stats.t.interval(0.95, df=max(1,n-1), loc=mu, scale=max(1e-12,se)) if n > 1 else (mu, mu)
    return {"mean": float(mu), "std": float(sd), "ci95": [float(ci[0]), float(ci[1])],
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": n}

def welch_test(a, b):
    a, b = np.array(a), np.array(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    ps = np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    d = (a.mean()-b.mean())/ps if ps > 1e-8 else 0.
    return {"t": float(t), "p": float(p), "d": float(d)}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()
    W = 120
    print("="*W)
    print("  RAI v7 — SCENARIO-RANDOMIZED GENERATOR EXPERIMENT")
    print("  Testing Decision-Relevant Scenario Diversity vs Frozen RAI v6 Control")
    print("="*W, flush=True)

    # 1. Train 5 seeds of RAI v7 Scenario-Randomized Generator
    jobs = [(seed, os.path.join(RESULTS_DIR, "models", f"rai_v7_scenario_seed_{seed:02d}.pt"), TOTAL_STEPS)
            for seed in range(1, N_SEEDS + 1)]

    print(f"\n  Dispatching {len(jobs)} RAI v7 training jobs across {NUM_WORKERS} workers...", flush=True)
    t1 = time.time()
    with mp.Pool(processes=NUM_WORKERS) as pool:
        outcomes = pool.map(_worker_train_v7, jobs)

    trained = sum(1 for o in outcomes if o[2] == "trained")
    cached = sum(1 for o in outcomes if o[2] == "exists")
    print(f"  ✓ RAI v7 training complete: {trained} trained, {cached} cached in {time.time()-t1:.0f}s", flush=True)

    # 2. Load Real-Market Evaluation Datasets
    TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]

    local_test_path = os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "test_prices.csv")
    eval_data = {}

    if os.path.exists(local_test_path):
        df_test = pd.read_csv(local_test_path, index_col=0, parse_dates=True)
        eval_data["2020-2024_OOS"] = df_test.values

    import yfinance as yf
    extra_periods = {
        "2024-2026_Holdout": ("2024-06-01", "2026-08-08"),
        "Full_2020-2026": ("2020-01-01", "2026-08-08")
    }

    for p_name, (start_d, end_d) in extra_periods.items():
        try:
            df = yf.download(TICKERS, start=start_d, end=end_d, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            df = df[TICKERS].dropna()
            if len(df) >= 35:
                eval_data[p_name] = df.values
        except Exception as e:
            print(f"  ⚠ Download issue for {p_name}: {e}", flush=True)

    # 3. Evaluate RAI v6 (Frozen Control) vs RAI v7 (Scenario Diversity)
    v6_models_dir = os.path.join(PROJECT_ROOT, "data", "robustness", "seeds")
    v7_models_dir = os.path.join(RESULTS_DIR, "models")

    v6_eval = {p_name: [] for p_name in eval_data}
    v7_eval = {p_name: [] for p_name in eval_data}

    # Evaluate v6 seeds
    for seed in range(1, N_SEEDS + 1):
        p_v6 = os.path.join(v6_models_dir, f"rai_v6_seed_{seed:02d}.pt")
        if os.path.exists(p_v6):
            m6 = DeepEndToEndTradingNet()
            m6.load_state_dict(torch.load(p_v6, weights_only=True))
            m6.eval()
            for p_name, p_vals in eval_data.items():
                v6_eval[p_name].append(eval_model_on_prices(m6, p_vals))

    # Evaluate v7 seeds
    for seed in range(1, N_SEEDS + 1):
        p_v7 = os.path.join(v7_models_dir, f"rai_v7_scenario_seed_{seed:02d}.pt")
        if os.path.exists(p_v7):
            m7 = DeepEndToEndTradingNet()
            m7.load_state_dict(torch.load(p_v7, weights_only=True))
            m7.eval()
            for p_name, p_vals in eval_data.items():
                v7_eval[p_name].append(eval_model_on_prices(m7, p_vals))

    # 4. Print Comparative Results
    print(f"\n{'═'*W}")
    print(f"  COMPARATIVE RESULTS: RAI v6 Baseline Control vs RAI v7 Scenario-Randomized Generator")
    print(f"{'═'*W}")

    master_summary = {}

    for p_name in eval_data:
        print(f"\n  ► TEST PERIOD: {p_name}")
        print(f"  {'Model Variant':<32} | {'Return (%)':<24} | {'Sharpe':<20} | {'Max DD (%)':<22}", flush=True)
        print(f"  {'-'*110}", flush=True)

        m6_rets = [m["return_pct"] for m in v6_eval[p_name]]
        m6_sh = [m["sharpe"] for m in v6_eval[p_name]]
        m6_dd = [m["max_dd_pct"] for m in v6_eval[p_name]]

        m7_rets = [m["return_pct"] for m in v7_eval[p_name]]
        m7_sh = [m["sharpe"] for m in v7_eval[p_name]]
        m7_dd = [m["max_dd_pct"] for m in v7_eval[p_name]]

        r6, s6, d6 = comp_stats(m6_rets), comp_stats(m6_sh), comp_stats(m6_dd)
        r7, s7, d7 = comp_stats(m7_rets), comp_stats(m7_sh), comp_stats(m7_dd)

        tt = welch_test(m7_rets, m6_rets)
        p_val = tt['p']
        sig_flag = "***" if p_val < .001 else "**" if p_val < .01 else "*" if p_val < .05 else "ns"
        sig_str = f" (vs v6: p={p_val:.3f} {sig_flag}, d={tt['d']:+.2f})"

        print(f"  {'RAI v6 (Frozen Control)':<32} | {r6['mean']:>+6.2f}±{r6['std']:<4.2f}%{'':<12} | "
              f"{s6['mean']:>+5.2f}±{s6['std']:<4.2f} | {d6['mean']:>+6.2f}±{d6['std']:<4.2f}%", flush=True)
        print(f"  {'RAI v7 (Scenario Diversity)':<32} | {r7['mean']:>+6.2f}±{r7['std']:<4.2f}%{sig_str:<12} | "
              f"{s7['mean']:>+5.2f}±{s7['std']:<4.2f} | {d7['mean']:>+6.2f}±{d7['std']:<4.2f}%", flush=True)

        master_summary[p_name] = {
            "v6_control": {"return": r6, "sharpe": s6, "max_dd": d6},
            "v7_scenarios": {"return": r7, "sharpe": s7, "max_dd": d7},
            "welch_test": tt
        }

    # Save JSON Output
    out_path = os.path.join(RESULTS_DIR, "v7_scenario_results.json")
    with open(out_path, 'w') as f:
        json.dump(master_summary, f, indent=2, default=str)

    print(f"\n{'═'*W}")
    print(f"  ✅ EXPERIMENT COMPLETE — {time.time()-T0:.1f} seconds")
    print(f"  Results saved to: {out_path}")
    print(f"{'═'*W}\n", flush=True)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
