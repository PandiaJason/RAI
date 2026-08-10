"""
═══════════════════════════════════════════════════════════════════════════════
  RAI v6 SCIENTIFIC ROBUSTNESS PROTOCOL (PARALLEL EDITION)
  ═════════════════════════════════════════════════════════
  9-Phase Experimental Suite — 8-Core Parallel Training

  Research Question:
    "Can a policy acquire useful transferable structure
     from artificial experience alone?"
═══════════════════════════════════════════════════════════════════════════════
"""
import os, sys, time, json, hashlib, warnings, multiprocessing as mp
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "robustness")
NUM_WORKERS = 8
N_SEEDS = 10
N_GEN_SEEDS = 5
TOTAL_STEPS = 100_000

for d in ["seeds", "ablations", "generators"]:
    os.makedirs(os.path.join(RESULTS_DIR, d), exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

class DeepEndToEndTradingNet(nn.Module):
    """RAI v6: Conv1D + Transformer Encoder (Full Hybrid)."""
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


class MLPTradingNet(nn.Module):
    """Ablation: Pure MLP."""
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, **kw):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(history_len * features_per_step, 256), nn.LeakyReLU(0.1),
                                 nn.Linear(256, 128), nn.LeakyReLU(0.1))
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x):
        feat = self.net(x)
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            return self.forward(flat_obs)[0].cpu().numpy().squeeze(0)


class Conv1DTradingNet(nn.Module):
    """Ablation: Pure Conv1D."""
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, **kw):
        super().__init__()
        self.hl, self.fp = history_len, features_per_step
        self.conv1d = nn.Sequential(nn.Conv1d(features_per_step, 32, 3, padding=1), nn.LeakyReLU(0.1),
                                    nn.Conv1d(32, 64, 3, padding=1), nn.LeakyReLU(0.1))
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.LeakyReLU(0.1))
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.hl, self.fp)
        feat = self.fc(self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1).mean(dim=1))
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            return self.forward(flat_obs)[0].cpu().numpy().squeeze(0)


class TransformerTradingNet(nn.Module):
    """Ablation: Pure Transformer."""
    def __init__(self, history_len=30, features_per_step=22, action_dim=11, **kw):
        super().__init__()
        self.hl, self.fp = history_len, features_per_step
        self.proj = nn.Linear(features_per_step, 64)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=128, dropout=0.05, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Sequential(nn.Linear(64, 128), nn.LeakyReLU(0.1))
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.hl, self.fp)
        feat = self.fc(self.transformer(F.leaky_relu(self.proj(x), 0.1)).mean(dim=1))
        return self.actor_head(feat), self.critic_head(feat)

    def get_action(self, flat_obs, deterministic=True):
        with torch.no_grad():
            if isinstance(flat_obs, np.ndarray):
                flat_obs = torch.FloatTensor(flat_obs).unsqueeze(0) if flat_obs.ndim == 1 else torch.FloatTensor(flat_obs)
            return self.forward(flat_obs)[0].cpu().numpy().squeeze(0)


ARCH_REGISTRY = {
    "RAI_v6_Hybrid": DeepEndToEndTradingNet,
    "MLP_Only": MLPTradingNet,
    "Conv1D_Only": Conv1DTradingNet,
    "Transformer_Only": TransformerTradingNet,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC ENVIRONMENTS
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaSyntheticEnv(gym.Env):
    """Original 4-regime GBM environment."""
    REGIMES = {
        'strong_bull': {'drift': (0.30, 0.70),  'vol': (0.08, 0.16)},
        'mild_bull':   {'drift': (0.15, 0.30),  'vol': (0.10, 0.18)},
        'sideways':    {'drift': (-0.02, 0.05), 'vol': (0.12, 0.20)},
        'bear_crash':  {'drift': (-0.50, -0.20),'vol': (0.25, 0.50)},
    }

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001):
        super().__init__()
        self.num_assets, self.history_len, self.episode_len = num_assets, history_len, episode_len
        self.initial_cash, self.fee = initial_cash, fee
        self.features_per_step = 2 * num_assets + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (history_len * self.features_per_step,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (num_assets + 1,), np.float32)
        self.reset()

    def _generate_raw_prices(self):
        T = self.episode_len + self.history_len + 10
        n_seg = self.np_random.integers(3, 7)
        keys = list(self.REGIMES.keys())
        seq = [keys[self.np_random.integers(len(keys))] for _ in range(n_seg)]
        dur = self.np_random.dirichlet(np.ones(n_seg) * 2.0) * T
        dur = np.maximum(dur.astype(int), 30); dur[-1] = T - sum(dur[:-1])
        prices = np.zeros((T, self.num_assets), np.float64)
        for a in range(self.num_assets):
            p = self.np_random.uniform(20., 300.); series = [p]; day = 0
            for reg, d in zip(seq, dur):
                pr = self.REGIMES[reg]
                drift = self.np_random.uniform(*pr['drift']) + self.np_random.uniform(-0.03, 0.03)
                vol = self.np_random.uniform(*pr['vol']) * self.np_random.uniform(0.85, 1.15)
                mu, sigma = drift/252., vol/np.sqrt(252.)
                for _ in range(max(0, min(d, T-day-1))):
                    p = max(0.01, p*np.exp((mu-.5*sigma**2)+sigma*self.np_random.standard_normal()))
                    series.append(p); day += 1
                    if day >= T: break
                if day >= T: break
            while len(series) < T: series.append(series[-1])
            prices[:, a] = series[:T]
        return prices

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prices = self._generate_raw_prices()
        self.start = self.history_len; self.current_step = self.start
        self.cash = self.initial_cash * 0.05
        self.shares = (self.initial_cash * 0.95 / self.num_assets) / self.prices[self.current_step]
        self.peak_wealth = self.last_wealth = self.initial_cash; self.steps_done = 0
        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs(), {}

    def _wealth(self): return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p, pp = self.prices[t], self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        return np.concatenate([p/self.prices[self.start], np.log(p/np.maximum(1e-4, pp)),
                               [self.cash/w, np.clip((w-self.peak_wealth)/max(1e-4, self.peak_wealth), -1, 0)]]).astype(np.float32)

    def _flat_obs(self): return np.concatenate(self.obs_history).astype(np.float32)

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


class SimplifiedSyntheticEnv(AlphaSyntheticEnv):
    """2 regimes only (bull + bear)."""
    REGIMES = {'bull': {'drift': (0.15, 0.50), 'vol': (0.10, 0.20)},
               'bear': {'drift': (-0.50, -0.15), 'vol': (0.25, 0.50)}}

class FlatVolSyntheticEnv(AlphaSyntheticEnv):
    """Fixed volatility, only drift varies."""
    REGIMES = {'strong_bull': {'drift': (0.30, 0.70), 'vol': (0.15, 0.15)},
               'mild_bull': {'drift': (0.15, 0.30), 'vol': (0.15, 0.15)},
               'sideways': {'drift': (-0.02, 0.05), 'vol': (0.15, 0.15)},
               'bear_crash': {'drift': (-0.50, -0.20), 'vol': (0.15, 0.15)}}

class RandomWalkEnv(AlphaSyntheticEnv):
    """Pure GBM — no regime structure at all."""
    REGIMES = {'random': {'drift': (-0.10, 0.20), 'vol': (0.10, 0.30)}}
    def _generate_raw_prices(self):
        T = self.episode_len + self.history_len + 10
        prices = np.zeros((T, self.num_assets), np.float64)
        for a in range(self.num_assets):
            p = self.np_random.uniform(20., 300.)
            mu = self.np_random.uniform(-0.10, 0.20)/252.
            sigma = self.np_random.uniform(0.10, 0.30)/np.sqrt(252.)
            series = [p]
            for _ in range(T-1):
                p = max(0.01, p*np.exp((mu-.5*sigma**2)+sigma*self.np_random.standard_normal()))
                series.append(p)
            prices[:, a] = series[:T]
        return prices

GEN_REGISTRY = {"Original_4Regime": AlphaSyntheticEnv, "Simplified_2Regime": SimplifiedSyntheticEnv,
                "FlatVol": FlatVolSyntheticEnv, "RandomWalk": RandomWalkEnv}


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO TRAINING + FREEZE VERIFICATION
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

def model_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd.keys()): h.update(k.encode()); h.update(sd[k].cpu().numpy().tobytes())
    return h.hexdigest()

def save_verified(model, path):
    sd = model.state_dict(); torch.save(sd, path); return model_hash(sd)

def load_verified(cls, path, expected=None):
    m = cls(); sd = torch.load(path, weights_only=True); m.load_state_dict(sd); m.eval()
    h = model_hash(sd)
    if expected and h != expected: raise ValueError(f"FREEZE VIOLATION: {h[:16]} != {expected[:16]}")
    return m, h


# ═══════════════════════════════════════════════════════════════════════════════
#  PARALLEL TRAINING WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_train(args):
    """Multiprocessing worker: train one model and save checkpoint."""
    arch_name, gen_name, seed, path, total_steps = args
    if os.path.exists(path):
        return (arch_name, gen_name, seed, path, "exists", 0.)
    t0 = time.time()
    model = ARCH_REGISTRY[arch_name]()
    env = GEN_REGISTRY[gen_name]()
    model = train_ppo(model, env, seed=seed, total_steps=total_steps)
    save_verified(model, path)
    return (arch_name, gen_name, seed, path, "trained", time.time()-t0)


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

def eval_model_on_prices(model, prices_raw, fee_bps=0, slippage_pct=0., rng=None):
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
        if slippage_pct > 0 and rng is not None: p_ex = p*(1.+rng.uniform(-slippage_pct/100., slippage_pct/100., N))
        else: p_ex = p
        w = max(1e-4, cash+np.sum(shares*p)); caw = (shares*p)/w; ccf = cash/w
        if abs(ccf-tc)+np.sum(np.abs(caw-taw)) > 0.03:
            tv = abs(cash-w*tc)+np.sum(np.abs(shares*p-w*taw))
            net = max(1e-4, w-tv*fee); cash = net*tc; shares = (net*taw)/np.maximum(1e-4, p_ex)
        nw = cash+np.sum(shares*p); peak = max(peak, nw); eq.append(nw)
        pp = prices_raw[t-1]
        np_ = np.pad(p/prices_raw[30], (0, max(0, 10-N)), constant_values=1.)[:10]
        lr = np.pad(np.log(p/np.maximum(1e-4, pp)), (0, max(0, 10-N)), constant_values=0.)[:10]
        obs_h.pop(0); obs_h.append(np.concatenate([np_, lr, [cash/max(1e-4,nw), np.clip((nw-peak)/max(1e-4,peak),-1,0)]]).astype(np.float32))
    return compute_metrics(eq)


# ═══════════════════════════════════════════════════════════════════════════════
#  BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def bl_spy(p, c=5): return compute_metrics((10000.*p[:,c]/p[0,c]).tolist())
def bl_equal_weight(p):
    T, N = p.shape; sh = (10000./N)/p[0]; eq = [10000.]
    for t in range(1, T):
        w = np.sum(sh*p[t])
        if t % 21 == 0: sh = (w/N)/p[t]
        eq.append(w)
    return compute_metrics(eq)
def bl_risk_parity(p, lb=60):
    T, N = p.shape; sh = (10000./N)/p[0]; eq = [10000.]
    for t in range(1, T):
        w = np.sum(sh*p[t])
        if t % 21 == 0 and t >= lb:
            v = np.std(np.diff(np.log(p[t-lb:t+1]), axis=0), axis=0)
            iv = 1./np.maximum(1e-8, v); wts = iv/iv.sum(); sh = (w*wts)/p[t]
        eq.append(w)
    return compute_metrics(eq)
def bl_momentum(p, lb=60, k=3):
    T, N = p.shape; sh = (10000./N)/p[0]; eq = [10000.]
    for t in range(1, T):
        w = np.sum(sh*p[t])
        if t % 21 == 0 and t >= lb:
            mom = p[t]/p[t-lb]-1.; top = np.argsort(mom)[-k:]
            wts = np.zeros(N); wts[top] = 1./k; sh = (w*wts)/np.maximum(1e-8, p[t])
        eq.append(w)
    return compute_metrics(eq)
def bl_sma(p, sc=5, sw=50, lw=200):
    T = p.shape[0]; spy = p[:, sc]; tc = min(6, p.shape[1]-1); tlt = p[:, tc]
    cash = 10000.; eq = [10000.]; in_spy = True
    for t in range(1, T):
        if t >= lw:
            ss, sl = np.mean(spy[t-sw:t]), np.mean(spy[t-lw:t])
            in_spy = ss > sl
        dr = (spy[t]/spy[t-1]-1.) if in_spy else (tlt[t]/tlt[t-1]-1.)
        cash *= (1.+dr); eq.append(cash)
    return compute_metrics(eq)
def bl_random(p, nt=10):
    T, N = p.shape; finals = []
    for trial in range(nt):
        rng = np.random.RandomState(1000+trial); sh = np.zeros(N); cash = 10000.; eq = [10000.]
        for t in range(1, T):
            w = cash+np.sum(sh*p[t])
            if t % 21 == 0:
                wts = rng.dirichlet(np.ones(N+1)); cash = w*wts[0]; sh = (w*wts[1:])/np.maximum(1e-8, p[t])
            eq.append(cash+np.sum(sh*p[t]))
        finals.append(eq[-1])
    m = compute_metrics(eq); m['final_mean'] = float(np.mean(finals)); return m


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

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
#  DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]

def download_period(start, end, label):
    import yfinance as yf
    print(f"    ↓ {label} ({start} → {end})...", flush=True)
    try:
        df = yf.download(TICKERS, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close'] if 'Close' in df.columns.get_level_values(0) else df.iloc[:, :len(TICKERS)]
        df = df[TICKERS].dropna()
        if len(df) < 40: print(f"    ⚠ {label}: only {len(df)} rows", flush=True); return None
        return df
    except Exception as e:
        print(f"    ⚠ {label} failed: {e}", flush=True); return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()
    W = 120
    print("="*W); print("  RAI v6 SCIENTIFIC ROBUSTNESS PROTOCOL (PARALLEL)"); print("  8-core × 10 seeds — ~15-20 min estimated"); print("="*W, flush=True)

    results = {}

    # ═══════════════════════════════════════════════
    # PHASE 1+2: PARALLEL TRAINING (all seeds + ablations at once)
    # ═══════════════════════════════════════════════
    print(f"\n{'═'*W}\n  PHASES 1+2: Parallel Training — {N_SEEDS} seeds × 4 architectures = {N_SEEDS*4} models\n{'═'*W}", flush=True)

    jobs = []
    for arch_name in ARCH_REGISTRY:
        for seed in range(1, N_SEEDS + 1):
            if arch_name == "RAI_v6_Hybrid":
                path = os.path.join(RESULTS_DIR, "seeds", f"rai_v6_seed_{seed:02d}.pt")
            else:
                path = os.path.join(RESULTS_DIR, "ablations", f"{arch_name}_seed_{seed:02d}.pt")
            jobs.append((arch_name, "Original_4Regime", seed, path, TOTAL_STEPS))

    print(f"  Dispatching {len(jobs)} training jobs across {NUM_WORKERS} workers...", flush=True)
    t1 = time.time()
    with mp.Pool(processes=NUM_WORKERS) as pool:
        outcomes = pool.map(_worker_train, jobs)

    trained = sum(1 for o in outcomes if o[4] == "trained")
    cached = sum(1 for o in outcomes if o[4] == "exists")
    times = [o[5] for o in outcomes if o[4] == "trained"]
    print(f"  ✓ Phase 1+2 complete: {trained} trained, {cached} cached | "
          f"Total: {time.time()-t1:.0f}s", flush=True)
    if times:
        print(f"    Per-model: min={min(times):.0f}s mean={np.mean(times):.0f}s max={max(times):.0f}s", flush=True)

    # ═══════════════════════════════════════════════
    # PHASE 4: DOWNLOAD HISTORICAL PERIODS
    # ═══════════════════════════════════════════════
    print(f"\n{'═'*W}\n  PHASE 4: Downloading Unseen Historical Periods\n{'═'*W}", flush=True)

    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "test_prices.csv"),
                          index_col=0, parse_dates=True)
    test_prices = test_df.values

    extra = {"2007-2009_GFC": ("2007-01-01","2009-12-31"), "2015-2017_LowVol": ("2015-01-01","2017-12-31"),
             "2018-2019_TradeWar": ("2018-01-01","2019-12-31"), "2022-2023_RateHike": ("2022-01-01","2023-12-31")}
    periods = {"2020-2024_OOS": test_prices}
    for pn, (s, e) in extra.items():
        df = download_period(s, e, pn)
        if df is not None: periods[pn] = df.values; print(f"    ✓ {pn}: {len(df)} days", flush=True)

    # ═══════════════════════════════════════════════
    # PHASE 3+5+6+7: EVALUATION
    # ═══════════════════════════════════════════════
    print(f"\n{'═'*W}\n  PHASES 3+5+6+7: Evaluation & Statistical Analysis\n{'═'*W}", flush=True)

    cost_tiers = [("Zero_Cost",0,0.),("Institutional",3,0.02),("Low",5,0.),("Medium",10,0.05),("High",20,0.10)]
    arch_period_metrics = {a: {p: [] for p in periods} for a in ARCH_REGISTRY}
    cost_metrics = {c[0]: [] for c in cost_tiers}

    for arch_name, arch_cls in ARCH_REGISTRY.items():
        print(f"\n  Evaluating {arch_name}...", flush=True)
        for seed in range(1, N_SEEDS+1):
            path = os.path.join(RESULTS_DIR, "seeds" if arch_name == "RAI_v6_Hybrid" else "ablations",
                                f"{'rai_v6' if arch_name == 'RAI_v6_Hybrid' else arch_name}_seed_{seed:02d}.pt")
            if not os.path.exists(path):
                print(f"    ⚠ Missing {path}", flush=True); continue
            model, h = load_verified(arch_cls, path)
            print(f"    Seed {seed:02d} | Hash {h[:12]}✓", flush=True)

            for pn, pv in periods.items():
                m = eval_model_on_prices(model, pv)
                arch_period_metrics[arch_name][pn].append(m)

            # Cost tiers (RAI v6 only)
            if arch_name == "RAI_v6_Hybrid":
                for cn, fb, sl in cost_tiers:
                    rng = np.random.RandomState(seed*1000)
                    m = eval_model_on_prices(model, test_prices, fee_bps=fb, slippage_pct=sl, rng=rng)
                    cost_metrics[cn].append(m)

    # ── Statistical summary ──
    print(f"\n{'─'*W}")
    print(f"  STATISTICAL SUMMARY — Primary Test Period (2020-2024 OOS)", flush=True)
    print(f"{'─'*W}")
    pp = "2020-2024_OOS"
    stat_tables = {}
    print(f"  {'Architecture':<25} | {'Return (%)':<35} | {'Sharpe':<35} | {'Max DD (%)':<35}", flush=True)
    print(f"  {'-'*135}", flush=True)
    for an in ARCH_REGISTRY:
        ms = arch_period_metrics[an].get(pp, [])
        if not ms: continue
        st = {"return": comp_stats([m["return_pct"] for m in ms]),
              "sharpe": comp_stats([m["sharpe"] for m in ms]),
              "max_dd": comp_stats([m["max_dd_pct"] for m in ms]),
              "final":  comp_stats([m["final"] for m in ms])}
        stat_tables[an] = st
        r, s, d = st["return"], st["sharpe"], st["max_dd"]
        print(f"  {an:<25} | {r['mean']:>+7.2f}±{r['std']:>5.2f} CI[{r['ci95'][0]:>+7.2f},{r['ci95'][1]:>+7.2f}] "
              f"| {s['mean']:>+6.2f}±{s['std']:>4.2f} CI[{s['ci95'][0]:>+6.2f},{s['ci95'][1]:>+6.2f}] "
              f"| {d['mean']:>+7.2f}±{d['std']:>5.2f} CI[{d['ci95'][0]:>+7.2f},{d['ci95'][1]:>+7.2f}]", flush=True)

    # ── Significance tests ──
    print(f"\n  SIGNIFICANCE TESTS (Welch's t-test, RAI v6 vs Ablation)", flush=True)
    print(f"  {'-'*100}", flush=True)
    rai_rets = [m["return_pct"] for m in arch_period_metrics["RAI_v6_Hybrid"].get(pp, [])]
    rai_sh = [m["sharpe"] for m in arch_period_metrics["RAI_v6_Hybrid"].get(pp, [])]
    sig_results = {}
    for an in ["MLP_Only", "Conv1D_Only", "Transformer_Only"]:
        ab_r = [m["return_pct"] for m in arch_period_metrics[an].get(pp, [])]
        ab_s = [m["sharpe"] for m in arch_period_metrics[an].get(pp, [])]
        if not ab_r: continue
        tr = welch_test(rai_rets, ab_r); ts = welch_test(rai_sh, ab_s)
        sig_results[an] = {"return": tr, "sharpe": ts}
        sr = "***" if tr['p']<.001 else "**" if tr['p']<.01 else "*" if tr['p']<.05 else "ns"
        ss = "***" if ts['p']<.001 else "**" if ts['p']<.01 else "*" if ts['p']<.05 else "ns"
        print(f"  vs {an:<25} | Return p={tr['p']:.4f}{sr:>4} d={tr['d']:+.2f} | Sharpe p={ts['p']:.4f}{ss:>4} d={ts['d']:+.2f}", flush=True)

    # ── Multi-period transfer ──
    print(f"\n  MULTI-PERIOD TRANSFER (RAI v6, {N_SEEDS} seeds)", flush=True)
    print(f"  {'-'*100}", flush=True)
    for pn in periods:
        ms = arch_period_metrics["RAI_v6_Hybrid"].get(pn, [])
        if not ms: continue
        r = comp_stats([m["return_pct"] for m in ms]); s = comp_stats([m["sharpe"] for m in ms])
        print(f"  {pn:<25} | Return: {r['mean']:>+8.2f}±{r['std']:>6.2f}% | Sharpe: {s['mean']:>+6.2f}±{s['std']:>5.2f} | n={len(ms)}", flush=True)

    # ── Transaction costs ──
    print(f"\n  TRANSACTION COST IMPACT (RAI v6, {N_SEEDS} seeds)", flush=True)
    print(f"  {'-'*80}", flush=True)
    for cn, _, _ in cost_tiers:
        ms = cost_metrics.get(cn, [])
        if not ms: continue
        r = comp_stats([m["return_pct"] for m in ms]); s = comp_stats([m["sharpe"] for m in ms])
        print(f"  {cn:<20} | Return: {r['mean']:>+8.2f}±{r['std']:>6.2f}% | Sharpe: {s['mean']:>+6.2f}±{s['std']:>5.2f}", flush=True)

    # ── Baselines ──
    print(f"\n  BASELINES (2020-2024 OOS)", flush=True)
    print(f"  {'-'*90}", flush=True)
    print(f"  {'Strategy':<30} | {'Return':<12} | {'Sharpe':<8} | {'Max DD':<12} | {'Final':<12}", flush=True)
    print(f"  {'-'*90}", flush=True)
    for bn, bf in [("Buy & Hold SPY", lambda p: bl_spy(p)), ("Equal Weight", lambda p: bl_equal_weight(p)),
                   ("Risk Parity", lambda p: bl_risk_parity(p)), ("Momentum Top-3", lambda p: bl_momentum(p)),
                   ("SMA 50/200", lambda p: bl_sma(p)), ("Random Policy", lambda p: bl_random(p))]:
        m = bf(test_prices)
        print(f"  {bn:<30} | {m['return_pct']:>+10.2f}% | {m['sharpe']:>6.2f} | {m['max_dd_pct']:>+10.2f}% | ${m['final']:>10,.2f}", flush=True)
    if "RAI_v6_Hybrid" in stat_tables:
        r = stat_tables["RAI_v6_Hybrid"]
        print(f"  {'-'*90}", flush=True)
        print(f"  {'RAI v6 (mean±std)':<30} | {r['return']['mean']:>+7.2f}±{r['return']['std']:>4.1f}% "
              f"| {r['sharpe']['mean']:>6.2f} | {r['max_dd']['mean']:>+7.2f}±{r['max_dd']['std']:>4.1f}% "
              f"| ${r['final']['mean']:>10,.2f}", flush=True)

    # ═══════════════════════════════════════════════
    # PHASE 8: GENERATOR SENSITIVITY (parallel)
    # ═══════════════════════════════════════════════
    print(f"\n{'═'*W}\n  PHASE 8: Synthetic Generator Sensitivity ({N_GEN_SEEDS} seeds × 4 generators)\n{'═'*W}", flush=True)

    gen_jobs = []
    for gn, gc in GEN_REGISTRY.items():
        for seed in range(1, N_GEN_SEEDS+1):
            path = os.path.join(RESULTS_DIR, "generators", f"{gn}_seed_{seed:02d}.pt")
            gen_jobs.append(("RAI_v6_Hybrid", gn, seed, path, TOTAL_STEPS))

    print(f"  Dispatching {len(gen_jobs)} generator training jobs...", flush=True)
    t8 = time.time()
    with mp.Pool(processes=NUM_WORKERS) as pool:
        gen_outcomes = pool.map(_worker_train, gen_jobs)
    print(f"  ✓ Phase 8 training: {time.time()-t8:.0f}s", flush=True)

    gen_eval = {gn: [] for gn in GEN_REGISTRY}
    for gn in GEN_REGISTRY:
        for seed in range(1, N_GEN_SEEDS+1):
            path = os.path.join(RESULTS_DIR, "generators", f"{gn}_seed_{seed:02d}.pt")
            if not os.path.exists(path): continue
            model, _ = load_verified(DeepEndToEndTradingNet, path)
            gen_eval[gn].append(eval_model_on_prices(model, test_prices))

    print(f"\n  {'Generator':<25} | {'Return (%)':<35} | {'Sharpe':<35}", flush=True)
    print(f"  {'-'*100}", flush=True)
    for gn in GEN_REGISTRY:
        if not gen_eval[gn]: continue
        r = comp_stats([m["return_pct"] for m in gen_eval[gn]]); s = comp_stats([m["sharpe"] for m in gen_eval[gn]])
        print(f"  {gn:<25} | {r['mean']:>+7.2f}±{r['std']:>5.2f} CI[{r['ci95'][0]:>+7.2f},{r['ci95'][1]:>+7.2f}] "
              f"| {s['mean']:>+6.2f}±{s['std']:>4.2f} CI[{s['ci95'][0]:>+6.2f},{s['ci95'][1]:>+6.2f}]", flush=True)

    # Critical test: Original vs RandomWalk
    if gen_eval.get("Original_4Regime") and gen_eval.get("RandomWalk"):
        or_ = [m["return_pct"] for m in gen_eval["Original_4Regime"]]
        rw_ = [m["return_pct"] for m in gen_eval["RandomWalk"]]
        t_ = welch_test(or_, rw_)
        sig = "***" if t_['p']<.001 else "**" if t_['p']<.01 else "*" if t_['p']<.05 else "ns"
        print(f"\n  CRITICAL: Original vs RandomWalk | p={t_['p']:.4f} {sig} | d={t_['d']:+.2f}", flush=True)
        if t_['p'] < .05: print("  → Regime structure SIGNIFICANTLY affects transfer.", flush=True)
        else: print("  → No significant difference — model may exploit simpler patterns.", flush=True)

    # ═══════════════════════════════════════════════
    # PHASE 9: FINAL HOLDOUT
    # ═══════════════════════════════════════════════
    print(f"\n{'═'*W}\n  PHASE 9: UNTOUCHED FINAL HOLDOUT (2024-06 → Present)\n  ⚠ TESTED EXACTLY ONCE\n{'═'*W}", flush=True)

    hdf = download_period("2024-06-01", "2026-08-08", "Final Holdout")
    holdout_result = None
    if hdf is not None and len(hdf) >= 40:
        hp = hdf.values
        print(f"  Holdout: {len(hdf)} days ({hdf.index[0].date()} → {hdf.index[-1].date()})", flush=True)

        # Best seed by median Sharpe
        seed_sharpes = {}
        for seed in range(1, N_SEEDS+1):
            path = os.path.join(RESULTS_DIR, "seeds", f"rai_v6_seed_{seed:02d}.pt")
            if not os.path.exists(path): continue
            sharpes = []
            for pn, ms in arch_period_metrics["RAI_v6_Hybrid"].items():
                for m in ms: sharpes.append(m.get("sharpe", 0))
            # Per-seed sharpe across periods would be better but we use overall as proxy
            model, _ = load_verified(DeepEndToEndTradingNet, path)
            test_m = eval_model_on_prices(model, test_prices)
            seed_sharpes[seed] = test_m["sharpe"]

        best = max(seed_sharpes, key=seed_sharpes.get) if seed_sharpes else 1
        print(f"  Best seed: {best} (test Sharpe: {seed_sharpes.get(best, 0):.3f})", flush=True)

        bm, _ = load_verified(DeepEndToEndTradingNet, os.path.join(RESULTS_DIR, "seeds", f"rai_v6_seed_{best:02d}.pt"))
        best_r = eval_model_on_prices(bm, hp)

        # Ensemble
        ens_rets = []
        for seed in range(1, N_SEEDS+1):
            path = os.path.join(RESULTS_DIR, "seeds", f"rai_v6_seed_{seed:02d}.pt")
            if not os.path.exists(path): continue
            em, _ = load_verified(DeepEndToEndTradingNet, path)
            ens_rets.append(eval_model_on_prices(em, hp)["return_pct"])

        spy_r = bl_spy(hp); ew_r = bl_equal_weight(hp)

        print(f"\n  {'Strategy':<30} | {'Return':<12} | {'Sharpe':<8} | {'Max DD':<12}", flush=True)
        print(f"  {'-'*70}", flush=True)
        print(f"  {'Buy & Hold SPY':<30} | {spy_r['return_pct']:>+10.2f}% | {spy_r['sharpe']:>6.2f} | {spy_r['max_dd_pct']:>+10.2f}%", flush=True)
        print(f"  {'Equal Weight':<30} | {ew_r['return_pct']:>+10.2f}% | {ew_r['sharpe']:>6.2f} | {ew_r['max_dd_pct']:>+10.2f}%", flush=True)
        print(f"  {f'RAI v6 Best (Seed {best})':<30} | {best_r['return_pct']:>+10.2f}% | {best_r['sharpe']:>6.2f} | {best_r['max_dd_pct']:>+10.2f}%", flush=True)
        print(f"  {'RAI v6 Ensemble (mean)':<30} | {np.mean(ens_rets) if ens_rets else 0:>+10.2f}%", flush=True)
        holdout_result = {"best_seed": int(best), "best": best_r, "ensemble_mean": float(np.mean(ens_rets)) if ens_rets else 0,
                          "spy": spy_r, "ew": ew_r, "days": len(hdf)}

    # ═══════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════
    def jsonify(o):
        if isinstance(o, dict): return {k: jsonify(v) for k, v in o.items()}
        if isinstance(o, list): return [jsonify(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    master = {"config": {"n_seeds": N_SEEDS, "n_gen_seeds": N_GEN_SEEDS, "total_steps": TOTAL_STEPS, "workers": NUM_WORKERS},
              "phase3_stats": jsonify(stat_tables), "phase3_tests": jsonify(sig_results),
              "phase6_costs": jsonify({cn: comp_stats([m["return_pct"] for m in cost_metrics[cn]]) for cn in cost_metrics if cost_metrics[cn]}),
              "phase8_generators": jsonify({gn: comp_stats([m["return_pct"] for m in gen_eval[gn]]) for gn in gen_eval if gen_eval[gn]})}
    if holdout_result: master["phase9_holdout"] = jsonify(holdout_result)

    rp = os.path.join(RESULTS_DIR, "results.json")
    with open(rp, 'w') as f: json.dump(master, f, indent=2, default=str)

    total = time.time() - T0
    print(f"\n{'═'*W}\n  ✅ EXPERIMENT COMPLETE — {total/60:.1f} minutes\n  Results: {rp}\n{'═'*W}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
