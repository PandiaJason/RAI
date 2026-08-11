"""
RAI v4.1: Train + Eval + Truth Test — All in One.
Sigmoid cash, episode death, 500k steps.
"""
import os, sys, time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rai.world.real_ai_env import RealAIEnv


class Monitor(BaseCallback):
    def __init__(self, freq=10000):
        super().__init__(1)
        self.freq = freq; self.acts = []; self.deaths = 0
    def _on_step(self):
        if self.locals.get('actions') is not None:
            self.acts.append(self.locals['actions'].copy())
        if self.locals.get('dones') is not None and np.any(self.locals['dones']):
            infos = self.locals.get('infos', [])
            for info in infos:
                if info.get('drawdown', 0) < -0.24:
                    self.deaths += 1
        if self.n_calls % self.freq == 0 and len(self.acts) > 50:
            a = np.concatenate(self.acts[-300:], axis=0)
            cash_logit = a[:, 0]
            cash_frac = 1.0 / (1.0 + np.exp(-np.clip(cash_logit, -10, 10)))
            std = np.mean(np.std(a, axis=0))
            print(f"  {self.n_calls:>7d} | action_std={std:.4f} | "
                  f"cash_frac: min={np.min(cash_frac):.3f} mean={np.mean(cash_frac):.3f} "
                  f"max={np.max(cash_frac):.3f} | deaths={self.deaths}", flush=True)
        return True


# ═══════════════════════════════════════════════
#  REAL MARKET EVAL ENV (matches v4.1 obs/action)
# ═══════════════════════════════════════════════

class RealMarketV41Env(gym.Env):
    def __init__(self, price_df, initial_cash=10000.0, history_len=16, max_assets=10, fee=0.001):
        super().__init__()
        self.prices = price_df.values[:, :max_assets].copy()
        self.T, self.N = self.prices.shape
        self.initial_cash = initial_cash
        self.history_len = history_len
        self.fee = fee
        self.single_obs_dim = 4 + 2 * self.N
        self.obs_dim = history_len * self.single_obs_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-5.0, 5.0, (self.N + 1,), np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = self.history_len + 20
        self.cash = self.initial_cash * 0.5
        p = self.prices[self.step_idx]
        self.shares = (self.initial_cash * 0.5 / self.N) / p
        self.peak = self.initial_cash
        self.obs_hist = [self._obs() for _ in range(self.history_len)]
        self.log_cash_frac = []; self.log_wealth = []; self.log_rebal = []
        self.log_actions = []
        return self._flat_obs(), {}

    def _w(self):
        return self.cash + np.sum(self.shares * self.prices[self.step_idx])

    def _obs(self):
        p = self.prices[self.step_idx]; t = self.step_idx
        w = max(1e-4, self._w())
        cw = self.cash / w
        dd = np.clip((w - self.peak) / max(1e-4, self.peak), -1, 0)
        r5 = np.mean((p - self.prices[max(0,t-5)]) / np.maximum(1e-4, self.prices[max(0,t-5)])) if t >= 5 else 0
        if t >= 10:
            sub = self.prices[t-10:t+1]
            r = (sub[1:]-sub[:-1])/np.maximum(1e-4,sub[:-1])
            vol = np.mean(np.std(r, axis=0))
        else:
            vol = 0
        aw = (self.shares * p) / w
        if t >= 50:
            s20 = np.mean(self.prices[t-20:t], axis=0)
            s50 = np.mean(self.prices[t-50:t], axis=0)
            trend = s20 / np.maximum(1e-4, s50) - 1.0
        else:
            trend = np.zeros(self.N)
        return np.concatenate([[cw, dd, r5, vol], aw, trend]).astype(np.float32)

    def _flat_obs(self):
        return np.concatenate(self.obs_hist).astype(np.float32)

    def step(self, action):
        self.log_actions.append(action.copy())
        cl = np.clip(action[0], -10, 10)
        target_cash = 1.0 / (1.0 + np.exp(-cl))
        target_stock = 1.0 - target_cash
        al = action[1:]
        ea = np.exp(al - np.max(al)); rw = ea / np.sum(ea)
        target_aw = rw * target_stock

        p = self.prices[self.step_idx]
        w = max(1e-4, self._w())
        caw = (self.shares * p) / w; ccf = self.cash / w
        drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))

        did_rebal = False
        if drift > 0.03:
            did_rebal = True
            tv = abs(self.cash - w*target_cash) + np.sum(np.abs(self.shares*p - w*target_aw))
            net = max(1e-4, w - tv * self.fee)
            self.cash = net * target_cash
            self.shares = (net * target_aw) / np.maximum(1e-4, p)

        self.log_cash_frac.append(target_cash)
        self.log_rebal.append(did_rebal)

        self.step_idx += 1
        done = self.step_idx >= self.T - 1
        nw = self._w()
        self.peak = max(self.peak, nw)
        self.log_wealth.append(nw)
        self.obs_hist.pop(0); self.obs_hist.append(self._obs())
        return self._flat_obs(), 0.0, done, False, {"portfolio_value": nw}


def metrics(eq):
    eq = np.array(eq, dtype=np.float64)
    if len(eq) < 2: return {}
    r = (eq[1:]-eq[:-1])/np.maximum(1e-8,eq[:-1])
    ret = (eq[-1]/eq[0]-1)*100
    vol = np.std(r)*np.sqrt(252)*100
    sh = np.mean(r)/np.std(r)*np.sqrt(252) if np.std(r)>1e-8 else 0
    pk = np.maximum.accumulate(eq)
    mdd = np.min((eq-pk)/pk)*100
    return {"return": ret, "vol": vol, "sharpe": sh, "max_dd": mdd, "final": eq[-1]}


def main():
    print("="*70, flush=True)
    print("  RAI v4.1: REAL AI — Sigmoid Cash + Episode Death", flush=True)
    print("="*70, flush=True)

    env_kw = dict(num_assets=10, episode_len=504, history_len=16,
                  initial_cash=10000.0, transaction_fee=0.001,
                  max_drawdown=-0.25, death_penalty=-10.0)

    vec = DummyVecEnv([lambda: RealAIEnv(**env_kw)])

    model = PPO("MlpPolicy", vec, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
                verbose=0, device="cpu", seed=42)

    tp = sum(p.numel() for p in model.policy.parameters())
    print(f"  Params: {tp:,}", flush=True)
    print(f"  Training 500k steps...\n", flush=True)

    t0 = time.time()
    model.learn(total_timesteps=500_000, callback=Monitor(freq=25000))
    el = time.time() - t0
    print(f"\n  Trained in {el:.0f}s ({500000/el:.0f} FPS)", flush=True)

    os.makedirs("./data/v0.4_rl_checkpoints/", exist_ok=True)
    model.save("./data/v0.4_rl_checkpoints/rai_v41_real_ai")
    print(f"  Saved.", flush=True)

    # ═══ EVALUATE ON REAL MARKET ═══
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)

    for label, df in [("2020-2024", test_df), ("2010-2019", train_df)]:
        print(f"\n{'='*70}", flush=True)
        print(f"  {label}", flush=True)
        print(f"{'='*70}", flush=True)

        env = RealMarketV41Env(price_df=df, max_assets=10)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)

        eq = [10000.0] + env.log_wealth
        m = metrics(eq)
        cf = np.array(env.log_cash_frac)
        ra = np.array(env.log_actions)
        rb = np.array(env.log_rebal)
        astd = np.mean(np.std(ra, axis=0))

        print(f"  Return: {m['return']:+.2f}%  Sharpe: {m['sharpe']:.2f}  MaxDD: {m['max_dd']:.2f}%", flush=True)
        print(f"  Cash fraction: min={np.min(cf):.4f}  mean={np.mean(cf):.4f}  max={np.max(cf):.4f}  std={np.std(cf):.4f}", flush=True)
        print(f"  Cash range: {(np.max(cf)-np.min(cf))*100:.1f}%", flush=True)
        print(f"  Action std: {astd:.6f}", flush=True)
        print(f"  Rebalances: {np.sum(rb)}/{len(rb)} ({np.sum(rb)/len(rb)*100:.1f}%)", flush=True)

        # COVID check (only for 2020-2024)
        if '2020' in label:
            dates = df.index
            offset = env.history_len + 20
            pre = [i-offset for i,d in enumerate(dates) if str(d)[:7]=='2020-01' and i>=offset and i-offset<len(cf)]
            crash = [i-offset for i,d in enumerate(dates) if str(d)[:7] in ['2020-02','2020-03'] and i>=offset and i-offset<len(cf)]
            post = [i-offset for i,d in enumerate(dates) if str(d)[:7] in ['2020-05','2020-06'] and i>=offset and i-offset<len(cf)]
            if pre and crash:
                print(f"\n  COVID CRASH TEST:", flush=True)
                print(f"    Before (Jan):     {np.mean([cf[i] for i in pre]):.4f} cash", flush=True)
                print(f"    During (Feb-Mar): {np.mean([cf[i] for i in crash]):.4f} cash", flush=True)
                if post:
                    print(f"    After (May-Jun):  {np.mean([cf[i] for i in post]):.4f} cash", flush=True)

        # SPY comparison
        spy = df['SPY'].values
        m_spy = metrics(10000*(spy/spy[0]))
        print(f"\n  vs SPY: {m_spy['return']:+.2f}% return, {m_spy['sharpe']:.2f} sharpe, {m_spy['max_dd']:.2f}% maxDD", flush=True)

    # ═══ TRUTH TEST VERDICT ═══
    print(f"\n{'='*70}", flush=True)
    print(f"  TRUTH TEST: Is v4.1 REAL AI?", flush=True)
    print(f"{'='*70}", flush=True)
    cf_test = np.array(env.log_cash_frac)  # From last eval (2010-2019)
    ra_test = np.array(env.log_actions)
    cash_range = np.max(cf_test) - np.min(cf_test)
    a_std = np.mean(np.std(ra_test, axis=0))

    checks = {
        "Actions non-constant (std > 0.01)": a_std > 0.01,
        "Cash swings > 5%": cash_range > 0.05,
        "Cash swings > 10%": cash_range > 0.10,
        "Cash swings > 20%": cash_range > 0.20,
    }
    for name, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {name} (actual: {cash_range*100:.1f}% range, std={a_std:.4f})", flush=True)

    print(f"\n  ✅ Done!", flush=True)


if __name__ == "__main__":
    main()
