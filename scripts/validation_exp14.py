"""
Validation Experiments 1 & 4: Training Required
=================================================
Exp 1: Multi-Seed Reproducibility (5 seeds × 150k steps)
Exp 4: Ablation Study (4 variants × 150k steps)
"""
import os, sys, time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


# ═══════════════════════════════════════════════
#  V3 Environment (imported inline)
# ═══════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rai.world.synthetic_v3_env import SyntheticMultiRegimeEnv


# ═══════════════════════════════════════════════
#  Ablation Variant: No Multi-Regime (Random GBM only)
# ═══════════════════════════════════════════════

class AblationRandomOnlyEnv(SyntheticMultiRegimeEnv):
    """Ablation: Remove multi-regime, use only random GBM (like v2 but continuous)."""
    def _generate_multi_regime_prices(self):
        T = self.episode_len + self.history_len + 210
        prices = np.zeros((T, self.num_assets), dtype=np.float64)
        for i in range(self.num_assets):
            mu_annual = np.random.uniform(-0.15, 0.40)
            sigma_annual = np.random.uniform(0.10, 0.60)
            initial_price = np.random.uniform(20.0, 300.0)
            mu_daily = mu_annual / 252.0
            sigma_daily = sigma_annual / np.sqrt(252.0)
            log_returns = (mu_daily - 0.5 * sigma_daily**2) + sigma_daily * np.random.randn(T - 1)
            log_prices = np.log(initial_price) + np.concatenate([[0.0], np.cumsum(log_returns)])
            prices[:, i] = np.exp(log_prices)
        # Same indicator precomputation
        sma50 = np.zeros_like(prices)
        sma200 = np.zeros_like(prices)
        vol20 = np.zeros_like(prices)
        for t in range(200, T):
            sma50[t] = np.mean(prices[t-50:t], axis=0)
            sma200[t] = np.mean(prices[t-200:t], axis=0)
            r = (prices[t-20:t] - prices[t-21:t-1]) / np.maximum(1e-4, prices[t-21:t-1])
            vol20[t] = np.std(r, axis=0)
        return prices, sma50, sma200, vol20


# ═══════════════════════════════════════════════
#  Ablation Variant: No Trend Features
# ═══════════════════════════════════════════════

class AblationNoTrendEnv(SyntheticMultiRegimeEnv):
    """Ablation: Remove trend features (SMA50/SMA200 ratios and Vol20) from observation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override observation dimension: only cash_weight + asset_weights + prices = 1 + 20 + 20 = 41
        self.single_obs_dim = 1 + self.num_assets + self.num_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.history_len * self.single_obs_dim,), dtype=np.float32
        )
    
    def _get_single_obs(self):
        prices = self.prices_matrix[self.current_step]
        wealth = max(1e-4, self._get_portfolio_value())
        w_cash = np.array([self.cash / wealth], dtype=np.float32)
        w_assets = ((self.shares * prices) / wealth).astype(np.float32)
        norm_prices = (prices / 100.0).astype(np.float32)
        # NO trend_ratio, NO vol20
        return np.concatenate([w_cash, w_assets, norm_prices]).astype(np.float32)


# ═══════════════════════════════════════════════
#  Ablation Variant: No Rebalance Threshold
# ═══════════════════════════════════════════════

class AblationNoThresholdEnv(SyntheticMultiRegimeEnv):
    """Ablation: Remove rebalance threshold (rebalance every step)."""
    def __init__(self, **kwargs):
        kwargs['rebalance_threshold'] = 0.0
        super().__init__(**kwargs)


# ═══════════════════════════════════════════════
#  Evaluation Environment for Real Market
# ═══════════════════════════════════════════════

class RealMarketEvalEnv(gym.Env):
    def __init__(self, price_df, initial_cash=10000.0, history_len=32,
                 max_resources=20, transaction_fee=0.001, rebalance_threshold=0.03,
                 include_trend=True):
        super().__init__()
        self.price_df = price_df.copy()
        self.prices_matrix = self.price_df.values
        self.num_steps, self.num_resources = self.prices_matrix.shape
        self.max_resources = max_resources
        self.initial_cash = initial_cash
        self.history_len = history_len
        self.transaction_fee = transaction_fee
        self.rebalance_threshold = rebalance_threshold
        self.include_trend = include_trend
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
        if include_trend:
            self.single_obs_dim = 1 + max_resources * 4
        else:
            self.single_obs_dim = 1 + max_resources * 2
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
        if self.include_trend:
            tr = np.ones(self.max_resources, dtype=np.float32)
            tr[:self.num_resources] = self.sma50[self.current_step] / np.maximum(1e-4, self.sma200[self.current_step])
            v = np.zeros(self.max_resources, dtype=np.float32)
            v[:self.num_resources] = self.vol20[self.current_step]
            return np.concatenate([w_cash, w_a, np_, tr, v]).astype(np.float32)
        else:
            return np.concatenate([w_cash, w_a, np_]).astype(np.float32)
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

def eval_model_on_real(model, df, include_trend=True, rebalance_threshold=0.03):
    env = RealMarketEvalEnv(price_df=df, include_trend=include_trend, rebalance_threshold=rebalance_threshold)
    obs, _ = env.reset()
    eq = [10000.0]; done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        eq.append(info["portfolio_value"])
    return eq

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


def train_model(env_class, seed, total_steps=150_000, save_path=None, **env_kwargs):
    from stable_baselines3.common.vec_env import DummyVecEnv
    def make_env():
        env = env_class(**env_kwargs)
        return env
    vec_env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, device="cpu", seed=seed)
    model.learn(total_timesteps=total_steps)
    if save_path:
        model.save(save_path)
    return model


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Multi-Seed Reproducibility
# ═══════════════════════════════════════════════════════════════════

def run_experiment_1(test_df, train_df):
    print("=" * 90)
    print("  EXPERIMENT 1: Multi-Seed Reproducibility (5 seeds × 150k steps)")
    print("=" * 90)
    
    os.makedirs("./data/v0.3_rl_checkpoints/seeds/", exist_ok=True)
    seeds = [0, 1, 2, 3, 4]
    results = []
    
    for seed in seeds:
        t0 = time.time()
        print(f"\n  Training seed {seed}...", end=" ", flush=True)
        
        save_path = f"./data/v0.3_rl_checkpoints/seeds/rai_v3_seed{seed}"
        model = train_model(
            SyntheticMultiRegimeEnv, seed=seed, total_steps=150_000,
            save_path=save_path,
            num_assets=20, episode_len=504, history_len=32,
            initial_cash=10000.0, transaction_fee=0.001, rebalance_threshold=0.03
        )
        elapsed = time.time() - t0
        print(f"done in {elapsed:.0f}s", flush=True)
        
        # Eval on 2020-2024
        eq_test = eval_model_on_real(model, test_df)
        m_test = compute_metrics(eq_test)
        
        # Eval on 2010-2019
        eq_train = eval_model_on_real(model, train_df)
        m_train = compute_metrics(eq_train)
        
        results.append({
            "seed": seed,
            "test_return": m_test["return_pct"],
            "test_sharpe": m_test["sharpe"],
            "test_max_dd": m_test["max_dd_pct"],
            "hist_return": m_train["return_pct"],
            "hist_sharpe": m_train["sharpe"],
        })
    
    print(f"\n\n  {'Seed':<6} | {'2020-2024 Return':>16} | {'Sharpe':>8} | {'Max DD':>8} | {'2010-2019 Return':>16} | {'Sharpe':>8}")
    print(f"  {'-'*75}")
    for r in results:
        print(f"  {r['seed']:<6} | {r['test_return']:>+15.2f}% | {r['test_sharpe']:>7.2f} | {r['test_max_dd']:>7.2f}% | {r['hist_return']:>+15.2f}% | {r['hist_sharpe']:>7.2f}")
    
    test_rets = [r["test_return"] for r in results]
    test_sharpes = [r["test_sharpe"] for r in results]
    hist_rets = [r["hist_return"] for r in results]
    
    print(f"  {'-'*75}")
    print(f"  {'MEAN':<6} | {np.mean(test_rets):>+15.2f}% | {np.mean(test_sharpes):>7.2f} |         | {np.mean(hist_rets):>+15.2f}% |")
    print(f"  {'STD':<6} | {np.std(test_rets):>15.2f}% | {np.std(test_sharpes):>7.2f} |         | {np.std(hist_rets):>15.2f}% |")
    print(f"  {'MIN':<6} | {np.min(test_rets):>+15.2f}% |         |         | {np.min(hist_rets):>+15.2f}% |")
    print(f"  {'MAX':<6} | {np.max(test_rets):>+15.2f}% |         |         | {np.max(hist_rets):>+15.2f}% |")
    
    return results


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 4: Ablation Study
# ═══════════════════════════════════════════════════════════════════

def run_experiment_4(test_df, train_df):
    print("\n\n" + "=" * 90)
    print("  EXPERIMENT 4: Ablation Study (4 Variants)")
    print("=" * 90)
    
    os.makedirs("./data/v0.3_rl_checkpoints/ablation/", exist_ok=True)
    
    ablations = [
        ("v3 Full (Baseline)", SyntheticMultiRegimeEnv, 
         dict(num_assets=20, episode_len=504, history_len=32, initial_cash=10000.0, 
              transaction_fee=0.001, rebalance_threshold=0.03),
         True, 0.03),
        
        ("v3 − Multi-Regime", AblationRandomOnlyEnv,
         dict(num_assets=20, episode_len=504, history_len=32, initial_cash=10000.0,
              transaction_fee=0.001, rebalance_threshold=0.03),
         True, 0.03),
        
        ("v3 − Trend Features", AblationNoTrendEnv,
         dict(num_assets=20, episode_len=504, history_len=32, initial_cash=10000.0,
              transaction_fee=0.001, rebalance_threshold=0.03),
         False, 0.03),
        
        ("v3 − Rebalance Threshold", AblationNoThresholdEnv,
         dict(num_assets=20, episode_len=504, history_len=32, initial_cash=10000.0,
              transaction_fee=0.001, rebalance_threshold=0.0),
         True, 0.0),
    ]
    
    results = []
    for name, env_cls, env_kwargs, include_trend, rb_thresh in ablations:
        t0 = time.time()
        print(f"\n  Training: {name}...", end=" ", flush=True)
        
        safe_name = name.replace(" ", "_").replace("−", "minus").replace("(", "").replace(")", "")
        save_path = f"./data/v0.3_rl_checkpoints/ablation/{safe_name}"
        
        model = train_model(env_cls, seed=42, total_steps=150_000, save_path=save_path, **env_kwargs)
        elapsed = time.time() - t0
        print(f"done in {elapsed:.0f}s", flush=True)
        
        eq_test = eval_model_on_real(model, test_df, include_trend=include_trend, rebalance_threshold=rb_thresh)
        m_test = compute_metrics(eq_test)
        
        eq_hist = eval_model_on_real(model, train_df, include_trend=include_trend, rebalance_threshold=rb_thresh)
        m_hist = compute_metrics(eq_hist)
        
        results.append({
            "name": name,
            "test_return": m_test["return_pct"],
            "test_sharpe": m_test["sharpe"],
            "test_max_dd": m_test["max_dd_pct"],
            "hist_return": m_hist["return_pct"],
            "hist_sharpe": m_hist["sharpe"],
        })
    
    print(f"\n\n  {'Variant':<30} | {'2020-24 Return':>14} | {'Sharpe':>7} | {'Max DD':>8} | {'2010-19 Return':>14} | {'Sharpe':>7}")
    print(f"  {'-'*90}")
    for r in results:
        print(f"  {r['name']:<30} | {r['test_return']:>+13.2f}% | {r['test_sharpe']:>6.2f} | {r['test_max_dd']:>7.2f}% | {r['hist_return']:>+13.2f}% | {r['hist_sharpe']:>6.2f}")
    print(f"  {'-'*90}")
    
    return results


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)
    
    run_experiment_1(test_df, train_df)
    run_experiment_4(test_df, train_df)
    
    print("\n\n  ✅ Experiments 1 & 4 complete!")

if __name__ == "__main__":
    main()
