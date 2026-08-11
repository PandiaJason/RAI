"""
RAI v5: Dual-Head Gated Policy with Auxiliary Regime Classifier
================================================================
Solves Policy Collapse (Greed vs Fear) via:
1. Shared Encoder (Obs -> 128 hidden features)
2. Auxiliary Classifier Head (128 -> 3 regime probabilities: Bull, Bear, Sideways)
3. Dual/Triple Action Heads:
   - Bull Head (High Stock Allocation)
   - Bear Head (High Cash Preservation)
   - Sideways Head (Balanced Allocation)
4. Softmax Gating: Blends heads based on predicted regime probability

Trained using PPO + Auxiliary Regime Classification Loss.
"""
import os, sys, time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rai.world.v5_regime_env import SyntheticRegimeSupervisedEnv


# ═══════════════════════════════════════════════════════════════════
#  DUAL-HEAD GATED NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

class DualHeadGatedPolicy(nn.Module):
    """
    Custom Neural Network with Auxiliary Regime Classification
    and Softmax-Gated Multi-Head Action Outputs.
    """
    def __init__(self, obs_dim, action_dim=11, num_regimes=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_regimes = num_regimes

        # Shared Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
        )

        # Auxiliary Head: Regime Classifier (Bull=0, Bear=1, Sideways=2)
        self.regime_classifier = nn.Linear(128, num_regimes)

        # Regime-Specific Action Heads
        self.bull_action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, action_dim)
        )
        self.bear_action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, action_dim)
        )
        self.sideways_action_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, action_dim)
        )

        # Action Standard Deviation (learnable log_std)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

        # Value Network (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        feat = self.encoder(obs)

        # 1. Regime prediction logits & probabilities
        regime_logits = self.regime_classifier(feat)
        regime_probs = F.softmax(regime_logits, dim=-1) # (batch, 3)

        # 2. Action head outputs
        bull_act = self.bull_action_head(feat)     # (batch, action_dim)
        bear_act = self.bear_action_head(feat)     # (batch, action_dim)
        side_act = self.sideways_action_head(feat) # (batch, action_dim)

        # 3. Softmax Gating: Weighted blend of action heads
        p_bull = regime_probs[:, 0:1]
        p_bear = regime_probs[:, 1:2]
        p_side = regime_probs[:, 2:3]

        gated_action_mean = (p_bull * bull_act +
                             p_bear * bear_act +
                             p_side * side_act)

        # Force Bear head to default to high cash if regime is predicted Bear
        # cash_logit is index 0: bias it positively during bear
        gated_action_mean[:, 0] = gated_action_mean[:, 0] + (p_bear.squeeze(-1) * 3.0) - (p_bull.squeeze(-1) * 2.0)

        value = self.value_head(obs)

        return gated_action_mean, regime_logits, regime_probs, value

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).unsqueeze(0) if obs.ndim == 1 else torch.FloatTensor(obs)
            
            mean, logits, probs, val = self.forward(obs)
            
            if deterministic:
                action = mean
            else:
                std = torch.exp(self.log_std)
                dist = Normal(mean, std)
                action = dist.sample()
            
            return action.cpu().numpy().squeeze(0), probs.cpu().numpy().squeeze(0)


# ═══════════════════════════════════════════════════════════════════
#  RAI v5 PPO + AUXILIARY REGIME TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_v5():
    print("=" * 80, flush=True)
    print("  RAI v5: Dual-Head Gated Policy + Auxiliary Regime Classifier", flush=True)
    print("=" * 80, flush=True)

    env = SyntheticRegimeSupervisedEnv(num_assets=10, episode_len=504, history_len=16)
    obs_dim = env.obs_dim
    action_dim = env.action_space.shape[0]

    policy = DualHeadGatedPolicy(obs_dim=obs_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Total Parameters: {total_params:,}", flush=True)
    print(f"  Obs Dim: {obs_dim} | Action Dim: {action_dim}", flush=True)
    print(f"  Training 200,000 steps with Joint RL + Auxiliary Losses...\n", flush=True)

    BATCH_SIZE = 64
    ROLLOUT_LEN = 2048
    TOTAL_STEPS = 200_000
    N_EPOCHS = 8

    obs_buf, act_buf, rew_buf, val_buf, logp_buf, reg_buf = [], [], [], [], [], []

    obs, info = env.reset(seed=42)
    step = 0
    t0 = time.time()

    total_deaths = 0

    while step < TOTAL_STEPS:
        # Collect Rollout
        obs_buf.clear(); act_buf.clear(); rew_buf.clear(); val_buf.clear(); logp_buf.clear(); reg_buf.clear()

        for _ in range(ROLLOUT_LEN):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean, logits, probs, val = policy(obs_t)
                std = torch.exp(policy.log_std)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            act_np = action.squeeze(0).numpy()
            next_obs, reward, done, _, info = env.step(act_np)

            obs_buf.append(obs)
            act_buf.append(act_np)
            rew_buf.append(reward)
            val_buf.append(val.item())
            logp_buf.append(log_prob.item())
            reg_buf.append(info['regime_label'])

            obs = next_obs
            step += 1

            if done:
                obs, info = env.reset()

        # Compute Advantages (GAE)
        with torch.no_grad():
            _, _, _, next_val = policy(torch.FloatTensor(obs).unsqueeze(0))
            next_val = next_val.item()

        rewards = np.array(rew_buf)
        values = np.array(val_buf + [next_val])
        deltas = rewards + 0.99 * values[1:] - values[:-1]

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + 0.99 * 0.95 * gae
            advantages[t] = gae
        returns = advantages + values[:-1]

        # Convert to Tensors
        obs_tensor = torch.FloatTensor(np.array(obs_buf))
        act_tensor = torch.FloatTensor(np.array(act_buf))
        adv_tensor = torch.FloatTensor(advantages)
        ret_tensor = torch.FloatTensor(returns)
        old_logp_tensor = torch.FloatTensor(np.array(logp_buf))
        reg_tensor = torch.LongTensor(np.array(reg_buf))

        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # PPO Update Loop
        dataset_size = len(obs_buf)
        for epoch in range(N_EPOCHS):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = indices[start:end]

                b_obs = obs_tensor[batch_idx]
                b_act = act_tensor[batch_idx]
                b_adv = adv_tensor[batch_idx]
                b_ret = ret_tensor[batch_idx]
                b_old_logp = old_logp_tensor[batch_idx]
                b_reg = reg_tensor[batch_idx]

                mean, logits, probs, val = policy(b_obs)
                std = torch.exp(policy.log_std)
                dist = Normal(mean, std)
                new_logp = dist.log_prob(b_act).sum(dim=-1)

                # PPO Ratio
                ratio = torch.exp(new_logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(val.squeeze(-1), b_ret)

                # Auxiliary Loss: Regime Classification
                aux_regime_loss = F.cross_entropy(logits, b_reg)

                # Total Loss
                total_loss = policy_loss + 0.5 * value_loss + 1.0 * aux_regime_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        # Monitoring
        if step % 20000 < ROLLOUT_LEN:
            with torch.no_grad():
                _, sample_logits, sample_probs, _ = policy(obs_tensor[:200])
                acc = (sample_logits.argmax(dim=-1) == reg_tensor[:200]).float().mean().item()
                c_frac = 1.0 / (1.0 + torch.exp(-act_tensor[:, 0]))
                c_min, c_mean, c_max = c_frac.min().item(), c_frac.mean().item(), c_frac.max().item()

            print(f"  Step {step:>7d} | Regime Acc: {acc*100:5.1f}% | Cash Frac: min={c_min:.3f} mean={c_mean:.3f} max={c_max:.3f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Trained 200k steps in {elapsed:.0f}s ({TOTAL_STEPS/elapsed:.0f} FPS)", flush=True)

    # Save v5 model
    os.makedirs("./data/v0.5_rl_checkpoints/", exist_ok=True)
    save_path = "./data/v0.5_rl_checkpoints/rai_v5_dual_head.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"  Model saved to {save_path}", flush=True)

    return policy


# ═══════════════════════════════════════════════════════════════════
#  REAL MARKET EVALUATION BRIDGE
# ═══════════════════════════════════════════════════════════════════

class RealMarketV5Env(gym.Env):
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
        self.log_regime_probs = []
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

    def step(self, action_tuple):
        action, regime_prob = action_tuple
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
        self.log_regime_probs.append(regime_prob)
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


def eval_v5(policy):
    test_df = pd.read_csv("./data/real_market_checkpoints/test_prices.csv", index_col=0, parse_dates=True)
    train_df = pd.read_csv("./data/real_market_checkpoints/train_prices.csv", index_col=0, parse_dates=True)

    policy.eval()

    for label, df in [("2020-2024 (Out-of-Sample)", test_df), ("2010-2019 (Historical)", train_df)]:
        print(f"\n{'='*80}", flush=True)
        print(f"  EVALUATION: {label}", flush=True)
        print(f"{'='*80}", flush=True)

        env = RealMarketV5Env(price_df=df, max_assets=10)
        obs, _ = env.reset()
        done = False

        while not done:
            act, probs = policy.get_action(obs, deterministic=True)
            obs, _, done, _, info = env.step((act, probs))

        eq = [10000.0] + env.log_wealth
        m = metrics(eq)
        cf = np.array(env.log_cash_frac)
        rp = np.array(env.log_regime_probs) # (T, 3)

        print(f"  Final Wealth:    ${m['final']:,.2f}", flush=True)
        print(f"  Total Return:    {m['return']:+.2f}%", flush=True)
        print(f"  Volatility:      {m['vol']:.2f}%", flush=True)
        print(f"  Sharpe Ratio:    {m['sharpe']:.2f}", flush=True)
        print(f"  Max Drawdown:    {m['max_dd']:.2f}%", flush=True)
        print(f"  Cash Fraction:   min={np.min(cf):.4f}  mean={np.mean(cf):.4f}  max={np.max(cf):.4f}", flush=True)
        print(f"  Cash Range:      {(np.max(cf)-np.min(cf))*100:.1f}%", flush=True)

        # Average Predicted Regime Distribution
        mean_p_bull = np.mean(rp[:, 0]) * 100
        mean_p_bear = np.mean(rp[:, 1]) * 100
        mean_p_side = np.mean(rp[:, 2]) * 100
        print(f"  Predicted Regime Split: Bull={mean_p_bull:.1f}%, Bear={mean_p_bear:.1f}%, Sideways={mean_p_side:.1f}%", flush=True)

        # COVID crash check
        if '2020' in label:
            dates = df.index
            offset = env.history_len + 20
            pre = [i-offset for i,d in enumerate(dates) if str(d)[:7]=='2020-01' and i>=offset and i-offset<len(cf)]
            crash = [i-offset for i,d in enumerate(dates) if str(d)[:7] in ['2020-02','2020-03'] and i>=offset and i-offset<len(cf)]
            post = [i-offset for i,d in enumerate(dates) if str(d)[:7] in ['2020-05','2020-06'] and i>=offset and i-offset<len(cf)]
            if pre and crash:
                print(f"\n  COVID CRASH ADAPTATION CHECK:", flush=True)
                print(f"    Jan 2020 (Pre-Crash):    {np.mean([cf[i] for i in pre]):.4f} Cash | Predicted Bear Prob: {np.mean([rp[i,1] for i in pre])*100:.1f}%", flush=True)
                print(f"    Feb-Mar 2020 (Crash):    {np.mean([cf[i] for i in crash]):.4f} Cash | Predicted Bear Prob: {np.mean([rp[i,1] for i in crash])*100:.1f}%", flush=True)
                if post:
                    print(f"    May-Jun 2020 (Post-Crash): {np.mean([cf[i] for i in post]):.4f} Cash | Predicted Bear Prob: {np.mean([rp[i,1] for i in post])*100:.1f}%", flush=True)

        # SPY comparison
        spy = df['SPY'].values
        m_spy = metrics(10000*(spy/spy[0]))
        print(f"\n  vs SPY Buy & Hold: {m_spy['return']:+.2f}% return, {m_spy['sharpe']:.2f} Sharpe, {m_spy['max_dd']:.2f}% maxDD", flush=True)


if __name__ == "__main__":
    policy = train_v5()
    eval_v5(policy)
