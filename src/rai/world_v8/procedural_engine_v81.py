"""
====================================================================================================
🌌 RAI v8.1 LARGE PROCEDURAL WORLD ENGINE (SCIENTIFICALLY REFINED)
====================================================================================================
Procedurally generates synthetic financial environments across a combinatorial space capable of 
producing millions of distinct worlds (~200 distinct procedural worlds sampled per 100k step run).

Features:
  1. Constrained Dirichlet Segment Partitioning (sum(durations) == total_T, duration_i >= 30).
  2. FIFO Action Delay Queue (executes action from t - delay).
  3. Active Factor Returns Model: r_{i,t} = sum_k B_{i,k} * F_{k,t} + epsilon_{i,t}.
  4. Active Correlation Shift during crisis regimes (rho -> 0.85).
====================================================================================================
"""

import numpy as np

class ProceduralWorldEngineV81:
    REGIME_PROPERTIES = {
        'bull':             {'drift': (0.15, 0.40),  'vol': (0.10, 0.20), 'corr_shift': 0.0},
        'bear':             {'drift': (-0.45, -0.15),'vol': (0.25, 0.50), 'corr_shift': 0.2},
        'sideways':         {'drift': (-0.05, 0.05),  'vol': (0.08, 0.18), 'corr_shift': 0.0},
        'stagflation':      {'drift': (-0.25, -0.05),'vol': (0.20, 0.40), 'corr_shift': 0.3},
        'liquidity_crisis': {'drift': (-0.60, -0.30),'vol': (0.35, 0.70), 'corr_shift': 0.6},
        'bubble_bust':      {'drift': (-0.50, 0.30),  'vol': (0.30, 0.60), 'corr_shift': 0.5},
    }

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001):
        self.num_assets = num_assets
        self.action_dim = num_assets + 1
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee
        self.features_per_step = 2 * num_assets + 2
        self.reset()

    def _sample_world_parameters(self):
        # 1. Macro Regime Sequence & Constrained Dirichlet Partitioning
        n_segments = np.random.randint(3, 7)
        keys = list(self.REGIME_PROPERTIES.keys())
        regime_seq = [keys[np.random.randint(len(keys))] for _ in range(n_segments)]
        
        total_T = self.episode_len + self.history_len + 15
        min_dur = 30
        rem_T = total_T - n_segments * min_dur
        props = np.random.dirichlet(np.ones(n_segments))
        segment_durations = (min_dur + props * rem_T).astype(int)
        segment_durations[-1] = total_T - int(np.sum(segment_durations[:-1]))

        # 2. Multi-Factor Exposure Matrix B (N x K)
        n_factors = np.random.randint(2, 5)
        factor_loadings = np.random.uniform(-0.8, 0.8, size=(self.num_assets, n_factors))

        # 3. Base Asset Correlation Matrix
        A = np.random.randn(self.num_assets, self.num_assets)
        corr_matrix = A @ A.T
        d = np.sqrt(np.diag(corr_matrix))
        base_corr = corr_matrix / np.outer(d, d)

        # 4. Micro Noise & Execution Delay
        execution_delay = int(np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
        noise_level = float(np.random.uniform(0.001, 0.005))
        heavy_tail_df = float(np.random.uniform(4.0, 15.0))
        jump_intensity = float(np.random.uniform(0.005, 0.03))
        jump_size_std = float(np.random.uniform(0.01, 0.06))

        return {
            'regimes': regime_seq,
            'durations': segment_durations,
            'n_factors': n_factors,
            'factor_loadings': factor_loadings,
            'base_corr': base_corr,
            'execution_delay': execution_delay,
            'noise_level': noise_level,
            'heavy_tail_df': heavy_tail_df,
            'jump_intensity': jump_intensity,
            'jump_size_std': jump_size_std,
            'total_T': total_T
        }

    def _generate_procedural_prices(self, cfg):
        total_T = cfg['total_T']
        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)

        init_prices = np.random.uniform(20.0, 300.0, size=self.num_assets)
        prices[0] = init_prices

        day = 0
        current_vol = np.ones(self.num_assets) * (0.15 / np.sqrt(252.0))**2
        omega, alpha, beta = 0.000005, 0.05, 0.90

        for reg_idx, dur in enumerate(cfg['durations']):
            reg_name = cfg['regimes'][reg_idx]
            reg_props = self.REGIME_PROPERTIES[reg_name]

            corr_shift = reg_props['corr_shift']
            target_corr = (1.0 - corr_shift) * cfg['base_corr'] + corr_shift * np.ones((self.num_assets, self.num_assets))
            np.fill_diagonal(target_corr, 1.0)
            
            try:
                L = np.linalg.cholesky(target_corr)
            except np.linalg.LinAlgError:
                L = np.eye(self.num_assets)

            drift_annual = np.random.uniform(*reg_props['drift']) + np.random.uniform(-0.05, 0.05, size=self.num_assets)

            for _ in range(max(0, min(dur, total_T - day - 1))):
                day += 1
                if day >= total_T: break

                factor_returns = np.random.normal(0, 1.0, size=cfg['n_factors'])
                factor_component = cfg['factor_loadings'] @ factor_returns

                z_raw = np.random.standard_t(df=cfg['heavy_tail_df'], size=self.num_assets)
                z_raw = np.clip(z_raw, -4.0, 4.0)
                z = L @ z_raw + factor_component * 0.3

                current_vol = omega + alpha * (z**2) + beta * current_vol
                current_vol = np.clip(current_vol, 1e-6, 0.01)
                stoch_vol = np.sqrt(current_vol)

                jump_occured = (np.random.rand(self.num_assets) < cfg['jump_intensity'])
                jumps = jump_occured * np.random.normal(0, cfg['jump_size_std'], size=self.num_assets)
                jumps = np.clip(jumps, -0.15, 0.15)

                total_drift = drift_annual / 252.0
                p_prev = prices[day - 1]
                log_return = (total_drift - 0.5 * stoch_vol**2) + stoch_vol * z + jumps

                if cfg['noise_level'] > 0:
                    log_return += np.random.normal(0, cfg['noise_level'], size=self.num_assets)

                log_return = np.clip(log_return, -0.25, 0.25)
                prices[day] = np.maximum(0.01, p_prev * np.exp(log_return))

            if day >= total_T: break

        while day < total_T:
            prices[day] = prices[max(0, day - 1)]
            day += 1

        return np.nan_to_num(prices, nan=100.0, posinf=500.0, neginf=0.01)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.world_cfg = self._sample_world_parameters()
        self.prices = self._generate_procedural_prices(self.world_cfg)
        self.start = self.history_len
        self.current_step = self.start
        self.cash = self.initial_cash * 0.5
        init_p = self.prices[self.current_step]
        self.shares = (self.initial_cash * 0.5 / self.num_assets) / np.maximum(1e-4, init_p)
        self.peak_wealth = self.initial_cash
        self.last_wealth = self.initial_cash
        self.steps_done = 0

        # Clean FIFO Action Delay Queue
        self.delay = self.world_cfg['execution_delay']
        self.action_queue = [np.zeros(self.action_dim, dtype=np.float32) for _ in range(self.delay)]

        self.obs_history = [self._obs_at(self.start - self.history_len + i) for i in range(self.history_len)]
        return self._flat_obs()

    def _wealth(self):
        return self.cash + np.sum(self.shares * self.prices[self.current_step])

    def _obs_at(self, t):
        p = self.prices[t]; p_prev = self.prices[max(0, t-1)]
        w = max(1e-4, self.cash + np.sum(self.shares * p))
        norm_prices = p / np.maximum(1e-4, self.prices[self.start])
        log_rets = np.log(p / np.maximum(1e-4, p_prev))
        log_rets = np.clip(log_rets, -0.5, 0.5)
        cash_ratio = self.cash / w
        dd = np.clip((w - self.peak_wealth) / max(1e-4, self.peak_wealth), -1.0, 0.0)
        obs = np.concatenate([norm_prices, log_rets, [cash_ratio, dd]]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def _flat_obs(self):
        return np.concatenate(self.obs_history).astype(np.float32)

    def step(self, action):
        action = np.nan_to_num(action, nan=0.0)
        
        # Clean FIFO Queue Execution Delay
        self.action_queue.append(action)
        exec_action = self.action_queue.pop(0)

        cash_logit = np.clip(exec_action[0], -5.0, 5.0)
        target_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash_frac
        asset_logits = exec_action[1:]
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
        new_wealth = max(1e-4, self._wealth())
        self.peak_wealth = max(self.peak_wealth, new_wealth)

        daily_ret = (new_wealth - self.last_wealth) / max(1e-4, self.last_wealth)
        daily_ret = np.clip(daily_ret, -0.5, 0.5)
        reward = daily_ret * 5.0
        if daily_ret < 0: reward *= 2.0
        drawdown = (new_wealth - self.peak_wealth) / max(1e-4, self.peak_wealth)
        if drawdown < -0.10: reward += drawdown * 2.0

        done = self.current_step >= self.prices.shape[0] - 1 or self.steps_done >= self.episode_len
        self.last_wealth = new_wealth
        self.obs_history.pop(0)
        self.obs_history.append(self._obs_at(self.current_step))

        return self._flat_obs(), float(reward), done, {"portfolio_value": new_wealth}
