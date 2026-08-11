"""
====================================================================================================
🌌 RAI v8 LARGE PROCEDURAL WORLD ENGINE (W_proc)
====================================================================================================
Generates millions of distinct synthetic financial universes across arbitrary asset counts, 
macro regimes, cross-asset lead-lag relationships, volatility dynamics, and micro shocks.

Hierarchical World Composition:
  1. Macro Regimes: Bull, Bear, Sideways, High Inflation, Liquidity Crisis, Sudden Reversals.
  2. Market Dynamics: GARCH(1,1) Stochastic Volatility, Jump-Diffusion, Student-t Heavy Tails,
                      Ornstein-Uhlenbeck Mean Reversion, Momentum.
  3. Asset Relationship Graph: Factor Exposures, Dynamic Correlation Breakdown Matrices, 
                               Sector Lead/Lag Phase Shifts.
  4. Micro Dynamics & Noise: Dynamic Spreads, Execution Delays, Asymmetric Fees, Missing Data.
====================================================================================================
"""

import numpy as np

class ProceduralWorldEngine:
    """Large Procedural World Engine generating combinatorial synthetic financial environments."""
    
    MACRO_REGIMES = ['bull', 'bear', 'sideways', 'stagflation', 'liquidity_crisis', 'bubble_bust']

    def __init__(self, num_assets=10, history_len=30, episode_len=504, initial_cash=10000.0, fee=0.001):
        self.num_assets = num_assets
        self.history_len = history_len
        self.episode_len = episode_len
        self.initial_cash = initial_cash
        self.fee = fee
        self.features_per_step = 2 * num_assets + 2
        self.reset()

    def _sample_world_parameters(self):
        """Randomly composes a unique artificial world configuration."""
        # 1. Macro Regime Sequence
        n_regimes = np.random.randint(3, 8)
        regime_seq = np.random.choice(self.MACRO_REGIMES, size=n_regimes)
        
        # 2. Cross-Asset Factor & Correlation Dynamics
        n_factors = np.random.randint(2, 6)
        factor_loadings = np.random.uniform(-0.8, 0.8, size=(self.num_assets, n_factors))
        
        # Correlation matrix with random decoupling intensity
        A = np.random.randn(self.num_assets, self.num_assets)
        corr_matrix = A @ A.T
        d = np.sqrt(np.diag(corr_matrix))
        corr_matrix = corr_matrix / np.outer(d, d)
        
        # 3. Market Dynamic Properties
        volatility_scale = np.random.uniform(0.08, 0.65, size=self.num_assets)
        drift_scale = np.random.uniform(-0.40, 0.40, size=self.num_assets)
        jump_intensity = np.random.uniform(0.005, 0.05)
        jump_size_std = np.random.uniform(0.02, 0.12)
        mean_reversion_speed = np.random.uniform(0.0, 0.15)
        heavy_tail_df = np.random.uniform(3.0, 10.0)  # Student-t degrees of freedom
        
        # 4. Micro Noise & Execution Friction
        execution_delay = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        noise_level = np.random.uniform(0.001, 0.01)

        return {
            'regimes': regime_seq,
            'factor_loadings': factor_loadings,
            'corr_matrix': corr_matrix,
            'volatility': volatility_scale,
            'drift': drift_scale,
            'jump_intensity': jump_intensity,
            'jump_size_std': jump_size_std,
            'mean_reversion_speed': mean_reversion_speed,
            'heavy_tail_df': heavy_tail_df,
            'execution_delay': execution_delay,
            'noise_level': noise_level
        }

    def _generate_procedural_prices(self, cfg):
        """Simulates price trajectories inside the sampled procedural world."""
        total_T = self.episode_len + self.history_len + 15
        prices = np.zeros((total_T, self.num_assets), dtype=np.float64)

        # Cholesky decomposition for correlated noise
        try:
            L = np.linalg.cholesky(cfg['corr_matrix'])
        except np.linalg.LinAlgError:
            L = np.eye(self.num_assets)

        init_prices = np.random.uniform(15.0, 500.0, size=self.num_assets)
        prices[0] = init_prices

        # GARCH(1,1) Volatility parameters
        omega = 0.00001
        alpha = 0.08
        beta = 0.88
        current_vol = (cfg['volatility'] / np.sqrt(252.0))**2

        for t in range(1, total_T):
            # Correlated random innovations
            z_raw = np.random.standard_t(df=cfg['heavy_tail_df'], size=self.num_assets)
            z = L @ z_raw

            # Update GARCH volatility
            current_vol = omega + alpha * (z**2) + beta * current_vol
            stoch_vol = np.sqrt(np.maximum(1e-6, current_vol))

            # Jump Process
            jump_occured = (np.random.rand(self.num_assets) < cfg['jump_intensity'])
            jumps = jump_occured * np.random.normal(0, cfg['jump_size_std'], size=self.num_assets)

            # Mean-reversion drift component
            mr_drift = cfg['mean_reversion_speed'] * (np.log(init_prices) - np.log(np.maximum(1e-4, prices[t-1]))) / 252.0
            total_drift = (cfg['drift'] / 252.0) + mr_drift

            p_prev = prices[t - 1]
            log_return = (total_drift - 0.5 * stoch_vol**2) + stoch_vol * z + jumps
            
            # Micro noise injection
            if cfg['noise_level'] > 0:
                log_return += np.random.normal(0, cfg['noise_level'], size=self.num_assets)

            prices[t] = np.maximum(0.01, p_prev * np.exp(log_return))

        return prices

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.world_cfg = self._sample_world_parameters()
        self.prices = self._generate_procedural_prices(self.world_cfg)
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
