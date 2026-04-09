import numpy as np

from dope.rwa.vars import Vars


class GBMTransientJumpProcess:
    """
    GBM with transient (reverting) jump state.

    Log-return over a step dt:
        dlogV_t = (mu - 0.5*sigma^2) * dt + sigma*sqrt(dt)*Z_t + s_t

    Jump-state dynamics:
        s_{t+1} = -reversion * s_t + eta_t

    where eta_t is a rare jump innovation (compound Poisson-normal, split into
    positive and negative innovations), and reversion in [0, 1) controls how
    quickly jump impacts revert with sign flip.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        reversion: float = 0.5,
        lambda_pos: float = 0.0,
        lambda_neg: float = 0.0,
        jump_mean_pos: float = 0.0,
        jump_std_pos: float = 0.0,
        jump_mean_neg: float = 0.0,
        jump_std_neg: float = 0.0,
    ):
        self.mu = float(mu)
        self.sigma = float(max(sigma, 0.0))
        self.reversion = float(np.clip(reversion, 0.0, 0.999))

        self.lambda_pos = float(max(lambda_pos, 0.0))
        self.lambda_neg = float(max(lambda_neg, 0.0))
        self.jump_mean_pos = float(jump_mean_pos)
        self.jump_std_pos = float(max(jump_std_pos, 0.0))
        self.jump_mean_neg = float(jump_mean_neg)
        self.jump_std_neg = float(max(jump_std_neg, 0.0))

    @staticmethod
    def _step_grid(n_days: float, step_size: float) -> list[float]:
        horizon = float(n_days)
        dt = float(step_size)
        if horizon < 0:
            raise ValueError("n_days must be >= 0")
        if dt <= 0:
            raise ValueError("step_size must be > 0")

        steps = []
        remaining = horizon
        while remaining > 1e-12:
            step = min(dt, remaining)
            steps.append(step)
            remaining -= step
        return steps

    @staticmethod
    def _sum_normal_jumps(
        n_events: np.ndarray, mean: float, std: float, rng: np.random.Generator
    ) -> np.ndarray:
        totals = np.zeros_like(n_events, dtype=float)
        idx = np.where(n_events > 0)[0]
        for i in idx:
            totals[i] = np.sum(
                rng.normal(loc=mean, scale=std, size=int(n_events[i]))
            )
        return totals

    def _jump_innovation(
        self, n: int, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        dt_years = dt / Vars.TRADING_DAYS_PER_YEAR
        n_pos = rng.poisson(self.lambda_pos * dt_years, size=n)
        n_neg = rng.poisson(self.lambda_neg * dt_years, size=n)
        jumps_pos = self._sum_normal_jumps(
            n_pos, self.jump_mean_pos, self.jump_std_pos, rng
        )
        jumps_neg = self._sum_normal_jumps(
            n_neg, self.jump_mean_neg, self.jump_std_neg, rng
        )
        return jumps_pos + jumps_neg

    def simulate_terminal(
        self,
        V0: float,
        n_days: float,
        n_sims: int,
        rng: np.random.Generator,
        step_size: float = 1.0,
    ) -> np.ndarray:
        V = np.full(n_sims, float(V0), dtype=float)
        state = np.zeros(n_sims, dtype=float)

        for dt in self._step_grid(n_days, step_size):
            dt_years = dt / Vars.TRADING_DAYS_PER_YEAR
            drift = (self.mu - 0.5 * self.sigma**2) * dt_years
            diffusion = drift + self.sigma * np.sqrt(dt_years) * rng.standard_normal(n_sims)
            step = diffusion + state
            V *= np.exp(step)
            state = -self.reversion * state + self._jump_innovation(n_sims, dt, rng)

        return V

    def simulate_path(
        self,
        V0: float,
        n_days: float,
        rng: np.random.Generator,
        step_size: float = 1.0,
    ) -> np.ndarray:
        steps = self._step_grid(n_days, step_size)
        path = np.empty(len(steps) + 1, dtype=float)
        path[0] = float(V0)
        state = 0.0

        for t, dt in enumerate(steps, start=1):
            dt_years = dt / Vars.TRADING_DAYS_PER_YEAR
            drift = (self.mu - 0.5 * self.sigma**2) * dt_years
            diffusion = drift + self.sigma * np.sqrt(dt_years) * rng.standard_normal()
            step = diffusion + state
            path[t] = path[t - 1] * np.exp(step)
            state = -self.reversion * state + self._jump_innovation(1, dt, rng)[0]

        return path

    @classmethod
    def calibrate(
        cls,
        returns,
        is_log_returns: bool = False,
        jump_z: float = 2.5,
        min_std: float = 1e-8,
    ):
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        if r.size < 20:
            raise ValueError(
                "Need at least 20 return observations to calibrate transient jump model."
            )

        if not is_log_returns:
            if np.any(r <= -1.0):
                raise ValueError("Simple returns must be > -1.")
            r = np.log1p(r)

        center = float(np.mean(r))
        scale = float(np.std(r, ddof=1))
        scale = max(scale, min_std)

        z = (r - center) / scale
        pos_mask = z > jump_z
        neg_mask = z < -jump_z
        jump_mask = pos_mask | neg_mask
        base = r[~jump_mask]
        if base.size < 5:
            base = r

        mu = float(np.mean(base))
        sigma = float(max(np.std(base, ddof=1), min_std))

        centered = r - mu
        if centered.size > 1:
            x = centered[:-1]
            y = centered[1:]
            denom = float(np.dot(x, x))
            acf1 = float(np.dot(x, y) / denom) if denom > 0 else 0.0
        else:
            acf1 = 0.0
        reversion = float(np.clip(-acf1, 0.0, 0.999))

        n = float(r.size)
        lambda_pos = float(np.sum(pos_mask) / n)
        lambda_neg = float(np.sum(neg_mask) / n)

        pos_vals = centered[pos_mask]
        neg_vals = centered[neg_mask]

        jump_mean_pos = float(np.mean(pos_vals)) if pos_vals.size > 0 else 0.0
        jump_std_pos = (
            float(max(np.std(pos_vals, ddof=1), min_std)) if pos_vals.size > 1 else 0.0
        )
        jump_mean_neg = float(np.mean(neg_vals)) if neg_vals.size > 0 else 0.0
        jump_std_neg = (
            float(max(np.std(neg_vals, ddof=1), min_std)) if neg_vals.size > 1 else 0.0
        )

        return cls(
            mu=mu,
            sigma=sigma,
            reversion=reversion,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            jump_mean_pos=jump_mean_pos,
            jump_std_pos=jump_std_pos,
            jump_mean_neg=jump_mean_neg,
            jump_std_neg=jump_std_neg,
        )

    @classmethod
    def fit(cls, returns, **kwargs):
        return cls.calibrate(returns=returns, **kwargs)
