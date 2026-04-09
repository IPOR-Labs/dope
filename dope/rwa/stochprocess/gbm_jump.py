import numpy as np

from dope.rwa.vars import Vars


class GBMJumpProcess:
    """
    Geometric Brownian motion with independent positive and negative jump channels.

    Parameters are interpreted on an annualized basis:
    - ``mu`` is the annualized drift of log returns
    - ``sigma`` is the annualized diffusion volatility
    - ``lambda_pos`` and ``lambda_neg`` are annual jump arrival intensities

    Simulation ``step_size`` is expressed in days and converted internally to a
    year fraction using ``Vars.TRADING_DAYS_PER_YEAR`` (``from dope.rwa.vars import Vars``).

    Over a single step, the log-return increment is:

    ``dlogV = (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z + J_pos + J_neg``

    where ``dt`` is the step length in years, ``Z`` is standard normal, and
    ``J_pos`` and ``J_neg`` are compound Poisson-normal jump contributions.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        lambda_pos: float = 0.0,
        lambda_neg: float = 0.0,
        jump_mean_pos: float = 0.0,
        jump_std_pos: float = 0.0,
        jump_mean_neg: float = 0.0,
        jump_std_neg: float = 0.0,
    ):
        self.mu = float(mu)
        self.sigma = float(max(sigma, 0.0))
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

    def _log_step(self, n: int, dt: float, rng: np.random.Generator) -> np.ndarray:
        dt_years = dt / Vars.TRADING_DAYS_PER_YEAR
        diffusion = (self.mu - 0.5 * self.sigma**2) * dt_years + self.sigma * np.sqrt(
            dt_years
        ) * rng.standard_normal(n)

        n_pos = rng.poisson(self.lambda_pos * dt_years, size=n)
        n_neg = rng.poisson(self.lambda_neg * dt_years, size=n)

        raw_pos = self._sum_normal_jumps(
            n_pos, self.jump_mean_pos, self.jump_std_pos, rng
        )
        raw_neg = self._sum_normal_jumps(
            n_neg, self.jump_mean_neg, self.jump_std_neg, rng
        )

        jumps_pos = np.maximum(raw_pos, 0.0)
        jumps_neg = np.minimum(raw_neg, 0.0)
        return diffusion + jumps_pos + jumps_neg

    def simulate_terminal(
        self,
        V0: float,
        n_days: float,
        n_sims: int,
        rng: np.random.Generator,
        step_size: float = 1.0,
    ) -> np.ndarray:
        V = np.full(n_sims, float(V0), dtype=float)
        for dt in self._step_grid(n_days, step_size):
            step = self._log_step(n_sims, dt, rng)
            V *= np.exp(step)
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
        for t, dt in enumerate(steps, start=1):
            step = self._log_step(1, dt, rng)[0]
            path[t] = path[t - 1] * np.exp(step)
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
        if r.size < 10:
            raise ValueError("Need at least 10 return observations to calibrate jump model.")

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
        base_mask = ~(pos_mask | neg_mask)

        base = r[base_mask]
        if base.size < 5:
            base = r

        mu = float(np.mean(base))
        sigma = float(max(np.std(base, ddof=1), min_std))

        n = float(r.size)
        lambda_pos = float(np.sum(pos_mask) / n)
        lambda_neg = float(np.sum(neg_mask) / n)

        pos_vals = r[pos_mask]
        neg_vals = r[neg_mask]

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
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            jump_mean_pos=jump_mean_pos,
            jump_std_pos=jump_std_pos,
            jump_mean_neg=jump_mean_neg,
            jump_std_neg=jump_std_neg,
        )
