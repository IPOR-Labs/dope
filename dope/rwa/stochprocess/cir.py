import numpy as np

from dope.rwa.vars import Vars


class CIRProcess:
    """
    Cox-Ingersoll-Ross (CIR) mean-reverting square-root diffusion.
    """

    def __init__(self, kappa: float, theta: float, sigma: float):
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)

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
            dt_years = dt / Vars.TRADING_DAYS_PER_YEAR
            sqrt_dt = np.sqrt(dt_years)
            dW = rng.standard_normal(n_sims) * sqrt_dt
            sqrt_V = np.sqrt(np.maximum(V, 0.0))
            V = V + self.kappa * (self.theta - V) * dt_years + self.sigma * sqrt_V * dW
            V = np.maximum(V, 1e-12)

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
        path[0] = max(float(V0), 1e-12)

        for t, dt in enumerate(steps, start=1):
            prev = max(path[t - 1], 0.0)
            dt_years = dt / Vars.TRADING_DAYS_PER_YEAR
            sqrt_dt = np.sqrt(dt_years)
            dW = rng.standard_normal() * sqrt_dt
            nxt = (
                prev
                + self.kappa * (self.theta - prev) * dt_years
                + self.sigma * np.sqrt(prev) * dW
            )
            path[t] = max(nxt, 1e-12)

        return path
