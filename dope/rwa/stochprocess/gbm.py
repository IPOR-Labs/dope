import numpy as np

from dope.rwa.vars import Vars


class GBMProcess:
    def __init__(self, mu: float, sigma: float):
        self.mu = float(mu)
        self.sigma = float(max(sigma, 0.0))

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
        T = float(n_days) / Vars.TRADING_DAYS_PER_YEAR
        Z = rng.standard_normal(n_sims)
        log_return = ((self.mu - 0.5 * self.sigma**2) * T) + (
            self.sigma * np.sqrt(T) * Z
        )
        return V0 * np.exp(log_return)

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
            z = rng.standard_normal()
            dt_years = dt / Vars.TRADING_DAYS_PER_YEAR
            drift = (self.mu - 0.5 * self.sigma**2) * dt_years
            step = drift + self.sigma * np.sqrt(dt_years) * z
            path[t] = path[t - 1] * np.exp(step)

        return path
