from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt

from dope.rwa.stochprocess import GBMProcess, CIRProcess, GBMJumpProcess, GBMTransientJumpProcess
from dope.rwa.utilities import cara_liquidity_value


class RWA:
    """
    RWA = Real World Asset liquidity decision model.
    """

    def __init__(
        self,
        V0: float,
        gamma: float = 1.0,
        fee: float = 0.0,
        seed: int = 42,
        model: str = "gbm",
        process=None,
        model_params: dict = None,
        step_size: float = 1.0,
    ):
        """
        Initialize an RWA liquidity-decision model.

        Parameters
        ----------
        V0 : float
            Current asset value.
        gamma : float, default 1.0
            CARA risk-aversion parameter used in liquidity valuation.
        fee : float, default 0.0
            Proportional fee applied to the terminal value when waiting.
        seed : int, default 42
            Random seed used for Monte Carlo simulation.
        model : str, default "gbm"
            Built-in process family. Supported values are ``"gbm"``, ``"cir"``,
            ``"gbm_jump"``, and ``"gbm_transient_jump"``.
        process : object, optional
            Custom process object. If provided, it is used directly instead of
            constructing one from ``model`` and ``model_params``.
        model_params : dict, optional
            Parameters used to build the selected built-in process.
            For GBM-style models use ``mu`` and ``sigma``.
            For CIR use ``kappa``, ``theta``, and ``sigma``.
        step_size : float, default 1.0
            Simulation step size in days. This is passed through to
            ``simulate_terminal`` and ``simulate_path``. The default is one day.
            
        Example:
        ```python
        
        from dope.rwa import RWA

        rwa = RWA(
            V0=1.07,
            gamma=gamma,
            model="gbm",
            model_params={
                "mu": -0.0001,
                "sigma": 0.0015,
            },
            seed=42,
        )
        gbm_path = rwa.simulate_path_with_timestamps(n_days=30, step_size=1)

        decision = rwa.decide(n_days=30, slippage=0.005)

        print("Decision:", decision["decision"])
        print("I sell if I get more than         :", decision["L_wait"])
        print("Current swap price (with slippage):", decision["X_sell"])
        
        ```
        """
        self.V0 = V0
        self.gamma = gamma
        self.fee = fee
        self.rng = np.random.default_rng(seed)
        self.model = model.lower()
        self.step_size = float(step_size)
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0")
        params = dict(model_params or {})

        if process is not None:
            self.process = process
        elif self.model == "gbm":
            self.process = GBMProcess(
                mu=float(params.pop("mu")),
                sigma=float(params.pop("sigma")),
            )
        elif self.model in {"cir", "cox", "mean_reverting", "mean-reverting"}:
            self.process = CIRProcess(
                kappa=float(params.pop("kappa", 0.2)),
                theta=float(params.pop("theta", V0)),
                sigma=float(params.pop("sigma")),
            )
        elif self.model in {"gbm_jump", "jump_gbm", "gbm-jump", "jump-diffusion"}:
            self.process = GBMJumpProcess(
                mu=float(params.pop("mu")),
                sigma=float(params.pop("sigma")),
                lambda_pos=float(params.pop("lambda_pos", 0.0)),
                lambda_neg=float(params.pop("lambda_neg", 0.0)),
                jump_mean_pos=float(params.pop("jump_mean_pos", 0.0)),
                jump_std_pos=float(params.pop("jump_std_pos", 0.0)),
                jump_mean_neg=float(params.pop("jump_mean_neg", 0.0)),
                jump_std_neg=float(params.pop("jump_std_neg", 0.0)),
            )
        elif self.model in {"gbm_transient_jump", "transient_jump", "transient-jump"}:
            self.process = GBMTransientJumpProcess(
                mu=float(params.pop("mu")),
                sigma=float(params.pop("sigma")),
                reversion=float(params.pop("reversion", 0.5)),
                lambda_pos=float(params.pop("lambda_pos", 0.0)),
                lambda_neg=float(params.pop("lambda_neg", 0.0)),
                jump_mean_pos=float(params.pop("jump_mean_pos", 0.0)),
                jump_std_pos=float(params.pop("jump_std_pos", 0.0)),
                jump_mean_neg=float(params.pop("jump_mean_neg", 0.0)),
                jump_std_neg=float(params.pop("jump_std_neg", 0.0)),
            )
        else:
            raise ValueError(
                f"Unsupported model '{model}'. Use 'gbm', 'cir', 'gbm_jump', or 'gbm_transient_jump'."
            )

        if params:
            raise ValueError(f"Unknown model_params keys for model '{self.model}': {sorted(params.keys())}")

    def simulate_VT(self, n_days: int, n_sims: int = 200_000):
        return self.process.simulate_terminal(
            self.V0,
            n_days,
            n_sims,
            self.rng,
            step_size=self.step_size,
        )

    def _step_grid(self, n_days: float, step_size: float = None) -> list[float]:
        step_size = self.step_size if step_size is None else float(step_size)
        horizon = float(n_days)
        if horizon < 0:
            raise ValueError("n_days must be >= 0")

        steps = []
        remaining = horizon
        while remaining > 1e-12:
            step = min(step_size, remaining)
            steps.append(step)
            remaining -= step
        return steps

    def simulate_path_with_timestamps(self, n_days: int, start_time: datetime = None, step_size: float = None):
        step_size = self.step_size if step_size is None else float(step_size)
        if n_days < 0:
            raise ValueError("n_days must be >= 0")

        if not hasattr(self.process, "simulate_path"):
            raise ValueError("Selected process does not implement simulate_path")

        start = start_time or datetime.utcnow()
        values = self.process.simulate_path(
            self.V0,
            n_days,
            self.rng,
            step_size=step_size,
        )
        elapsed = 0.0
        timestamps = [start]
        for step in self._step_grid(n_days, step_size=step_size):
            elapsed += step
            timestamps.append(start + timedelta(days=elapsed))
        
        plt.figure(figsize=(20, 3.5))
        
        plt.plot(timestamps, values, linewidth=3, color="tab:blue")
        plt.title("one simulated path", fontsize=14)
        plt.ylabel("asset price", fontsize=14)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()

        return {
            "timestamps": timestamps,
            "values": values,
        }

    def liquidity_value(self, X: np.ndarray) -> float:
        return cara_liquidity_value(X, self.gamma)

    def liquidity_of_waiting(self, n_days: int, n_sims: int = 200_000):
        VT = self.simulate_VT(n_days, n_sims)
        X_wait = (1.0 - self.fee) * VT
        return self.liquidity_value(X_wait)

    def sell_value(self, slippage: float) -> float:
        return (1.0 - slippage) * self.V0

    def decide(self, n_days: int, slippage: float, n_sims: int = 200_000):
        L_wait = self.liquidity_of_waiting(n_days, n_sims)
        X_sell = self.sell_value(slippage)

        decision = "HOLD" if L_wait >= X_sell else "SELL"

        return {
            "decision": decision,
            "L_wait": L_wait,
            "X_sell": X_sell,
            "indifference_haircut": 1 - L_wait / self.V0,
        }

    def decision_gap(
        self,
        n_days: int,
        slippage: float,
        n_sims: int = 200_000,
    ) -> float:
        decision = self.decide(n_days=n_days, slippage=slippage, n_sims=n_sims)
        return float(decision["L_wait"] - decision["X_sell"])

    def decision_gap_confidence_interval(
        self,
        n_days: int,
        slippage: float,
        n_sims: int = 200_000,
        n_bootstrap: int = 500,
        confidence_level: float = 0.95,
        bootstrap_seed: int = None,
    ):
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be > 0")

        VT = self.simulate_VT(n_days, n_sims)
        X_wait = (1.0 - self.fee) * VT
        X_sell = self.sell_value(slippage)
        gap_estimate = self.liquidity_value(X_wait) - X_sell

        bootstrap_rng = (
            np.random.default_rng(bootstrap_seed)
            if bootstrap_seed is not None
            else self.rng
        )
        bootstrap_gaps = np.empty(n_bootstrap, dtype=float)
        sample_size = len(X_wait)

        for i in range(n_bootstrap):
            sample_idx = bootstrap_rng.integers(0, sample_size, size=sample_size)
            sample = X_wait[sample_idx]
            bootstrap_gaps[i] = self.liquidity_value(sample) - X_sell

        alpha = (1.0 - confidence_level) / 2.0
        ci_low, ci_high = np.quantile(bootstrap_gaps, [alpha, 1.0 - alpha])

        if ci_low > 0:
            decision = "HOLD"
        elif ci_high < 0:
            decision = "SELL"
        else:
            decision = "INCONCLUSIVE"

        return {
            "decision": decision,
            "gap_estimate": float(gap_estimate),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "confidence_level": float(confidence_level),
            "X_sell": float(X_sell),
            "L_wait": float(gap_estimate + X_sell),
            "bootstrap_std": float(np.std(bootstrap_gaps, ddof=1))
            if n_bootstrap > 1
            else 0.0,
        }

    def diagnostics(self, n_days: int, slippage: float, n_sims: int = 200_000):
        VT = self.simulate_VT(n_days, n_sims)
        X_wait = (1.0 - self.fee) * VT
        X_sell = self.sell_value(slippage)

        prob_wait_worse = np.mean(X_wait < X_sell)
        expected_diff = np.mean(X_wait - X_sell)

        return {
            "P(wait < sell_now)": float(prob_wait_worse),
            "E(wait - sell_now)": float(expected_diff),
        }
