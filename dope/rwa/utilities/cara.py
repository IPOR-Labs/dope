import numpy as np


def cara_liquidity_value(X: np.ndarray, gamma: float) -> float:
    """
    CARA certainty equivalent:

    L(X) = -(1/gamma) * log(E[exp(-gamma X)])
    """
    z = -gamma * X
    z_max = np.max(z) # avoid exploding exponential
    log_mean_exp = z_max + np.log(np.mean(np.exp(z - z_max)))
    return float(-(1.0 / gamma) * log_mean_exp)


class ImpliedGamma:
    """
    Solve for gamma in a two-point return gamble under CARA utility.

    V_T = V0*(1 + r)  with probability p
    V_T = V0*(1 - l)  with probability (1-p)

    We solve for gamma such that:

        E[V_T] - L(V_T; gamma) = target_premium

    where:
        L(X) = -(1/gamma) * log( E[exp(-gamma X)] )
    """

    def __init__(self, V0: float):
        self.V0 = float(V0)

    def solve_gamma(
        self,
        r: float,
        l: float,
        target_premium: float,
        p: float = 0.5,
        gamma_lo: float = 1e-8,
        gamma_hi: float = 1e4,
        tol: float = 1e-10,
        max_iter: int = 200,
    ) -> float:
        up = self.V0 * (1.0 + r)
        down = self.V0 * (1.0 - l)
        EV = p * up + (1 - p) * down

        def premium_at_gamma(g):
            z1 = -g * up
            z2 = -g * down
            zmax = max(z1, z2)

            log_mix = zmax + np.log(
                p * np.exp(z1 - zmax) + (1 - p) * np.exp(z2 - zmax)
            )

            L = -(1.0 / g) * log_mix
            return EV - L

        if premium_at_gamma(gamma_lo) > target_premium:
            return gamma_lo

        if premium_at_gamma(gamma_hi) < target_premium:
            raise ValueError("Increase gamma_hi — solution not bracketed.")

        lo, hi = gamma_lo, gamma_hi

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            prem = premium_at_gamma(mid)

            if abs(prem - target_premium) < tol:
                return mid

            if prem < target_premium:
                lo = mid
            else:
                hi = mid

        return 0.5 * (lo + hi)
