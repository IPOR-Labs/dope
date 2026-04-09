import numpy as np


def crra_utility(x: np.ndarray, rho: float) -> np.ndarray:
    """
    CRRA utility on positive wealth x.

    For rho != 1: u(x) = x^(1-rho) / (1-rho)
    For rho == 1: u(x) = log(x)
    """
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    if np.isclose(rho, 1.0):
        return np.log(x)
    return np.power(x, 1.0 - rho) / (1.0 - rho)


def crra_certainty_equivalent(x: np.ndarray, rho: float) -> float:
    """
    Certainty equivalent under CRRA utility.
    """
    u_mean = float(np.mean(crra_utility(x, rho)))
    if np.isclose(rho, 1.0):
        return float(np.exp(u_mean))
    return float(np.power(np.maximum((1.0 - rho) * u_mean, 1e-12), 1.0 / (1.0 - rho)))
