import numpy as np

def cvar_beta(losses, probs=None, beta=0.95, alpha=0.0):
    """
    Rockafellar-Uryasev scenario function:
        F~_beta(x, alpha) = alpha + 1/(1-beta) * E[(L - alpha)^+]

    Finite-scenario form:
        - equal-probability scenarios:
            alpha + (1 / (q * (1 - beta))) * sum_k max(loss_k - alpha, 0)
        - user-specified probabilities:
            alpha + (1 / (1 - beta)) * sum_k p_k * max(loss_k - alpha, 0)

    Parameters
    ----------
    losses : array-like
        Scenario losses f(x, y_k).
    probs : array-like or None
        Scenario probabilities p_k.
        If None, assumes equal probabilities 1/q.
    beta : float
        Confidence level, e.g. 0.95.
    alpha : float
        Candidate VaR level.

    Returns
    -------
    float
        Value of F~_beta(x, alpha).
    """
    losses = np.asarray(losses, dtype=float)

    if not (0 < beta < 1):
        raise ValueError("beta must be between 0 and 1")

    excess = np.maximum(losses - alpha, 0.0)

    if probs is None:
        q = len(losses)
        if q == 0:
            raise ValueError("losses must not be empty")
        return alpha + excess.sum() / (q * (1 - beta))

    probs = np.asarray(probs, dtype=float)
    if losses.shape != probs.shape:
        raise ValueError("losses and probs must have the same length")
    if np.any(probs < 0):
        raise ValueError("probabilities must be nonnegative")

    total = probs.sum()
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    probs = probs / total

    return alpha + np.sum(probs * excess) / (1 - beta)