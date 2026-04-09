from dope.rwa.stochprocess import GBMProcess, CIRProcess, GBMJumpProcess, GBMTransientJumpProcess
from dope.rwa.pricing import RWA
from dope.rwa.utilities import ImpliedGamma, cara_liquidity_value, crra_utility, crra_certainty_equivalent

__all__ = [
    "ImpliedGamma",
    "GBMProcess",
    "CIRProcess",
    "GBMJumpProcess",
    "GBMTransientJumpProcess",
    "RWA",
    "cara_liquidity_value",
    "crra_utility",
    "crra_certainty_equivalent",
]


if __name__ == "__main__":
    rwa = RWA(
        V0=1.07,
        gamma=50.0,
        fee=0.0,
        model="gbm",
        model_params={"sigma": 0.0015, "mu": 0.0},
    )

    rwa_cir = RWA(
        V0=1.07,
        gamma=50.0,
        fee=0.0,
        model="cir",
        model_params={"kappa": 0.3, "theta": 1.07, "sigma": 0.01},
    )

    result = rwa.decide(n_days=30, slippage=0.005)
    diag = rwa.diagnostics(n_days=30, slippage=0.005)

    print("Decision:", result["decision"])
    print("Liquidity value (wait):", result["L_wait"])
    print("Sell value:", result["X_sell"])
    print("Indifference haircut:", result["indifference_haircut"])
    print("Diagnostics:", diag)
    print("CIR Decision:", rwa_cir.decide(n_days=30, slippage=0.005)["decision"])
