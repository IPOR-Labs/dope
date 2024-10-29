import numpy as np
import pandas as pd
from dataclasses import dataclass

import numpy as np
import cvxopt as opt
from cvxopt.solvers import qp as cvxopt_qp
from cvxopt.solvers import options as cvxopt_options
import sympy

from dope.backengine.agents.base import BaseAgent


@dataclass
class LenderQuadraticParams:
    columns: list
    sigmas: np.array
    mus: np.array
    cov: np.array
    inv_cov: np.array
    corr_matrix: np.array
    iota: np.array
    c_deposit: np.array

    def __iter__(self):
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                yield getattr(self, attr)


class LenderQuadratic(BaseAgent):

    def __init__(self, token, capital, risk_aversion=1, mean_window=1, cov_window=7):
        self.token = token
        self.capital = capital
        self.engine = None
        assert risk_aversion >= 0, "Risk aversion must be non negative"
        self.risk_aversion = risk_aversion
        self.mean_window = int(mean_window)
        self.cov_window = int(cov_window)

        self.rate_column = "apyBase"

    def _get_optimizer_params(self, date_ix):
        rate_column = "apyBase"
        df = self.data.to_block(self.token)[rate_column]
        # pd.concat(self.data[self.token], names=["datetime"]).unstack(level=0)[
        #     rate_column
        # ]
        # del df["cash"]
        df = df[df.index <= date_ix]

        mean_filter = df.index >= date_ix - pd.Timedelta(f"{self.mean_window}D")
        cov_filter = df.index >= date_ix - pd.Timedelta(f"{self.cov_window}D")
        if len(df) == 0:
            return None
        
        mus = mus = df[mean_filter].mean()
        valid_cols = list(mus[np.isfinite(mus)].index)
        df = df[valid_cols]

        sigma = df[cov_filter].std()
        valid_cols = list(sigma[np.isfinite(sigma)].index)
        df = df[valid_cols]
        
        sigma = np.array(df[cov_filter].std()).reshape(-1, 1)
        mus = np.array(df[mean_filter].mean()).reshape(-1, 1)
        cov = np.array(df[cov_filter].cov())
        inv_cov = cov  # np.linalg.inv(cov)
        corr_matrix = cov  # np.matrix(df[cov_filter].corr())

        # deposit costs
        c_deposit = self._get_depth_cost(date_ix, df.columns)

        iota = np.array(np.ones((len(cov), 1)))  # len(cov) = 8

        return LenderQuadraticParams(
            df.columns, sigma, mus, cov, inv_cov, corr_matrix, iota, c_deposit
        )

    def _get_depth_cost(self, date_ix, valid_protocols=None):
        deposit_cost = []
        if valid_protocols is None:
            valid_protocols = list(self.data[self.token].keys())
        for protocol, df in self.data[self.token].items():
            if protocol not in valid_protocols:
                continue
            # if protocol == "cash":
            #   continue
            # extra interest rate per percentage point of utilization rate
            _filter = df.index <= date_ix
            if len(df[_filter]) == 0:
                continue

            impact_1 = self.engine.mkt_impact[protocol].impact(
                date_ix, 0, is_borrow=False
            )
            impact_2 = self.engine.mkt_impact[protocol].impact(
                date_ix, self.capital / 2, is_borrow=False
            )
            slope = (impact_2 - impact_1) / (self.capital / 2)
            if protocol == "cash":
                c = 0
            else:
                c = (
                    self.capital
                    / df[_filter][self.rate_column].resample("1D").mean().iloc[-1]
                    * slope
                )
            deposit_cost.append(c)

        return np.array(deposit_cost).reshape(-1, 1)
        # return np.ones(n) *  deposit_cost

    def optimum_allocation(self, params, prec=None):
        """
        Notes about this functions:
        https://github.com/IPOR-Labs/latex-papers/blob/main/yield-optimization/main.pdf
        """

        n = len(params.sigmas)
        max_weight = 1
        opt.solvers.options["show_progress"] = False

        # deposit less money, more rates
        # cplus = ...
        # deposit more money, lower rates
        # cminus = params.c_deposit

        max_weight = 1
        r = opt.matrix(np.block([params.mus - params.c_deposit]))

        #print("r", r)

        A = opt.matrix(
            np.block(
                [
                    [np.ones(n)],
                    # [mus.flatten()],
                ]
            )
        )
        B = opt.matrix(
            np.block(
                [
                    1.0,
                    # np.array(mus[0])[0][0]
                ]
            )
        )

        Q = opt.matrix(
            np.block(
                [
                    params.cov,
                ]
            )
        )
        # sympy.Matrix(np.matrix(Q))

        # Create constraint matrices
        G = opt.matrix(np.block([[-np.eye(n)], [np.eye(n)]]))
        h = opt.matrix(
            np.block(
                [
                    [np.zeros(n), max_weight * np.ones(n)],
                ]
            ).T
        )

        sol = opt.solvers.qp(self.risk_aversion / 2 * Q, -r, G, h, A, B)
        # print(sympy.Matrix(np.matrix(sol["x"])))
        # print(sol["x"])
        solution = sol["x"]
        if prec is not None:
            solution = np.round(solution, prec)

        return solution

    def on_act(self, date_ix):
        """
        date_ix is the date index NOW.
        One can think as the index of the filtration $\\mathcal{F}_{ix}$, i.e.,
        the increasing sequence of information sets where the agent acts.

        """
        opt_params = self._get_optimizer_params(date_ix)
        self.opt_params = opt_params
        ws_list = self.optimum_allocation(opt_params, prec=None)
        self.ws_list = ws_list
        # print("ws:", ws_list)
        ws = {self.opt_params.columns[i]: abs(ws_list[i]) for i in range(len(ws_list))}

        return {self.token: ws}
