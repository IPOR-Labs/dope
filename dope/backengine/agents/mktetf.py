import numpy as np
import pandas as pd
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

import numpy as np
import cvxopt as opt
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
    capital: float

    def __iter__(self):
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                yield getattr(self, attr)


class LenderMKTETF(BaseAgent):

    def __init__(self, token, capital, time_window=7, triggers=None, verbose=True):
        """
        Mixed Integer Quadratic Programming Agent
        """
        self.token = token
        self.capital = capital
        self.engine = None

        self.time_window = int(time_window)
        self.triggers = triggers
        self.ws = {}
        self.verbose = verbose

        self.rate_column = "apyBase"

    def on_act(self, date_ix):
        """
        date_ix is the date index NOW.
        One can think as the index of the filtration \mathcal{F}_{ix}, i.e.,
        the increasing sequence of information sets where the agent acts.

        """
        if self.triggers is not None:
            if date_ix not in self.triggers:
                return self.ws
        if self.verbose:
            print("Acting....", date_ix)

        run_data = self.engine.data

        tmp = run_data.to_block(self.token)
        tmp = tmp[tmp.index <= date_ix]

        columns = tmp.totalSupplyUsd.columns
        columns = [c for c in columns if "cash" not in c]
        tmp = tmp.totalSupplyUsd[columns].rolling(f"{self.time_window}D").mean()
        _sum = tmp.sum(axis=1)
        # print(_sum.iloc[-1], "<<< SUM", "tmp.iloc[-1]", tmp.iloc[-1])
        self.ws = {
            self.token: {
                k: v
                for k, v in zip(columns, (tmp.iloc[-1] / _sum.iloc[-1]))
                if np.isfinite(v)
            }
        }
        return self.ws
