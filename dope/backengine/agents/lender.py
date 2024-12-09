import numpy as np
import pandas as pd

from dope.backengine.agents.base import BaseAgent


class SimpleLender(BaseAgent):

    def __init__(
        self,
        token,
        capital,
        parts=1,
        days_window=7,
        triggers=None,
        verbose=True,
    ):
        self.token = token
        self.steps = parts
        self.capital = capital
        self.days_window = days_window
        self.triggers = triggers
        self.engine = None
        self.ws = {}
        self.verbose = verbose
    
    def on_start(self, date_ix):
        self.engine.deposit_to_wallet(self.token, self.capital)

    def on_act(self, date_ix):
        """
        date_ix is the date index NOW.
        One can think as the index of the filtration $\\mathcal{F}_{ix}$, i.e.,
        the increasing sequence of information sets where the agent is acts.

        """
        if self.triggers is not None:
            if date_ix not in self.triggers:
                return self.ws
        if self.verbose:
            print("Acting....", date_ix)

        ws = {mkt: 0 for mkt in self.data[self.token].keys()}

        df = self.data.as_block(self.token)
        df = df[df.index <= date_ix]
        
        fraction = 1/5 
        supply_borrow_delta = (df.totalSupplyUsd - df.totalBorrowUsd).iloc[-1]
        sorted_expected_returns = (
            df.apyBase
            .rolling(window=self.days_window)
            .mean().iloc[-1]
            .sort_values(ascending=False)
        )

        allocations = {}
        init_capital = self.engine.get_capital(token=self.token)
        remaining_capital = init_capital

        for protocol, expected_return in sorted_expected_returns.items():
            if protocol == "cash":
                continue
            if remaining_capital <= 0:
                break
            if supply_borrow_delta[protocol] > 0:
                allocations[protocol] = min(
                    remaining_capital,
                    supply_borrow_delta[protocol] * fraction
                )
                remaining_capital -= allocations[protocol]
        
        total_allocations = sum(allocations.values())
        cash_allocation = init_capital - total_allocations
        if cash_allocation > 0:
            allocations["cash"] = cash_allocation

        # Needs to return a dictionary of weights
        ws = {k: v / init_capital for k, v in allocations.items()}
        ws = {k: w for k, w in ws.items() if w != 0}

        self.ws = {self.token: ws}
        return self.ws
