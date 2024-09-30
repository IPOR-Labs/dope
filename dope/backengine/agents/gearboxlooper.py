import numpy as np

from dope.backengine.agents.base import BaseAgent


class GearBoxLooper(BaseAgent):

    def __init__(self, leverage, capital, debt_pool, deposit_pool, *args, **kwargs):
        super().__init__(capital=capital, *args, **kwargs)
        self.token = "NO-TOKEN"

        self.leverage = leverage
        self.debt_pool = debt_pool
        self.deposit_pool = deposit_pool

        self.debt_cap = self.leverage * self.capital - self.capital
        self.supply_cap = self.capital * leverage

        self.verbose = False

    def on_start(self, date_ix):
        self.engine.π.add_debt(self.debt_pool, self.debt_cap)
        self.engine.π.add_deposit(self.deposit_pool, self.supply_cap)
        price_now = self.engine.price_data.price_row_at(date_ix)
        self.ws = self.engine.π.weights(price_now)
        print(self.ws)
        return self.ws

    def on_liquidation(self, date_ix):
        return {}

    def on_act(self, date_ix):
        """
        date_ix is the date index NOW.
        One can think as the index of the filtration $\\mathcal{F}_{ix}$, i.e.,
        the increasing sequence of information sets where the agent is acts.

        """
        if self.verbose:
            print("Acting....", date_ix)
        return self.ws
