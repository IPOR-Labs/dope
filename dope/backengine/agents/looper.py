import numpy as np

from dope.backengine.agents.base import BaseAgent
from dope.names.poolname import PoolName


class Looper(BaseAgent):

    def __init__(self, capital, loop_n, pair_pool, deposit_pool, *args, **kwargs):
        super().__init__(capital=capital, *args, **kwargs)
        self.token = "NO-TOKEN"

        self.pair_pool = pair_pool
        self.deposit_pool = deposit_pool
        self.loop_n = loop_n

        self.verbose = False

    def on_start(self, date_ix):
        self.engine.do_loop(
            date_ix,
            self.capital,
            pair_pool=self.pair_pool,
            deposit_pool=self.deposit_pool,
            loop_times=self.loop_n
        )
        #self.engine.π.add_deposit(self.deposit_pool, self.supply_cap)
        #price_now = self.engine.price_data.price_row_at(date_ix)
        #self.ws = self.engine.π.weights(price_now)
        
        #print()
        return 

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
