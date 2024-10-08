import numpy as np

from dope.backengine.agents.base import BaseAgent

from dope.price_model.mvga import MavgPredictor


class Idler(BaseAgent):

    def __init__(self, *args, **kwargs):
        self.token = "NO-TOKEN"

        self.ws = {
            "Ethereum:lido:STETH": 10.151175624436906,
            "Ethereum:gearbox:WETH": -9.151175624436906,
        }
        self.verbose = False

    def on_start(self):
        return self.ws

    def on_liquidation(self):
        return {}

    def on_act(self, date_ix):
        """
        date_ix is the date index NOW.
        One can think as the index of the filtration \mathcal{F}_{ix}, i.e.,
        the increasing sequence of information sets where the agent is acts.

        """
        if self.verbose:
            print("Acting....", date_ix)
        return self.ws
