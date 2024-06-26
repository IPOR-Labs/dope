from dope.backengine.agents.base import BaseAgent

from dope.price_model.mvga import MavgPredictor

class LenderJackReaper(BaseAgent):

  def __init__(self, token, capital, parts=1, days_window=7, triggers=None, predictor=None, verbose=True):
    self.token = token
    self.steps = parts
    self.capital = capital
    self.days_window = days_window
    self.triggers = triggers
    self.engine = None
    self.ws = {}
    self.verbose = verbose
    
    _predictor = predictor or MavgPredictor(days_window=days_window)
    
    self.register_price_predictor(_predictor)
    
  
  def register_price_predictor(self, price_predictor):
    price_predictor.register_agent(self)
    self.predictor = price_predictor

  def act(self, date_ix):
    """
    date_ix is the date index NOW. 
    One can think as the index of the filtration \mathcal{F}_{ix}, i.e.,
    the increasing sequence of information sets where the agent is acts.
    
    """
    if self.triggers is not None:
      if date_ix not in self.triggers:
        return self.ws
    if self.verbose:
      print("Acting....", date_ix)
    #print("CAPITAL", self.capital, self.days_window)

    ws = {mkt:0 for mkt in self.data[self.token].keys()}
    steps = self.steps
    CAP =self.engine.get_capital(token=self.token)
    import pandas as pd
    
    expected_returns_dict = self.predictor.exp_returns(date_ix, self.token)
    
    for _ in range(steps):
      values = {}
      for protocol, Er in expected_returns_dict.items():
        impact = self.engine.mkt_impact[protocol].impact(
          date_ix,
          CAP * (1/steps + ws[protocol]),
          is_borrow=False
        )
        if impact is None:
          continue
        #print(impact)
        values[protocol] = Er + impact
        # get market with higest returns
      mkt = max(values, key=values.get)
      #print()
      #rint(df[_filter].iloc[0])
      #print(df[_filter].iloc[-1])
      #
      ws[mkt] += 1/steps
      #print(values)
      #for k, v in values.items(): print(k, v)
      #print(ws)
    self.ws = {self.token:ws}
    # print(self.ws)
    return self.ws
