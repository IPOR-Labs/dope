from dope.backengine.agents.base import BaseAgent

class LenderJackReaper(BaseAgent):

  def __init__(self, token, capital, parts=1, days_window=7, triggers=None):
    self.token = token
    self.steps = parts
    self.capital = capital
    self.days_window = days_window
    self.triggers = triggers
    self.engine = None
    self.ws = {}

  def act(self, date_ix):
    """
    date_ix is the date index NOW. 
    One can think as the index of the filtration \mathcal{F}_{ix}, i.e.,
    the increasing sequence of information sets where the agent is acts.
    
    """
    if self.triggers is not None:
      if date_ix not in self.triggers:
        return self.ws
    print("Acting....", date_ix)

    ws = {mkt:0 for mkt in self.data[self.token].keys()}
    steps = self.steps
    for _ in range(steps):
      values = {}
      for protocol, df in self.data[self.token].items():
        # extra interest rate per percentage point of utilization rate
        _filter = df.index <= date_ix
        if len(df[_filter]) == 0:
          continue
        #print("protocol", protocol)
        impact = self.engine.mkt_impact[protocol].impact(
          date_ix,
          self.capital * (1/steps + ws[protocol]),
          is_borrow=False
        )

        values[protocol] = (df[_filter]["apyBase"]+impact).rolling(f"{self.days_window}D").mean().iloc[-1]
        # get market with higest returns
      mkt = max(values, key=values.get)
      #print()
      #rint(df[_filter].iloc[0])
      #print(df[_filter].iloc[-1])
      #print(date_ix, mkt, values)
      ws[mkt] += 1/steps
    self.ws = {self.token:ws}
    # print(self.ws)
    return self.ws
