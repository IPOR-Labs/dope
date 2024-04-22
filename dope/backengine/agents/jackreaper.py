from dope.backengine.agents.base import BaseAgent

class LenderJackReaper(BaseAgent):

  def __init__(self, capital, parts=1):
    self.steps = parts
    self.capital = capital
    self.engine = None
  
  def act(self, date_ix):
    
    ws = {mkt:0 for mkt in self.data.keys()}
    steps = self.steps
    for _ in range(steps):
      values = {}
      for protocol, df in self.data.items():
        # extra interest rate per percentage point of utilization rate
        _filter = df.index <= date_ix
        if len(df[_filter]) == 0:
          continue
        slope = self.engine.mkt_impact[protocol].get_slope(df[_filter].utilizationRate.iloc[-1])
        impact = self.capital/df[_filter].tvlUsd * slope * (1/steps + ws[protocol])

        values[protocol] = (df[_filter]["apy"]-impact).rolling("7D").mean().iloc[-1]
        # get market with higest returns
      mkt = max(values, key=values.get)
      ws[mkt] += 1/steps
    return ws
