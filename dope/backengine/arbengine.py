import pandas as pd


class ArbBacktester:

  def __init__(self, strategy, data, mkt_impact):
    self.strategy = strategy
    self.data = data
    self.mkt_impact = mkt_impact
    self.dates = self.get_dates()
    self.summary = None

  def get_dates(self):
    dates = set()
    for token, data in self.data.items():
      for protocol, df in data.items():
        dates.update(df.index)
    dates = list(dates)
    dates.sort()
    return dates
  
  def __call__(self):
    end_timestamp = pd.to_datetime("2023-07-23")
    start_timestamp = pd.to_datetime("2024-01-01")
    self.strategy.register_engine(self)
    rows = []
    for i in range(20, len(self.dates[:])-1):
      date_prev = self.dates[i]
      date_now = self.dates[i+1]
      if date_now < start_timestamp:
        continue
      #if date_now >= end_timestamp:
      #  break
      #print(date_now)
      
      ws = self.strategy.act(date_prev)
      #rate = 0
      r_breakdown = {}
      slopes = {}
      impacts = {}
      for token, mkt_ws in ws.items():
        for mkt, w in mkt_ws.items():
          #mkt = f"{mkt}_{side}"
          df = self.data[token][mkt]
          _filter = df.index < date_now
          if len(df[_filter]) == 0:
            continue
          #print(mkt, w)
          sign = 1 if w > 0 else -1
          side_name = "apyBase" if w > 0	else "apyBaseBorrow"
          slopes[mkt] = self.mkt_impact[mkt].get_slope(df[_filter].utilizationRate.iloc[-1]) 
          impacts[mkt] = self.strategy.capital/df[_filter].totalSupplyUsd.iloc[-1] * slopes[mkt]

          #mkt_impact = lender.capital/df[_filter].tvlUsd.iloc[-1] * slope
          r_breakdown[mkt] = w * max(0, df[_filter][side_name].iloc[-1] - sign * impacts[mkt])
        #rate += r_breakdown[mkt]
        #print(date_now, w, mkt, mkt_impact, (w*(df[_filter]["apy"].iloc[-1] - mkt_impact)))
      #print()
      #mkt, rate
      #break
      rows.append([date_now, ws, sum(r_breakdown.values()), r_breakdown, slopes, impacts])
    strategy = pd.DataFrame(rows, columns=["datetime", "ws", "rate", "breakdown", "slope", "mkt_impact"])
    strategy = strategy.set_index("datetime")
    self.summary = strategy
    return self.summary
