import pandas as pd



class TokenPortfolio:
  def __init__(self):
    self.allocation = {}
  
  def __len__(self):
    return len(self.allocation)
  
  def compound(self, protocol_rate_dict, dt):
    for protocol, rate in protocol_rate_dict.items():
      if protocol in self.allocation:
        self.allocation[protocol] = self.allocation[protocol] * (1 + rate/100*dt)
  
  def capital(self):
    # #print("@Capital()")
    # print(self.allocation)
    # print(self.allocation.values())
    # print(sum(self.allocation.values()))
    # print("Done with capital()")
    return sum(self.allocation.values())
  
  def weights(self):
    total = self.capital()
    return {protocol: capital/total for protocol, capital in self.allocation.items()}
  
  def rebalance(self, new_weights, capital=None):
    _capital = capital or self.capital()
    #print(f"{_capital = :,} ")
    
    for protocol, weight in new_weights.items():
      self.allocation[protocol] = weight * _capital

class ArbBacktester:

  def __init__(self, strategy, borrow_lend_data, data, mkt_impact, tokens=None):
    self.strategy = strategy
    self.data = data
    self.borrow_lend_data = borrow_lend_data
    self.mkt_impact = mkt_impact
    self.dates = self.get_dates()
    self.summary = None
    
    self.tokens = tokens or list(self.data.keys())
    self.πs = {token:TokenPortfolio() for token in self.tokens}

  def get_dates(self):
    dates = set()
    for token, data in self.data.items():
      for protocol, df in data.items():
        dates.update(df.index)
    dates = list(dates)
    dates.sort()
    return dates

  def get_capital(self, token):
    return self.πs[token].capital()

  def __call__(self):
    end_timestamp = pd.to_datetime("2023-12-08")
    start_timestamp = pd.to_datetime("2023-12-07")
    self.strategy.register_engine(self)
    rows = []
    Ws = []
    _days = len(self.dates)
    print(f"Running Backtest for {_days:,} | token:{self.strategy.token }")
    _tenpct = _days//10
    for i in range(0, len(self.dates[:])-1):
      if i % _tenpct == 0:
        print(f"{i:>10,}/{_days:>10,}", end="\r")
      date_prev = self.dates[i]
      date_now = self.dates[i+1]
      
      # if date_now < start_timestamp:
      #   continue
      #if date_now >= end_timestamp:
      # break
      #print(date_now)
      # The agent does not know the future.
      #print(date_now)
      
      # Step 1: Position gets Accrued:
      for token, π in self.πs.items():
        r_breakdown = {}
        slopes = {}
        impacts = {}
        if len(π) > 0:
          ws_before = π.weights()
        
        for mkt, capital in π.allocation.items():
          if mkt not in self.borrow_lend_data[token]:
            continue
          df = self.borrow_lend_data[token][mkt]
          _filter = df.index <= date_now
          if len(df[_filter]) == 0:
            continue
          sign = 1 if capital >= 0 else -1
          side_name = "apyBase" if capital >= 0	else "apyBaseBorrow"
          if mkt == "cash":
            assert side_name != "apyBaseBorrow", "Cannot Borrow from own wallet."
          impacts[mkt] = self.mkt_impact[mkt].impact(date_now, capital, is_borrow=False)
          r_breakdown[mkt] = max(0, df[_filter][side_name].iloc[-1] + sign * impacts[mkt])


        π.compound(r_breakdown, dt=(date_now - date_prev).total_seconds()/365/24/3600)
        for mkt in r_breakdown.keys():
          r_breakdown[mkt] = ws_before.get(mkt, 0) * r_breakdown[mkt]
        if len(π)>0:
          timestamp = df[_filter].timestamp.iloc[-1]
          rows.append([date_now, timestamp, {token:π.weights()}, sum(r_breakdown.values()), π.capital(), r_breakdown, slopes, impacts])
          Ws.append([date_now, timestamp, self.strategy.token, {token:ws_before}, π.capital()])
    
      # Step 2: Strategy Acts
      ws = self.strategy.act(date_now)
      #print("Ws::::",ws)
      
      # step 3: Rebalance
      for token, _ws in ws.items():
        if len(self.πs[token]) ==0:
          self.πs[token].rebalance(_ws, self.strategy.capital)
        else:
          self.πs[token].rebalance(_ws, None)
      
      #print("π:::::",π.allocation)
    strategy = pd.DataFrame(rows, columns=["datetime", "timestamp", "ws", "rate", "capital", "breakdown", "slope", "mkt_impact"])
    strategy = strategy.set_index("datetime")
    self.summary = strategy
    self.Ws = pd.DataFrame(Ws, columns=["datetime", "timestamp", "token", "ws", "capital"])
    return self.summary, self.Ws
