import pandas as pd
from dope.backengine.estimators.baseestimator import BaseEstimator

class Trigger:
  def __init__(self, df, lag, est, protocolList=[]):
    self.df = df
    self.lag = lag
    self.est = est
    self.protocolList = protocolList
    
  def pairs(self):
    protocols = [x[1] for x in list(self.df.columns)]
    if self.protocolList!=[]:
      protocols = [v for v in protocols if v in self.protocolList]      
    N = len(protocols)
    return [(protocols[i],protocols[j]) for i in range(0,N) for j in range(i+1,N)]
    
  def dates(self):
    mu = self.est.rolling_fit_mu(df=self.df, lag=self.lag, rt_col="apyBaseBorrow") # could equivalently use 'supplyRate' or borrowRate'
    sigma = self.est.rolling_fit_sigma(df=self.df, lag=self.lag, rt_col="apyBaseBorrow") # identify which column to use for calculating triggers
    k = 0.5
    t_dn = (mu - k*sigma).dropna()
    t_up = (mu + k*sigma).dropna()
    _filter = pd.Series(index=pd.Index([],dtype=int), dtype=bool) # initialise to empty
    for pair in self.pairs():
      asset1, asset2 = pair[0], pair[1]
      t = t_dn.copy().rename(columns={asset1: asset1+'_dn', asset2: asset2+'_dn'})
      t[asset1+'_up'] = t_up[asset1]
      t[asset2+'_up'] = t_up[asset2]
      t['diff_1']=t[asset1+'_up']-t[asset2+'_dn']
      t['diff_2']=t[asset1+'_dn']-t[asset2+'_up']
      _filter1 = t['diff_1']>0
      _filter1 &= t['diff_1'].shift(1)<0
      _filter2 = t['diff_2']>0
      _filter2 &= t['diff_2'].shift(1)<0
      _filter12 = (_filter1 | _filter2)
      _filter = _filter12 if _filter.empty else _filter | _filter12
    return list(t[_filter].index)
