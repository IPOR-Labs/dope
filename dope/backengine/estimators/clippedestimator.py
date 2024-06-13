from dope.backengine.estimators.baseestimator import BaseEstimator
import numpy as np

class ClippedEstimator(BaseEstimator):
  def __init__(self, quantile):
    self.quantile = quantile
  
  def rolling_fit_mu(self, df, lag, rt_col):
    """
    df: Dataframe with index as datetime
    lag: Rolling window
    rt_col: column with the rates data to fit
    """
    
    df = df[rt_col]
    
    quantiles = df.quantile(self.quantile)
    df_clipped = df.clip(upper=quantiles, axis=1) # column-wise
    rolling = df_clipped.rolling(lag)
    k = np.sqrt(2)

    mu = rolling.mean()
    
    return mu
  
  def rolling_fit_sigma(self, df, lag, rt_col):
    """
    df: Dataframe with index as datetime
    lag: Rolling window
    rt_col: column with the rates data to fit
    """
    df = df[rt_col]

    quantiles = df.quantile(self.quantile)
    df_clipped = df.clip(upper=quantiles, axis=1) # column-wise
    rolling = df_clipped.rolling(lag)
    k = np.sqrt(2)

    sigma = rolling.std()
    return sigma
