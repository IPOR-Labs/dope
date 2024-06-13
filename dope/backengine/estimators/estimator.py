from dope.backengine.estimators.baseestimator import BaseEstimator
import numpy as np

class Estimator(BaseEstimator):
  def __init__(self):
    pass
  
  def rolling_fit_mu(self, df, lag, rt_col):
    """
    df: Dataframe with index as datetime
    dt: Rolling window
    rt_col: column with the rates data to fit
    """
    
    rolling = df[rt_col].rolling(lag)
    k = np.sqrt(2)

    mu = rolling.mean()
    
    return mu
  
  def rolling_fit_sigma(self, df, lag, rt_col):
    """
    df: Dataframe with index as datetime
    lag: Rolling window
    rt_col: column with the rates data to fit
    """
    
    rolling = df[rt_col].rolling(lag)
    k = np.sqrt(2)

    sigma = rolling.std()
    return sigma
