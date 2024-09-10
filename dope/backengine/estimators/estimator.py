from dope.backengine.estimators.baseestimator import BaseEstimator
import numpy as np
from datetime import timedelta


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
        sqrt_dt = np.sqrt(timedelta(1) / ((df[rt_col].index[-1] - df[rt_col].index[0]) / (df[rt_col].count() - 1)))

        sigma = sqrt_dt * rolling.std()
        return sigma
