from dope.backengine.estimators.baseestimator import BaseEstimator
import numpy as np
from datetime import timedelta


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
        df_clipped = df.clip(upper=quantiles, axis=1)  # column-wise
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
        df_clipped = df.clip(upper=quantiles, axis=1)  # column-wise
        rolling = df_clipped.rolling(lag)
        sqrt_dt = np.sqrt(
            timedelta(1)
            / ((df_clipped.index[-1] - df_clipped.index[0]) / (df_clipped.count() - 1))
        )

        sigma = sqrt_dt * rolling.std()
        return sigma
