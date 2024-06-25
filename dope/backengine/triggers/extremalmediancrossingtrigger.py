import pandas as pd
from dope.backengine.estimators.baseestimator import BaseEstimator
from dope.backengine.triggers.basetrigger import BaseTrigger


class ExtremalMedianCrossingTrigger(BaseTrigger):
    def __init__(self, df, lag, est, rt_col, protocolList=[]):
        self.df = df
        self.lag = lag
        self.est = est
        self.protocolList = protocolList
        self.rt_col = rt_col

    def pairs(self):
        protocols = [x[1] for x in list(self.df.columns)]
        if self.protocolList != []:
            protocols = [v for v in protocols if v in self.protocolList]
        N = len(protocols)
        return [(protocols[i], protocols[j]) for i in range(0, N) for j in range(i + 1, N)]

    def dates(self):
        _df = self.df[[self.rt_col]]
        mu = self.est.rolling_fit_mu(df=_df, lag=self.lag,
                                     rt_col=self.rt_col)  # could equivalently use 'supplyRate' or borrowRate'
        _mu = mu.copy()
        _mu['min'] = mu.min(axis=1)
        _mu['median'] = mu.median(axis=1)
        _mu['max'] = mu.max(axis=1)

        _filter = pd.Series(index=pd.Index([], dtype=int), dtype=bool)  # initialise to empty

        for asset in mu.columns:
            _filter1 = _mu[asset] == _mu['max']
            _filter1 &= _mu[asset].shift(1) < _mu['median'].shift(1)
            _filter2 = _mu[asset] == _mu['min']
            _filter2 &= _mu[asset].shift(1) > _mu['median'].shift(1)
            _filter12 = (_filter1 | _filter2)
            _filter = _filter12 if _filter.empty else _filter | _filter12

        return list(_mu[_filter].index)
