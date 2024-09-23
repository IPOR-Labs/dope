import pandas as pd
from dope.backengine.estimators.baseestimator import BaseEstimator
from dope.backengine.triggers.basetrigger import BaseTrigger


class ConfidenceBandTrigger(BaseTrigger):
    def __init__(self, df, lag, lagSD, est, rt_col, k=0.5, protocolList=[]):
        self.df = df
        self.lag = lag
        self.lagSD = -lagSD if lagSD < 0 else lagSD
        self.equaliseSDs = lagSD < 0
        self.est = est
        self.protocolList = protocolList
        self.rt_col = rt_col
        self.k = k

    def pairs(self):
        protocols = [x[1] for x in list(self.df.columns) if x != "cash"]
        if self.protocolList != []:
            protocols = [v for v in protocols if v in self.protocolList and v != "cash"]
        N = len(protocols)
        return [
            (protocols[i], protocols[j]) for i in range(0, N) for j in range(i + 1, N)
        ]

    def rates(self):
        _df = self.df[[self.rt_col]]
        if self.lag == 0:
            mu = _df[self.rt_col]  # could equivalently use 'supplyRate' or borrowRate'
        else:
            mu = self.est.rolling_fit_mu(
                df=_df, lag=self.lag, rt_col=self.rt_col
            )  # could equivalently use 'supplyRate' or borrowRate'

        sigma = self.est.rolling_fit_sigma(
            df=_df, lag=self.lagSD, rt_col=self.rt_col
        )  # identify which column to use for calculating triggers
        if self.equaliseSDs:
            sigma_avg = sigma.mean(axis=1)
            for col in sigma.columns:
                sigma[col] = sigma_avg

        t_dn = mu - self.k * sigma
        t_up = mu + self.k * sigma
        mu.columns = [(self.rt_col, c + ":mu") for c in mu.columns]
        sigma.columns = [(self.rt_col, c + ":vol") for c in sigma.columns]
        t_dn.columns = [(self.rt_col, c + ":dn") for c in t_dn.columns]
        t_up.columns = [(self.rt_col, c + ":up") for c in t_up.columns]
        return pd.concat([_df, sigma, t_dn, t_up], axis=1)

    def dates(self):
        _df = self.df[[self.rt_col]]

        if self.lag == 0:
            mu = _df[self.rt_col]  # could equivalently use 'supplyRate' or borrowRate'
        else:
            mu = self.est.rolling_fit_mu(
                df=_df, lag=self.lag, rt_col=self.rt_col
            )  # could equivalently use 'supplyRate' or borrowRate'

        sigma = self.est.rolling_fit_sigma(
            df=_df, lag=self.lagSD, rt_col=self.rt_col
        )  # identify which column to use for calculating triggers

        if self.equaliseSDs:
            sigma_avg = sigma.mean(axis=1)
            for col in sigma.columns:
                sigma[col] = sigma_avg

        t_dn = mu - self.k * sigma
        t_up = mu + self.k * sigma
        _filter = pd.Series(
            index=pd.Index([], dtype=int), dtype=bool
        )  # initialise to empty
        t = t_dn.copy()
        for pair in self.pairs():
            asset1, asset2 = pair[0], pair[1]
            if t_dn[asset1].notnull().sum() == 0 or t_dn[asset2].notnull().sum() == 0:
                continue
            t = t_dn.copy().rename(
                columns={asset1: asset1 + "_dn", asset2: asset2 + "_dn"}
            )
            t[asset1 + "_up"] = t_up[asset1]
            t[asset2 + "_up"] = t_up[asset2]
            t["diff_1"] = t[asset1 + "_up"] - t[asset2 + "_dn"]
            t["diff_2"] = t[asset1 + "_dn"] - t[asset2 + "_up"]

            t = t.fillna(
                0
            )  # if NaN, the zero will mean t[diff_1] and t[diff_2] will neither be >0 nor <0 so no trigger should eventuate

            _filter1 = t["diff_1"] > 0
            _filter1 &= t["diff_1"].shift(1) < 0
            _filter2 = t["diff_2"] > 0
            _filter2 &= t["diff_2"].shift(1) < 0
            _filter12 = _filter1 | _filter2
            _filter = _filter12 if _filter.empty else _filter | _filter12
        #      _filter = _filter2 if _filter.empty else _filter | _filter2
        return list(t[_filter].index)

    def crossings(self):
        _df = self.df[[self.rt_col]]

        if self.lag == 0:
            mu = _df[self.rt_col]  # could equivalently use 'supplyRate' or borrowRate'
        else:
            mu = self.est.rolling_fit_mu(
                df=_df, lag=self.lag, rt_col=self.rt_col
            )  # could equivalently use 'supplyRate' or borrowRate'

        sigma = self.est.rolling_fit_sigma(
            df=_df, lag=self.lagSD, rt_col=self.rt_col
        )  # identify which column to use for calculating triggers
        if self.equaliseSDs:
            sigma_avg = sigma.mean(axis=1)
            for col in sigma.columns:
                sigma[col] = sigma_avg

        t_dn = mu - self.k * sigma
        t_up = mu + self.k * sigma
        t = t_dn.copy()
        retList = []
        for pair in self.pairs():
            asset1, asset2 = pair[0], pair[1]
            if t_dn[asset1].notnull().sum() == 0 or t_dn[asset2].notnull().sum() == 0:
                continue
            t = t_dn.copy().rename(
                columns={asset1: asset1 + "_dn", asset2: asset2 + "_dn"}
            )
            t[asset1 + "_up"] = t_up[asset1]
            t[asset2 + "_up"] = t_up[asset2]
            t["diff_1"] = t[asset1 + "_up"] - t[asset2 + "_dn"]
            t["diff_2"] = t[asset1 + "_dn"] - t[asset2 + "_up"]
            _filter1 = t["diff_1"] > 0
            _filter1 &= t["diff_1"].shift(1) < 0
            _filter2 = t["diff_2"] > 0
            _filter2 &= t["diff_2"].shift(1) < 0
            _filter12 = _filter1 | _filter2
            if len(t[_filter12].index) > 0:
                retList.append([(asset1, asset2, x) for x in t[_filter12].index])
        return retList


import pandas as pd
from dope.backengine.estimators.baseestimator import BaseEstimator
from dope.backengine.triggers.basetrigger import BaseTrigger


class ConfidenceBandTrigger(BaseTrigger):
    def __init__(self, df, lag, lagSD, est, rt_col, k=0.5, protocolList=[]):
        self.df = df
        self.lag = lag
        self.lagSD = -lagSD if lagSD < 0 else lagSD
        self.equaliseSDs = lagSD < 0
        self.est = est
        self.protocolList = protocolList
        self.rt_col = rt_col
        self.k = k

    def pairs(self):
        protocols = [x[1] for x in list(self.df.columns) if x != "cash"]
        if self.protocolList != []:
            protocols = [v for v in protocols if v in self.protocolList and v != "cash"]
        N = len(protocols)
        return [
            (protocols[i], protocols[j]) for i in range(0, N) for j in range(i + 1, N)
        ]

    def rates(self):
        _df = self.df[[self.rt_col]]
        if self.lag == 0:
            mu = _df[self.rt_col]  # could equivalently use 'supplyRate' or borrowRate'
        else:
            mu = self.est.rolling_fit_mu(
                df=_df, lag=self.lag, rt_col=self.rt_col
            )  # could equivalently use 'supplyRate' or borrowRate'

        sigma = self.est.rolling_fit_sigma(
            df=_df, lag=self.lagSD, rt_col=self.rt_col
        )  # identify which column to use for calculating triggers
        if self.equaliseSDs:
            sigma_avg = sigma.mean(axis=1)
            for col in sigma.columns:
                sigma[col] = sigma_avg

        t_dn = mu - self.k * sigma
        t_up = mu + self.k * sigma
        mu.columns = [(self.rt_col, c + ":mu") for c in mu.columns]
        sigma.columns = [(self.rt_col, c + ":vol") for c in sigma.columns]
        t_dn.columns = [(self.rt_col, c + ":dn") for c in t_dn.columns]
        t_up.columns = [(self.rt_col, c + ":up") for c in t_up.columns]
        return pd.concat([_df, sigma, t_dn, t_up], axis=1)

    def dates(self):
        _df = self.df[[self.rt_col]]

        if self.lag == 0:
            mu = _df[self.rt_col]  # could equivalently use 'supplyRate' or borrowRate'
        else:
            mu = self.est.rolling_fit_mu(
                df=_df, lag=self.lag, rt_col=self.rt_col
            )  # could equivalently use 'supplyRate' or borrowRate'

        sigma = self.est.rolling_fit_sigma(
            df=_df, lag=self.lagSD, rt_col=self.rt_col
        )  # identify which column to use for calculating triggers

        if self.equaliseSDs:
            sigma_avg = sigma.mean(axis=1)
            for col in sigma.columns:
                sigma[col] = sigma_avg

        t_dn = mu - self.k * sigma
        t_up = mu + self.k * sigma
        _filter = pd.Series(
            index=pd.Index([], dtype=int), dtype=bool
        )  # initialise to empty
        t = t_dn.copy()
        for pair in self.pairs():
            asset1, asset2 = pair[0], pair[1]
            if t_dn[asset1].notnull().sum() == 0 or t_dn[asset2].notnull().sum() == 0:
                continue
            t = t_dn.copy().rename(
                columns={asset1: asset1 + "_dn", asset2: asset2 + "_dn"}
            )
            t[asset1 + "_up"] = t_up[asset1]
            t[asset2 + "_up"] = t_up[asset2]
            t["diff_1"] = t[asset1 + "_up"] - t[asset2 + "_dn"]
            t["diff_2"] = t[asset1 + "_dn"] - t[asset2 + "_up"]

            t = t.fillna(
                0
            )  # if NaN, the zero will mean t[diff_1] and t[diff_2] will neither be >0 nor <0 so no trigger should eventuate

            _filter1 = t["diff_1"] > 0
            _filter1 &= t["diff_1"].shift(1) < 0
            _filter2 = t["diff_2"] > 0
            _filter2 &= t["diff_2"].shift(1) < 0
            _filter12 = _filter1 | _filter2
            _filter = _filter12 if _filter.empty else _filter | _filter12
        #      _filter = _filter2 if _filter.empty else _filter | _filter2
        return list(t[_filter].index)

    def crossings(self):
        _df = self.df[[self.rt_col]]

        if self.lag == 0:
            mu = _df[self.rt_col]  # could equivalently use 'supplyRate' or borrowRate'
        else:
            mu = self.est.rolling_fit_mu(
                df=_df, lag=self.lag, rt_col=self.rt_col
            )  # could equivalently use 'supplyRate' or borrowRate'

        sigma = self.est.rolling_fit_sigma(
            df=_df, lag=self.lagSD, rt_col=self.rt_col
        )  # identify which column to use for calculating triggers
        if self.equaliseSDs:
            sigma_avg = sigma.mean(axis=1)
            for col in sigma.columns:
                sigma[col] = sigma_avg

        t_dn = mu - self.k * sigma
        t_up = mu + self.k * sigma
        t = t_dn.copy()
        retList = []
        for pair in self.pairs():
            asset1, asset2 = pair[0], pair[1]
            if t_dn[asset1].notnull().sum() == 0 or t_dn[asset2].notnull().sum() == 0:
                continue
            t = t_dn.copy().rename(
                columns={asset1: asset1 + "_dn", asset2: asset2 + "_dn"}
            )
            t[asset1 + "_up"] = t_up[asset1]
            t[asset2 + "_up"] = t_up[asset2]
            t["diff_1"] = t[asset1 + "_up"] - t[asset2 + "_dn"]
            t["diff_2"] = t[asset1 + "_dn"] - t[asset2 + "_up"]
            _filter1 = t["diff_1"] > 0
            _filter1 &= t["diff_1"].shift(1) < 0
            _filter2 = t["diff_2"] > 0
            _filter2 &= t["diff_2"].shift(1) < 0
            _filter12 = _filter1 | _filter2
            if len(t[_filter12].index) > 0:
                retList.append([(asset1, asset2, x) for x in t[_filter12].index])
        return retList
