import pandas as pd
from dope.price_model.base import BasePredictor


class MavgPredictor(BasePredictor):
    def __init__(self, days_window):
        # super().__init__()
        self.days_window = days_window

    def exp_returns(self, date_ix, token):

        exp_returns = {}

        for protocol, df in self.data_ref[token].items():
            _filter = df.index <= date_ix
            _filter &= df.index > date_ix - pd.Timedelta(f"{self.days_window}D")
            if len(df[_filter]) == 0:
                continue
            exp_returns[protocol] = df[_filter]["apyBase"].mean()
        return exp_returns
