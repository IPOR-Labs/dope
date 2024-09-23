import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from dope.price_model.base import BasePredictor


class VectorAutoRegPredictor(BasePredictor):
    def __init__(self, days_window_past, steps_in_the_future):
        # super().__init__()
        self.days_window = days_window_past
        self.steps = steps_in_the_future

    def exp_returns(self, date_ix, token):

        df = self.data.to_block(token).apyBase.dropna()
        df.index = pd.DatetimeIndex(df.index.values, freq=df.index.inferred_freq)

        # Fit the VAR model
        _filter = df.index <= date_ix
        _filter &= df.index > date_ix - pd.Timedelta(f"{self.days_window}D")
        model = VAR(df[_filter])
        results = model.fit(maxlags=7)

        steps = 7
        forecast = results.forecast(df.values[-results.k_ar :], steps=steps)

        last_date = df.index[-1]
        freq = df.index.freq
        next_steps = pd.date_range(start=last_date + freq, periods=steps, freq=freq)
        forecast = pd.DataFrame(forecast, columns=df.columns, index=next_steps)

        exp_returns = {}

        def plot(self):
            import matplotlib.pyplot as plt

            colors = plt.cm.tab20(np.linspace(0, 0.45, len(df.columns))[::])
            COLS = df.columns
            for ix in range(len(COLS)):
                c = COLS[ix]
                plt.plot(df.index, df[c], color=colors[ix], label=c)
                plt.plot(forecast.index, forecast[c], "--", color=colors[ix])

            _ = plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Strategy")
