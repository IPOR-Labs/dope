import pandas as pd


class Backtester:

    def __init__(self, lender, data, mkt_impact):
        self.lender = lender
        self.data = data
        self.mkt_impact = mkt_impact
        self.dates = self.get_dates()
        self.summary = None

    def get_dates(self):
        dates = set()
        for protocol, df in self.data.items():
            dates.update(df.index)
        dates = list(dates)
        dates.sort()
        return dates

    def prep(self):
        print("perp?")
        self.lender.register_engine(self)

        self.data.add_cash_mkt()
        for _token in self.data.keys():
            for mkt in self.data[_token].keys():
                self.mkt_impact[mkt].set_data_ref(self.data[_token][mkt])

    def __call__(self):
        end_timestamp = pd.to_datetime("2023-07-23")
        start_timestamp = pd.to_datetime("2023-07-15")

        self.prep()  # register data, slippage model, strategy

        rows = []
        for i in range(20, len(self.dates[:]) - 1):
            date_prev = self.dates[i]
            date_now = self.dates[i + 1]
            # if date_now < start_timestamp:
            #  continue
            # if date_now >= end_timestamp:
            #  break
            # print(date_now)

            ws = self.lender.on_act(date_prev)
            # rate = 0
            r_breakdown = {}
            slopes = {}
            impacts = {}
            for mkt, w in ws.items():
                df = self.data[mkt]
                _filter = df.index < date_now
                if len(df[_filter]) == 0:
                    continue

                slopes[mkt] = self.mkt_impact[mkt].get_slope(
                    df[_filter].utilizationRate.iloc[-1]
                )
                impacts[mkt] = (
                    self.lender.capital / df[_filter].tvlUsd.iloc[-1] * slopes[mkt]
                )
                # mkt_impact = lender.capital/df[_filter].tvlUsd.iloc[-1] * slope
                r_breakdown[mkt] = max(
                    0, (w * (df[_filter]["apy"].iloc[-1] - impacts[mkt]))
                )
                # rate += r_breakdown[mkt]
                # print(date_now, w, mkt, mkt_impact, (w*(df[_filter]["apy"].iloc[-1] - mkt_impact)))
            # print()
            # mkt, rate
            # break
            rows.append(
                [date_now, ws, sum(r_breakdown.values()), r_breakdown, slopes, impacts]
            )
        strategy = pd.DataFrame(
            rows, columns=["datetime", "ws", "rate", "breakdown", "slope", "mkt_impact"]
        )
        strategy = strategy.set_index("datetime")
        self.summary = strategy
        return self.summary
