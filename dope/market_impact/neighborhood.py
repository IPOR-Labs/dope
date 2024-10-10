import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def find_closest_rows(df, column_name, value):
    # Ensure the DataFrame is sorted by the specified column
    df = df.sort_values(by=column_name)

    # Check if the value is less than the minimum or greater than the maximum
    if value <= df[column_name].iloc[0]:
        return df.iloc[:2]
    elif value >= df[column_name].iloc[-1]:
        return df.iloc[-2:]

    # Find the two rows between which the value falls
    for i in range(1, len(df)):
        if df[column_name].iloc[i - 1] <= value <= df[column_name].iloc[i]:
            return df.iloc[i - 1 : i + 1]


def interpolate_value(df, apy_col, ur_col, value):
    """
    It interpolates if value is between two values
    otherwise it extrapolates
    """
    rows = find_closest_rows(df, ur_col, value)
    if rows is None:
        return np.nan
    

    x0, y0 = rows[ur_col].iloc[0], rows[apy_col].iloc[0]
    x1, y1 = rows[ur_col].iloc[1], rows[apy_col].iloc[1]

    # Linear interpolation
    interpolated_value = y0 + (value - x0) * (y1 - y0) / (x1 - x0)

    return interpolated_value


class NeighborhoodMktImpactModel:
    def __init__(self, past_window_days=7, future_window_days=0):
        self._data_ref = None
        self.past_window_days = past_window_days
        self.future_window_days = future_window_days

    def set_data_ref(self, data_ref):
        self._data_ref = data_ref

    def __getitem__(self, _):
        return self

    @property
    def data_ref(self):
        return self._data_ref

    def get_data_slice_for_date(self, date_ix):
        end_index = date_ix + pd.DateOffset(days=self.future_window_days)
        start_index = date_ix - pd.DateOffset(days=self.past_window_days)

        _filter = start_index <= self.data_ref.index
        _filter &= self.data_ref.index <= end_index

        df = self.data_ref[_filter]
        return df

    def impact(self, timestamp, capital, is_borrow, should_plot=False):
        """
        is_borrow is a boolean that is true if the borrow rate should be used. otherwise use lend/supply rate.
        """

        df = self.get_data_slice_for_date(timestamp)
        if len(df) == 0:
            return np.nan

        _filter = self.data_ref.index <= timestamp
        row = self.data_ref[_filter].iloc[-1]

        if row["totalSupplyUsd"] is None:
            return np.nan
        if row["totalBorrowUsd"] is None:
            return np.nan
        if is_borrow:
            ur1 = (row["totalBorrowUsd"] + capital) / (row["totalSupplyUsd"])
        else:
            ur1 = row["totalBorrowUsd"] / (capital + row["totalSupplyUsd"])
        ur0 = row["totalBorrowUsd"] / (row["totalSupplyUsd"])

        r0 = interpolate_value(
            df, apy_col="apyBase", ur_col="utilizationRate", value=ur0
        )
        r1 = interpolate_value(
            df, apy_col="apyBase", ur_col="utilizationRate", value=ur1
        )
        if is_borrow:
            # borrow money from pool never decreases rates
            impact = max(0, r1 - r0)
        else:
            # lending money to pool never increases rates
            impact = min(0, r1 - r0)
        if should_plot:
            self.plot(ur0, r0, ur1, r1, df)
        return impact

    def fit(self, *args, **kwargs):
        pass

    def plot(self, ur0, r0, ur1, r1, df):
        plt.figure(figsize=(20, 5))
        tmp = df.sort_values("utilizationRate")
        plt.plot(tmp.utilizationRate, tmp.apyBase, linewidth=3)

        plt.scatter(ur0, r0, marker="o", color="tab:red")
        plt.scatter(ur1, r1, marker="o", color="tab:green")

        plt.axvline(ur0, color="tab:red", label="Start UR")
        plt.axvline(ur1, color="green", label="End UR")
        # plt.title(f"Slippage Model\n[${capital/1e6:,.2f}M] {mkt} @ {date_ix.date()}\n", fontsize=22)
        # plt.xlim(xmin*0.95, xmax*1.05)

        _min, _max = tmp.apyBase.min(), tmp.apyBase.max()
        start = (ur0 * 0.99, (_min + _max) / 2)
        end = (ur1 * 1.01, (_min + _max) / 2)
        arrowstyle = "<-" if ur0 < ur1 else "->"

        plt.annotate(
            "",
            xy=end,  # Ending point of the arrow
            xytext=start,  # Starting point of the arrow
            arrowprops=dict(facecolor="black", arrowstyle=arrowstyle),
            # textcoords='offset points',  # To position text relative to the arrow end
            # ha='center',
            # va='bottom'  # Align text vertically below the annotation point
        )

        midpoint_x = (start[0] + end[0]) / 2
        midpoint_y = (_min + _max) / 2
        plt.text(midpoint_x, midpoint_y, "Slippage", ha="center", va="bottom")

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
        ax.set_xlabel("Utilization Rate")
        ax.set_ylabel("Pool Yield (in %)")
