# fund of fund article misc functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, summary_dictionary):
        self.summary = pd.concat(summary_dictionary).unstack(level=0)

    def summary_view(self, summary=None, title_2="", xlim=None, ylim=None):
        summary = summary or self.summary

        plt.figure(figsize=(16, 4))
        ax = plt.gca()
        summary.rate.resample("1h").last().interpolate().plot(ax=ax)
        _ = plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Strategy")

        # Remove the top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Optionally, you can also move the left and bottom spines
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
        ax.set_xlabel("")
        ax.set_ylabel("Returns (in %)")
        if xlim is not None:
            _ = plt.xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))
        plt.show()

        r_t = (
            summary.capital / summary.capital.apply(lambda col: col.dropna().iloc[0])
            - 1
        ) * 100
        r_t = r_t.resample("1h").last().interpolate()
        T = 365.25 * 24 * 60 * 60
        annualized_returns = r_t.apply(
            lambda col: col.dropna().iloc[-1]
            * T
            / (col.dropna().index[-1] - col.dropna().index[0]).total_seconds()
        )
        annualized_vol = r_t.apply(
            lambda col: col.dropna().std()
            * np.sqrt(
                T / (col.dropna().index[-1] - col.dropna().index[0]).total_seconds()
            )
        )
        plt.figure(figsize=(16, 4))
        ax = plt.gca()
        annualized_rt = r_t.apply(
            lambda col: col.dropna()
            * T
            / (col.dropna().index[-1] - col.dropna().index[0]).total_seconds()
        )
        annualized_rt.plot(ax=ax, linewidth=4)
        ax.set_title(title_2, fontsize=22)
        ax.set_xlabel("")
        ax.set_ylabel("Returns (in %)")
        _ = plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Strategy")
        # Remove the top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Optionally, you can also move the left and bottom spines
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
        if xlim is not None:
            _ = plt.xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))
        if ylim is not None:
            _ = plt.ylim(ylim[0], ylim[1])
        plt.show()

        plt.figure(figsize=(6, 6))
        returns = annualized_returns
        vols = annualized_vol

        colors = plt.cm.tab10(np.linspace(0, 1, len(returns))[::])
        COLS = returns.sort_values(ascending=False).index
        for i in range(len(COLS)):
            c = COLS[i]
            if "arb" in c:
                marker = "x"
            elif "jack" in c:
                marker = "s"
            else:
                marker = "o"
            plt.scatter(vols[c], returns[c], label=c, color=colors[i], marker=marker)
        _ = plt.legend(
            loc="upper left", bbox_to_anchor=(1, 1), title="Strategy (Higher PnL First)"
        )

        plt.title(f"Risk vs Return")
        plt.show()
        print(returns)
