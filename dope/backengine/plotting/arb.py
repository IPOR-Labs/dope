# fund of fund article misc functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, summary_dictionary):
        self.summary = pd.concat(summary_dictionary).unstack(level=0)
    
    
    def _plot_risk_reward(self, summary=None):
        summary = summary or self.summary
        
        
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

        plt.figure(figsize=(6, 6))
        returns = annualized_returns
        vols = annualized_vol

        colors = plt.cm.tab10(np.linspace(0, 1, len(returns))[::])
        COLS = returns.sort_values(ascending=False).index
        for i in range(len(COLS)):
            c = COLS[i]
            marker = "o"
            plt.scatter(vols[c], returns[c], label=c, color=colors[i], marker=marker)
        _ = plt.legend(
            loc="upper left", bbox_to_anchor=(1, 1), title="Strategy (Higher PnL First)"
        )

        plt.title(f"Risk vs Return")
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Annualized Returns (%)")
        plt.show()
        print("Annualized Returns (base currency):")
        print(returns.sort_values(ascending=False))
        return returns
            
        
    def _plot_annualized_returns(self, summary=None, title="", xlim=None, ylim=None, ax=None, price_token=None):
        if ax is None:
            plt.figure(figsize=(16, 4))
            ax = plt.gca()
        
        if price_token is None:
            r_t = (
                (summary.capital) 
                / (summary.capital.apply(lambda col: col.dropna().iloc[0]))
                - 1
            ) * 100
        else:
            price_df = summary.prices.apply(lambda x: x.apply(lambda y: y[price_token]) , axis=0)
            non_nan_indices = {col: summary[('capital', col)].first_valid_index() for col in summary['capital'].columns}

            price_init = np.array([
                summary.at[non_nan_indices[col], ('prices', col)][price_token]
                for col in summary['capital'].columns
            ])
            capital_init = np.array([
                summary.at[non_nan_indices[col], ('capital', col)]
                for col in summary['capital'].columns
            ])
            r_t = (
                (summary.capital / price_df) 
                / (capital_init / price_init)
                - 1
            ) * 100
        r_t = r_t.resample("1h").last().interpolate()
        T = 365.25 * 24 * 60 * 60

        annualized_rt = r_t.apply(
            lambda col: col.dropna()
            * T
            / (col.dropna().index[-1] - col.dropna().index[0]).total_seconds()
        )
        annualized_rt.plot(ax=ax, linewidth=4)
        ax.set_title(f"Cumulative Annualized returns {title} (in {price_token})", fontsize=22)
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

    def summary_view(self, summary=None, title_1="", title_2="", xlim=None, ylim=None):
        summary = summary or self.summary

        plt.figure(figsize=(16, 4))
        ax = plt.gca()
        summary.rate.resample("5min").last().interpolate().plot(ax=ax)
        ax.axhline(0, color="grey")
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
        ax.set_title(f"Daily Returns (Token Aggregated) {title_1}", fontsize=22)
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
        if "prices" in summary.columns:
            cols = summary.prices.columns
            prices_keys = summary.prices[cols[0]].iloc[0].keys()
            for price_token in prices_keys:
                self._plot_annualized_returns(summary, title_2, xlim, ylim, price_token=price_token)
        else:
            self._plot_annualized_returns(summary, title_2, xlim, ylim)

        plt.figure(figsize=(6, 6))
        returns = annualized_returns
        vols = annualized_vol

        colors = plt.cm.tab10(np.linspace(0, 1, len(returns))[::])
        COLS = returns.sort_values(ascending=False).index
        for i in range(len(COLS)):
            c = COLS[i]
            marker = "o"
            plt.scatter(vols[c], returns[c], label=c, color=colors[i], marker=marker)
        _ = plt.legend(
            loc="upper left", bbox_to_anchor=(1, 1), title="Strategy (Higher PnL First)"
        )

        plt.title(f"Risk vs Return")
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Annualized Returns (%)")
        plt.show()
        print("Annualized Returns (base currency):")
        print(returns.sort_values(ascending=False))
    
    def weights_timeseries(self, strategy_name, stack=True, token=None, figsize=(16, 4)):
        
        if token is None:
            token = list(self.summary.ws.dropna().iloc[0,0].keys())
            if len(token) == 0:
                raise ValueError("No token found in the summary. Try to pass `token` variable to method",)
            token = token[0]

        if strategy_name not in self.summary.ws.columns:
            raise ValueError(
                f"Strategy {strategy_name} not found in the summary."
                f" Available strategies are {self.summary.ws.columns}"
            )
        
        colormap = plt.get_cmap('tab20')
        plt.figure(figsize=figsize)
        plt.title("Weights Timeseries")
        ax = plt.gca()
        pd.json_normalize(
            self.summary.ws[strategy_name].apply(lambda row: row[token])
        ).plot(kind='area', stacked=stack, colormap=colormap, ax=ax)
        
        
        _ = plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Pools")
        # Remove the top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Optionally, you can also move the left and bottom spines
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
