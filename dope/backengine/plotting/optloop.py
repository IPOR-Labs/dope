# fund of fund article misc functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from itertools import chain as itertools_chain, combinations, product
from dataclasses import dataclass, field
from typing import Dict
import pandas as pd


@dataclass
class Leaf:
    """Represents a leaf node containing a dataframe and weight."""
    name: str
    weight: float
    leverage: float
    dataframe: pd.DataFrame


@dataclass
class Protocol:
    """Represents a protocol node containing leaves."""
    leaves: Dict[str, Leaf] = field(default_factory=dict)  # Leaf objects by name


@dataclass
class Chain:
    """Represents a chain node containing protocols."""
    protocols: Dict[str, Protocol] = field(default_factory=dict)  # Protocols by name


@dataclass
class LoopTree:
    """Represents the overall tree structure."""
    chains: Dict[str, Chain] = field(default_factory=dict)  # Chains by name

    def add_leaf(self, chain_name: str, protocol_name: str, leaf_name: str, weight: float, leverage:float, dataframe: pd.DataFrame):
        """
        Adds a leaf to a protocol under a chain, creating chains and protocols as needed.
        :param chain_name: Name of the chain
        :param protocol_name: Name of the protocol
        :param leaf_name: Name of the leaf
        :param weight: Weight of the leaf
        :param dataframe: Dataframe to associate with the leaf
        """
        # Ensure the chain exists
        if chain_name not in self.chains:
            self.chains[chain_name] = Chain()

        # Ensure the protocol exists
        chain = self.chains[chain_name]
        if protocol_name not in chain.protocols:
            chain.protocols[protocol_name] = Protocol()

        # Add the leaf
        protocol = chain.protocols[protocol_name]
        protocol.leaves[leaf_name] = Leaf(leaf_name, weight, leverage, dataframe)

    def get_portfolio(self, chain_name: str) -> Dict[str, Leaf]:
        """
        Returns a portfolio of one leaf from each protocol under the specified chain.
        The selection is based on the highest weight.
        :param chain_name: Name of the chain
        :return: Dictionary of protocol names to the selected leaf
        """
        chain = self.chains.get(chain_name)
        if chain is None:
            raise ValueError(f"Chain {chain_name} does not exist.")

        portfolio = {}
        for protocol_name, protocol in chain.protocols.items():
            if not protocol.leaves:
                raise ValueError(f"No leaves defined for protocol {protocol_name} in chain {chain_name}.")

            # Select the leaf with the highest weight
            best_leaf_name = max(protocol.leaves, key=lambda leaf_name: protocol.leaves[leaf_name].weight)
            portfolio[protocol_name] = protocol.leaves[best_leaf_name]

        return portfolio

    def generate_portfolios(self, chain_name: str, target_weight: float):
        """
        Generates all possible portfolios such that:
        - Each portfolio has one leaf per protocol or a subset of protocols.
        - The sum of selected leaf weights does not exceed the target weight.
        :param chain_name: Name of the chain.
        :param target_weight: Target sum of weights for the portfolio.
        :return: List of possible portfolios, each represented as a dictionary {protocol_name: leaf}.
        """
        chain = self.chains.get(chain_name)
        if chain is None:
            raise ValueError(f"Chain {chain_name} does not exist.")

        # Collect all leaves for each protocol
        protocol_leaves = {
            protocol_name: list(protocol.leaves.items())
            for protocol_name, protocol in chain.protocols.items()
        }

        # Generate all subsets of protocols (including single protocols)
        protocols = list(protocol_leaves.keys())
        all_protocol_subsets = itertools_chain.from_iterable(combinations(protocols, r) for r in range(1, len(protocols) + 1))

        valid_portfolios = []
        for protocol_subset in all_protocol_subsets:
            # Generate all combinations of one leaf per protocol in the subset
            subset_leaves = {protocol: protocol_leaves[protocol] for protocol in protocol_subset}
            all_combinations = product(*subset_leaves.values())

            for combination in all_combinations:
                portfolio = {protocol_name: leaf for protocol_name, (leaf_name, leaf) in zip(subset_leaves.keys(), combination)}
                total_weight = sum(leaf.weight for leaf in portfolio.values())
                if total_weight <= target_weight:  # Portfolio weight constraint
                    valid_portfolios.append(portfolio)

        return valid_portfolios






class Plotter:
    def __init__(self, summary_dictionary):
        self.raw_summary = summary_dictionary
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

        #
        #colors = plt.cm.viridis(np.linspace(0, 1, 3500)[::])
        COLS = returns.sort_values(ascending=False).index
        colors = set()
        tags = set()
        for i in range(len(COLS)):            
            c = COLS[i]
            cap = float(c.split("_")[-2].split(":")[-1][:-1])
            colors.add(cap)
            tag = str(c.split(":")[1])
            tags.add(tag)

        cmap = plt.cm.tab10(np.linspace(0, 1, len(colors))[::])
        colors = {tag: cmap[i] for i, tag in enumerate(colors)}
        markers = ["o", "x", "s", "D", "^", "v", "<", ">", "p", "P", "*", "h", "H", "+", "X"]
        marker_dict = {tag: markers[i] for i, tag in enumerate(tags)}

        for i in range(len(COLS)):            
            c = COLS[i]
            cap = float(c.split("_")[-2].split(":")[-1][:-1])
            tag = str(c.split(":")[1])
            if returns[c] <0:
                continue
            plt.scatter(vols[c], returns[c], label=cap, color=colors[cap], marker=marker_dict[tag])

        # handles, unique_labels = plt.gca().get_legend_handles_labels()
        # unique_labels_dict = dict(zip(unique_labels, handles))
        # _ = plt.legend(unique_labels_dict.values(), unique_labels_dict.keys(),
        #     loc="upper left", bbox_to_anchor=(1, 1), title="Strategy (Higher PnL First)"
        # )
        
        # Create legend entries for colors
        color_handles = [
            plt.Line2D([0], [0], marker='>', color='w', markerfacecolor=color, markersize=10, label=f"{label*100:.0f}%")
            for label, color in colors.items()
        ]

        # Create legend entries for markers
        marker_handles = [
            plt.Line2D([0], [0], marker=marker, color='k', markersize=5, label=label, linestyle='None')
            for label, marker in marker_dict.items()
        ]

        # Add legends to the plot
        plt.legend(
            handles=color_handles + marker_handles, 
            title="Capital / Pools", 
            loc='upper left',
            bbox_to_anchor=(1, 1)
        )


        plt.title(f"Risk vs Return")
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Annualized Returns (%)")
        plt.show()
        print("Annualized Returns (base currency):")
        print(returns[returns>0].sort_values(ascending=False))
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
        cols = summary.prices.columns
        prices_keys = summary.prices[cols[0]].iloc[0].keys()
        for price_token in prices_keys:
            self._plot_annualized_returns(summary, title_2, xlim, ylim, price_token=price_token)

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
        
    def tree_risk_return(self, weight_cut=1):
        tree = LoopTree()
        run_summary = self.raw_summary

        
        for k in run_summary.keys():
            p, cap, w, loop = k.split("_")
            chain, protocol = p.split(":")[1:3]
            tree.add_leaf(
                chain_name=chain,
                protocol_name=protocol, 
                leaf_name=k,
                weight=float(w.split(":")[1]),
                leverage=float(loop.split(":")[1][:-1]),
                dataframe=run_summary[k][run_summary[k].index >= "2024-10-20"]
            )

        dfs = {}
        for chain_name in tree.chains.keys():
            πs = tree.generate_portfolios(chain_name, weight_cut)
            ROWS = []
            for π in πs:
                df = None
                df_rets = None
                ws = 0
                names = ""
                counter = 0
                for k in π.keys():
                    w = π[k].weight
                    ws += w
                for k in π.keys():
                    counter+=1
                    level = π[k].leverage
                    w = π[k].weight
                    names = names + f"{k}_{level:.2f}_{w:.2f}_"
                    r_t = (
                            π[k].dataframe.capital 
                            / π[k].dataframe.capital.dropna().iloc[0]
                            - 1
                    ) * 100
                    rets_t = (
                            π[k].dataframe.capital 
                            / π[k].dataframe.capital.shift()
                            - 1
                    ) * 100
                    r_t = r_t.resample("1D").last().interpolate()
                    rets_t = rets_t.resample("1D").last().interpolate()
                    df = w/ws * r_t if df is None else df + w/ws * r_t
                    df_rets = w/ws * rets_t if df_rets is None else df_rets + w/ws * rets_t
                df = df.dropna()
                if len(df) < 2:
                    continue
                # r_t = r_t.resample("1h").last().interpolate()
                # T = 365.25 * 24 * 60 * 60
                # annualized_vol = r_t.apply(lambda col: col.dropna().std() * np.sqrt(dt))
                T = 365.25 * 24 * 60 * 60
                dt = 1
                scale = T / (df.dropna().index[-1] - df.dropna().index[0]).total_seconds()
                annualized_returns = df.iloc[-1] * scale #/ ws
                rets = (df_rets).dropna()
                rets = rets[np.isfinite(df_rets)]
                dt = T / rets.index.to_series().diff().dt.total_seconds().mean()
                dt = T / df.index.to_series().diff().dt.total_seconds().mean()
                annualized_vol = (df).std() * np.sqrt(dt)

                ROWS.append( [names[:-1], ws, counter, names[:-1], annualized_returns, annualized_vol] )
            dfs[π[k].name] = pd.DataFrame(ROWS, columns=["protocol", "w", "counter", "name", "mu", "sigma"])
            
            df = dfs[π[k].name]
            _filter = df.w >= .1
            _filter &= df.mu>0
            tmp = df[_filter]
            plt.figure(figsize=(5,5))

            ws = df.counter.unique()

            c = 2
            for w in tmp.w.unique():
                if w == 1: continue
                ws_filter = _filter & (tmp.counter == c) & (tmp.w > w)
                if len(df[ws_filter]) == 0: continue
                plt.scatter(df[ws_filter].sigma, df[ws_filter].mu, label=f"{c}:w={w:.2f}", alpha=0.4)
                
            c = 1
            for w in tmp.w.unique():
                ws_filter = _filter & (tmp.counter == c) & (tmp.w > w)
                if len(df[ws_filter]) == 0: continue
                plt.scatter(df[ws_filter].sigma, df[ws_filter].mu, label=f"{c}:w={w:.2f}", marker=".")

            w = 1
            c = 1
            ws_filter = _filter & (tmp.counter == c) & (tmp.w == w)
            plt.scatter(df[ws_filter].sigma, df[ws_filter].mu, label=f"{c}:w={w:.2f}", marker="X", s=100, alpha=0.7)

            plt.legend(
                #     handles=color_handles + marker_handles, 
                title="#Protocols : Weight", 
                loc='upper left',
                bbox_to_anchor=(1, 1)
            )
            plt.title(chain_name)
            plt.show()
            
            import plotly.express as px
            import plotly.graph_objects as go

            # Assuming df = dfs[π[k].name]
            _filter = (df.w >= 1) & (df.mu > 0)
            tmp = df[_filter]

            # Unique counters and weights
            # ws = df.counter.unique()

            fig = go.Figure()

            # Plot for counter 2
            c = 2
            for w in tmp.w.unique():
                if w == 1: 
                    continue
                ws_filter = _filter & (tmp.counter == c) & (tmp.w == w)
                if len(df[ws_filter]) == 0:
                    continue
                filtered_df = df[ws_filter]
                fig.add_trace(go.Scatter(
                    x=filtered_df.sigma,
                    y=filtered_df.mu,
                    mode='markers',
                    marker=dict(opacity=0.4),
                    name=f"{c}:w={w:.2f}",
                    hovertemplate="Volatility: %{x}<br>Returns: %{y}<br>Name: %{customdata[0]}",customdata=filtered_df[['protocol']].values
                ))

            # Plot for counter 1
            c = 1
            for w in tmp.w.unique():
                if w == 1:
                    continue
                ws_filter = _filter & (tmp.counter == c) & (tmp.w == w)
                if len(df[ws_filter]) == 0:
                    continue
                filtered_df = df[ws_filter]
                fig.add_trace(go.Scatter(
                    x=filtered_df.sigma,
                    y=filtered_df.mu,
                    mode='markers',
                    marker=dict(symbol='circle'),
                    name=f"{c}:w={w:.2f}",
                    hovertemplate="Volatility: %{x}<br>Returns: %{y}<br>Name: %{customdata[0]}",customdata=filtered_df[['protocol']].values
                ))

            # Highlight for w = 1 and c = 1
            w = 1
            c = 1
            ws_filter = _filter & (tmp.counter == c) & (tmp.w == w)
            if len(df[ws_filter]) > 0:
                filtered_df = df[ws_filter]
                fig.add_trace(go.Scatter(
                    x=filtered_df.sigma,
                    y=filtered_df.mu,
                    mode='markers',
                    marker=dict(symbol='x', size=15, opacity=0.7),
                    name=f"{c}:w={w:.2f}",
                    # hovertemplate="Sigma: %{x}<br>Mu: %{y}<br>Counter: 1<br>Weight: %{marker.size}"
                    hovertemplate="Volatility: %{x}<br>Returns: %{y}<br>Name: %{customdata[0]}",customdata=filtered_df[['protocol']].values
                ))

            # Update layout
            fig.update_layout(
                title=f"Chain: {chain_name}",
                xaxis_title="volatility",
                yaxis_title="returns",
                legend_title="#Protocols : Weight",
                legend=dict(orientation="h", x=0, y=-0.3),
                height=600,
                width=600
            )

            fig.show()

            break
        return dfs
