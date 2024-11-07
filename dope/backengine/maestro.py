import warnings
import itertools
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict


from dope.market_impact.linear import LinearMktImpactModel
from dope.market_impact.neighborhood import NeighborhoodMktImpactModel
from dope.backengine.backtestdata import BacktestData, DataCollection, PriceCollection
from dope.fetcher.coingecko import CoinGecko
from dope.names.poolname import PoolName
from dope.pools.pools import Pool


class Chains(str, Enum):
    ethereum = "Ethereum"
    arbitrum = "Arbitrum"


class PriceRow:
    def __init__(self, row, price_dict):
        self.row = row
        self.price_dict = price_dict

    def __repr__(self):
        line = f"Price @ {self.row.name} = "
        for k, v in self.row.items():
            line += f"{k}: {v:.4f} | "
        line = line[:-2]
        return line

    def get_or_zero(self, protocol):
        if protocol in self.price_dict:
            # print("PRICE:::::",protocol, self.row[self.price_dict[protocol]])
            return self.row[self.price_dict[protocol]]
        else:
            return 0


class PriceData:
    def __init__(self, token_price_dict: dict[str, pd.DataFrame]):
        """
        Example
        price_data = PriceData({"ETH":eth, "STETH":steth})
        protocol_token_name_dict = {
          # Pool Token : Price Token
          "WETH": "ETH",
          "WSTETH":"STETH",
          "STETH":"STETH"
        }

        price_data.set_mkt_to_token(run_data, protocol_token_name_dict)
        """
        self.token_price_dict = token_price_dict
        self.price_dict = {}

        warnings.simplefilter(action="ignore", category=FutureWarning)
        self._price = (
            pd.concat(self.token_price_dict).unstack(level=0).price.reset_index()
        )
        warnings.simplefilter("default")
        self._price["date"] = pd.to_datetime(self._price.reset_index().date.dt.date)
        self._price = self._price.groupby("date").mean()

    def price_row_at(self, date_ix):
        row = self._price.loc[date_ix]
        price_row = PriceRow(row, self.price_dict)
        return price_row

    def set_mkt_to_token(self, run_data, mkt_token_dict):
        price_dict = {}
        for mkt in run_data.get_markets():
            for k, v in mkt_token_dict.items():                
                if isinstance(mkt, str) and f":{k}" in mkt:
                    price_dict[mkt] = v
                    break
                elif isinstance(mkt, PoolName) and mkt.token_name == k:
                    price_dict[mkt] = v
                    break
        self.price_dict = price_dict


class DataLoader:
    
    def __init__(self):
        from dope.fetcher.llama import Llama
        self.llama = Llama()
        self.pools_df = None

    def load_one_from_defi_llama(self, pool_id,  start_period="2023-06-01"):
        if self.pools_df is None:
            self.pools_df = self.llama.get_pools()
        pools = self.pools_df
        
        data, borrow_lend_data = self.llama.load_data_from_pool_ids(
            pools[pools.pool.isin([pool_id])], start_period=start_period
        )
        key = list(borrow_lend_data.keys())[0]
        return key, borrow_lend_data[key]

    @classmethod
    def load_from_defi_llama(
        self, tokens, chain, tvl_cut=10_000_000, start_period="2023-06-01"
    ):
        from dope.fetcher.llama import Llama
        llama = Llama()

        data, borrow_lend_data = {}, {}

        for token in tokens[:]:
            print(token)
            data[token], borrow_lend_data[token] = llama.load_data_for_asset(
                token, start_period=start_period, tvl_cut=tvl_cut, chain=chain
            )
        return data, borrow_lend_data

    @classmethod
    def load_from_token_pool_id_dict(
        cls, token_pool_id_dict, start_period="2023-06-01"
    ):
        from dope.fetcher.llama import Llama

        llama = Llama()
        pools = llama.get_pools()

        data, borrow_lend_data = {}, {}
        for token, pool_ids in token_pool_id_dict.items():
            print(token)
            data[token], borrow_lend_data[token] = llama.load_data_from_pool_ids(
                pools[pools.pool.isin(pool_ids)], start_period=start_period
            )
        return data, borrow_lend_data

    @classmethod
    def load_from_defi_llama_query(self, query, start_period="2023-06-01"):
        """
        query example:
        query = "https://defillama.com/yields?token=USDC&project=aave-v3&project=compound-v3&project=morpho-blue&chain=Optimism&chain=Arbitrum&chain=Base&chain=Polygon&chain=Avalanche"
        """
        from dope.fetcher.llama import Llama

        llama = Llama()

        input_list = [v for v in query.split("?")[1].split("&")]

        # Extract categories and values
        categories = defaultdict(list)

        for item in input_list:
            key, value = item.split("=")
            categories[key].append(value)

        # Generate combinations
        keys, values = zip(*categories.items())
        combinations = [
            dict(zip(keys, combination)) for combination in itertools.product(*values)
        ]

        pools = llama.get_pools()

        _filter = np.array([False for _ in pools.iterrows()])
        for combo in combinations:
            _filter_tmp = np.array([True for _ in pools.iterrows()])
            for k, v in combo.items():
                k = k.replace("token", "symbol")
                _filter_tmp &= pools[k] == v
            _filter |= _filter_tmp

        data, borrow_lend_data = {}, {}
        tmp = pools[_filter]
        for token in tmp.symbol.unique():
            print(token)
            data[token], borrow_lend_data[token] = llama.load_data_from_pool_ids(
                tmp[tmp.symbol == token], start_period=start_period
            )

        return data, borrow_lend_data


class BackEngineMaestro:
    chains = Chains

    def __init__(self):
        self.data_loader = DataLoader()
        self.borrow_lend_data = None
        self.pools = {}

        self.base_token = "dollar"
        self.rates_data_in_base_token = None
        self.rates_data_collection = DataCollection(name="rates")
        self.price_data_collection = PriceCollection(name="prices")
        
        
    
    def load_pool_data(self, pool: Pool):
        if pool.debt_pool_id is not None:
            if not self.rates_data_collection.id_isin(pool.debt_pool_id):
                pool_name_debt, borrow_lend_data = self.data_loader.load_one_from_defi_llama(pool.debt_pool_id)
                self.rates_data_collection.add(pool_name_debt, borrow_lend_data)
            else:
                for k in self.rates_data_collection.collection.keys():
                    if pool.debt_pool_id in k.pool_id:
                        pool_name_debt = k
                # print("pool_name_debt", pool_name_debt)
            pool.debt_rate_keyid = pool_name_debt
            
        if pool.deposit_pool_id is not None:
            if not self.rates_data_collection.id_isin(pool.deposit_pool_id):
                pool_name_deposit, borrow_lend_data = self.data_loader.load_one_from_defi_llama(pool.deposit_pool_id)
                self.rates_data_collection.add(pool_name_deposit, borrow_lend_data)
                
            else:
                for k in self.rates_data_collection.collection.keys():
                    if pool.deposit_pool_id in k.pool_id:
                        pool_name_deposit = k
                # print("pool_name_deposit", pool_name_deposit)
            pool.deposit_rate_keyid = pool_name_deposit
        return pool
    
    def load_pools_data(self, pools: list[Pool]):
        for p in pools:
            pool = self.load_pool_data(p)
            self.pools[pool] = pool
        return self.pools
    
    def load_price_data(self, pools: list[Pool]):
        
        cg = CoinGecko()
        for p in pools:
            if p.deposit_token_keyid is None:
                continue
            if p.deposit_token not in self.price_data_collection.collection:
                price_data = cg.get_last_year_price(p.deposit_token_keyid)
                self.price_data_collection.add(p.deposit_token, price_data)
        for p in pools:
            if p.debt_token_keyid is None:
                continue
            if p.debt_token not in self.price_data_collection.collection:
                price_data = cg.get_last_year_price(p.debt_token_keyid)
                self.price_data_collection.add(p.debt_token, price_data)
    
    def set_base_token(self, token_name):
        self.base_token = token_name
    
    def convert_data_to_base_token(self, base_token=None):
        if base_token is not None:
            self.set_base_token(base_token)
        self.price_data_collection.set_base_token_name(self.base_token)
        self.price_data_collection.set_up_price_timeseries()
        
        price_timeseries = self.price_data_collection._price["dollar"]
        self.rates_data_in_base_token = self.rates_data_collection.convert_tvl_from_usd(1/price_timeseries)

    def set_data(self, borrow_lend_data: BacktestData):
        self.borrow_lend_data = borrow_lend_data

    def load_data(
        self,
        tokens: list,
        source="defi_llama",
        tvl_cut=10_000_000,
        start_period="2023-06-01",
        chain=None,
    ):
        if chain is not None:
            chain = self.chains.ethereum
        if "lama" in source.lower():
            print("Loading data from DeFi Llama")
            data, borrow_lend_data = DataLoader.load_from_defi_llama(
                tokens, chain, tvl_cut=tvl_cut, start_period=start_period
            )

        return data, BacktestData(borrow_lend_data)

    def load_data_from_defilama_query(
        self, query, source="defi_llama", start_period="2023-06-01"
    ):
        if "lama" in source.lower():
            print("Loading data from DeFi Llama")
            data, borrow_lend_data = DataLoader.load_from_defi_llama_query(
                query=query, start_period=start_period
            )

        return data, BacktestData(borrow_lend_data)

    def load_defilama_with_token_pool_id(
        self, token_pool_id_dict, source="defi_llama", start_period="2023-06-01"
    ):
        """
        token_pool_id_dict = {
          "USDC":
          [
            "aa70268e-4b52-42bf-a116-608b370f9501", # Ethereum:aave-v3:USDC
            "7da72d09-56ca-4ec5-a45f-59114353e487", # Ethereum:compound-v3:USDC
            "cefa9bb8-c230-459a-a855-3b94e96acd8c", # Ethereum:compound-v2:USDC
            "a349fea4-d780-4e16-973e-70ca9b606db2", # Ethereum:aave-v2:USDC
          ]
        }
        """
        data, borrow_lend_data = DataLoader.load_from_token_pool_id_dict(
            token_pool_id_dict, start_period=start_period
        )

        return data, BacktestData(borrow_lend_data)

    def interpolate_data(self, borrow_lend_data, agg_str="1h"):
        """
        Interporlates the data to a short time frame.
        agg_str: str, the time frame to interpolate the data to. Default to "1h" (one hour),
          for other options see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """
        enhanced_borrow_lend_data = {t: {} for t in borrow_lend_data.keys()}
        for token in enhanced_borrow_lend_data.keys():
            for mkt in borrow_lend_data[token].keys():
                enhanced_borrow_lend_data[token][mkt] = (
                    borrow_lend_data[token][mkt]
                    .resample(agg_str)
                    .last()
                    .infer_objects(copy=False)
                    .apply(pd.to_numeric, errors="coerce")
                    .interpolate()
                )
        enhanced_borrow_lend_data = BacktestData(enhanced_borrow_lend_data)
        return enhanced_borrow_lend_data

    def data_collection_to_backtest_data(self, data_collection=None):
        data_collection = data_collection or self.rates_data_collection
        
        borrow_lend_data = {}

        for p in self.pools.values():
            if p.deposit_rate_keyid is not None:
                token = p.deposit_token
                if token not in borrow_lend_data:
                    borrow_lend_data[token] = {}
                borrow_lend_data[token][p.deposit_rate_keyid] = data_collection[p.deposit_rate_keyid]
            if p.debt_rate_keyid is not None:
                token = p.debt_token
                if token not in borrow_lend_data:
                    borrow_lend_data[token] = {}
                borrow_lend_data[token][p.debt_rate_keyid] = data_collection[p.debt_rate_keyid]
        return BacktestData(borrow_lend_data)

    def get_historical_mkt_impact_model(
        self, run_data, past_window_days=7, future_window_days=0
    ):
        mkt_impact = {
            mkt: NeighborhoodMktImpactModel(
                past_window_days=past_window_days, future_window_days=future_window_days
            )
            for mkt in run_data.get_markets()
        }
        mkt_impact["cash"] = LinearMktImpactModel.zero_instance()
        return mkt_impact

    def estimate_mkt_impact_model(self, kinks=None, should_plot=False):

        mkt_impact = {"cash": LinearMktImpactModel.zero_instance()}
        kinks = kinks or defaultdict(lambda: None)
        # kinks.update({
        #   "aave-v3":[0.9], "spark":[0.7], "aave-v2":[0.9],
        #   "compound-v2":[0.625], "compound-v3":[0.93]
        # })

        for token in self.borrow_lend_data.keys():
            for mkt, df in self.borrow_lend_data[token].items():
                _df = df[["utilizationRate", "apyBase"]].dropna()
                print(mkt, len(_df))
                if len(_df) == 0:
                    mkt_impact[mkt] = LinearMktImpactModel.zero_instance()
                    continue
                CUT = 90
                x = (_df["utilizationRate"]).values
                x = x[-CUT:]
                y = _df["apyBase"].values  # /100
                y = y[-CUT:]

                model = LinearMktImpactModel(x, y).fit(
                    kinks=kinks[mkt], should_plot=True
                )
                model.set_data_ref(df)
                mkt_impact[mkt] = model
                if should_plot:
                    plt.title(mkt)
                    ax = plt.gca()
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    plt.show()
        return mkt_impact

    # def run(self, strategy, mkt_impact, capital=1_000_000, start_period="2023-06-01", end_period="2024-07-01"):
    #   from dope.arbengine import ArbEngine
    #   engine = ArbEngine(strategy, self.borrow_lend_data, self.data, mkt_impact)
    #   summary, ws = engine()
    #   return summary, ws

    def plot_rates_ts(
        self, 
        borrow_lend_data=None,
        title=None, 
        agg_str="1D",
        rate_column="apyBase",
        figsize_xy=(20, 5)
    ):
        if borrow_lend_data is None:
            borrow_lend_data = self.borrow_lend_data
        elif isinstance(borrow_lend_data, DataCollection):
            borrow_lend_data = self.data_collection_to_backtest_data(borrow_lend_data)
        
        tokens = list(borrow_lend_data.keys())
        n_tokens = len(tokens)
        fig, axes = plt.subplots(
            n_tokens, 1, figsize=(figsize_xy[0], figsize_xy[1] * n_tokens)
        )
        # Check if there is only one token, axes will not be an array but a single AxesSubplot object
        if n_tokens == 1:
            axes = [axes]  # Make it iterable

        for ax, token in zip(axes, tokens):
            # Aggregate the data
            agg = borrow_lend_data.as_block(token)[rate_column]
            # Calculate the rolling mean and plot
            agg.rolling(agg_str).mean().plot(ax=ax)

            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Protocols")
            ax.set_title(title or f"[{token}] APY {agg_str} days moving average")
            # Hide the top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylabel("APY")
            ax.set_xlabel("")
        plt.xlabel("date")

        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.show()  # Show all plots in one figure

    def plotly_rates_ts(self, title=None):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        n_tokens = len(self.tokens)

        # Create a subplot figure with n rows and 1 column
        fig = make_subplots(
            rows=n_tokens,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[
                f"[{token}] APY 7 days moving average" for token in self.tokens
            ],
        )

        for i, token in enumerate(self.tokens, start=1):
            # Aggregate the data
            agg = self.borrow_lend_data.as_block(token).apyBaseBorrow
            # Calculate the rolling mean
            rolling_mean = agg.rolling("7D").mean()

            # Create a line plot
            for column in rolling_mean.columns:
                fig.add_trace(
                    go.Scatter(
                        x=rolling_mean.index,
                        y=rolling_mean[column],
                        mode="lines",
                        name=column,
                    ),
                    row=i,
                    col=1,
                )

        # Update layout for the figure
        fig.update_layout(
            height=400 * n_tokens,
            title_text=title or "APY 7 days moving average",
            showlegend=True,
        )

        # Update xaxis and yaxis labels
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="APY")

        fig.show()
