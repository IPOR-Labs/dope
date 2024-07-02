import pathlib
import itertools
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict

from dope.market_impact.linear import LinearMktImpactModel


class Chains(str, Enum):
  ethereum = "Ethereum"
  arbitrum = "Arbitrum"


@dataclass
class BacktestData:
  token_mkt_data: dict[str, dict[str, pd.DataFrame]]

  def __getitem__(self, key):
    return self.token_mkt_data[key]

  def keys(self):
    return self.token_mkt_data.keys()

  def items(self):
    return self.token_mkt_data.items()

  def to_block(self, token):
    return pd.concat(self.token_mkt_data[token], names=["datetime"]).unstack(level=0)

  def get_dates(self):
    dates = set()
    for token, data in self.token_mkt_data.items():
      for protocol, df in data.items():
        dates.update(df.index)
    dates = list(dates)
    dates.sort()
    return dates

  def apply_whitelist(self, whitelist): 
    """
    Usage example:

    whitelist = {
      "USDC": [
        #"makerdao:USDC",
        "aave-v3:USDC",
        "compound-v3:USDC",
        "aave-v2:USDC",
        "compound-v2:USDC"
        ],
      "wBTC": [
        "aave-v3:WBTC",
        "aave-v2:WBTC",
        "compound-v3:WBTC",
        "compound-v2:WBTC",
        #"makerdao:WBTC" 
      ],
      "USDT": [
        #"makerdao:USDC",
        "aave-v3:USDT",
        "compound-v3:USDT",
        "aave-v2:USDT",
        "compound-v2:USDT"
        ],
      "DAI": [
        #"makerdao:USDC",
        "aave-v3:DAI",
        "compound-v3:DAI",
        "aave-v2:DAI",
        "compound-v2:DAI"
        ],
    }
    """

    for t in self.token_mkt_data.keys():
      if t in whitelist:
        to_delete = [k for k in self.token_mkt_data[t].keys() if k not in whitelist[t]]
        for k in to_delete:
          del self.token_mkt_data[t][k]
    
  def apply_blacklist(self, blacklist):
    for t in self.token_mkt_data.keys():
      if t in blacklist:
        to_delete = [k for k in self.token_mkt_data[t].keys() if k in blacklist[t]]
        for k in to_delete:
          del self.token_mkt_data[t][k]
  
  def add_cash_mkt(self):
    for token in self.token_mkt_data.keys():
      mkts = list(self.token_mkt_data[token].keys())
      if len(mkts) == 0:
        return
      mkt_example = self.token_mkt_data[token][mkts[0]]
      self.token_mkt_data[token]["cash"] = mkt_example.copy(deep=True)
      self.token_mkt_data[token]["cash"]["apyBase"] = 0 
      self.token_mkt_data[token]["cash"]["apyBaseBorrow"] = np.inf
  
  def dump(self, filename, folderpath= pathlib.Path().home() / "s3/fusion/backtest-data"):
    pathlib.Path(folderpath / filename).mkdir(parents=True, exist_ok=True)
    for token, data in self.token_mkt_data.items():
      for mkt, df in data.items():
        df.to_csv(folderpath / filename / f"{token}_{mkt}.csv")
  
  @classmethod
  def load(cls, filename, folderpath= pathlib.Path().home() / "s3/fusion/backtest-data"):
    token_mkt_data = {}
    for file in pathlib.Path(folderpath / filename).iterdir():
      token, mkt = file.stem.split("_")
      if token not in token_mkt_data:
        token_mkt_data[token] = {}
      token_mkt_data[token][mkt] = pd.read_csv(file, index_col=0)
      token_mkt_data[token][mkt].index = pd.to_datetime(token_mkt_data[token][mkt].index)
    return cls(token_mkt_data)


class DataLoader:

  @classmethod
  def load_from_defi_llama(self, tokens, chain, tvl_cut=10_000_000, start_period="2023-06-01"):
    from dope.fetcher.llama import Llama
    llama = Llama()
    
    data, borrow_lend_data = {}, {}

    for token in tokens[:]:
      print(token)
      data[token], borrow_lend_data[token] = llama.load_data_for_asset(
        token,
        start_period=start_period,
        tvl_cut=tvl_cut,
        chain=chain
      )    
    return data, borrow_lend_data
  
  @classmethod
  def load_from_token_pool_id_dict(cls, token_pool_id_dict, start_period="2023-06-01"):
    from dope.fetcher.llama import Llama
    llama = Llama()
    pools = llama.get_pools()
    
    data, borrow_lend_data = {}, {}
    for token, pool_ids in token_pool_id_dict.items():
      print(token)
      data[token], borrow_lend_data[token] = llama.load_data_from_pool_ids(pools[pools.pool.isin(pool_ids)], start_period=start_period)
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
        key, value = item.split('=')
        categories[key].append(value)

    # Generate combinations
    keys, values = zip(*categories.items())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

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
      data[token], borrow_lend_data[token] = llama.load_data_from_pool_ids(tmp[tmp.symbol == token], start_period=start_period)
    
    return data, borrow_lend_data



class BackEngineMaestro:
  chains = Chains
  
  def __init__(self, tokens, chain):
    self.tokens = tokens
    self.chain = chain

  def set_data(self, borrow_lend_data: BacktestData):
    self.borrow_lend_data = borrow_lend_data

  def load_data(self, source="defi_llama", tvl_cut=10_000_000, start_period="2023-06-01"):
    if "lama" in source.lower():
      print("Loading data from DeFi Llama")
      data, borrow_lend_data = DataLoader.load_from_defi_llama(self.tokens, self.chain, tvl_cut=tvl_cut, start_period=start_period)

    return data, BacktestData(borrow_lend_data)
  
  
  def load_data_from_defilama_query(self, query, source="defi_llama", start_period="2023-06-01"):
    if "lama" in source.lower():
      print("Loading data from DeFi Llama")
      data, borrow_lend_data = DataLoader.load_from_defi_llama_query(query=query, start_period=start_period)

    return data, BacktestData(borrow_lend_data)
  
  def load_defilama_with_token_pool_id(self, token_pool_id_dict, source="defi_llama", start_period="2023-06-01"):
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
    data, borrow_lend_data = DataLoader.load_from_token_pool_id_dict(token_pool_id_dict, start_period=start_period)

    return data, BacktestData(borrow_lend_data)

  def estimate_mkt_impact_model(self, kinks = None, should_plot=False):

    mkt_impact = {"cash":LinearMktImpactModel.zero_instance()}
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
        y = _df["apyBase"].values#/100
        y = y[-CUT:]

        model = LinearMktImpactModel(x, y).fit(kinks=kinks[mkt], should_plot=True)
        model.set_data_ref(df)
        mkt_impact[mkt] = model
        if should_plot:
          plt.title(mkt)
          ax = plt.gca()
          ax.spines['top'].set_visible(False)
          ax.spines['right'].set_visible(False)
          plt.show()
    return mkt_impact
  
  # def run(self, strategy, mkt_impact, capital=1_000_000, start_period="2023-06-01", end_period="2024-07-01"):
  #   from dope.arbengine import ArbEngine
  #   engine = ArbEngine(strategy, self.borrow_lend_data, self.data, mkt_impact)
  #   summary, ws = engine()
  #   return summary, ws

  def plot_rates_ts(self, title=None, agg_str="1D", rate_column="apyBase"):
    tokens = list(self.borrow_lend_data.keys())
    n_tokens = len(tokens)
    fig, axes = plt.subplots(n_tokens, 1, figsize=(20, 5 * n_tokens)) 
    # Check if there is only one token, axes will not be an array but a single AxesSubplot object
    if n_tokens == 1:
        axes = [axes]  # Make it iterable

    for ax, token in zip(axes, tokens):
        # Aggregate the data
        agg = pd.concat(self.borrow_lend_data[token], names=["datetime"]).unstack(level=0)[rate_column]
        # Calculate the rolling mean and plot
        agg.rolling(agg_str).mean().plot(ax=ax)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Protocols")
        ax.set_title(title or f"[{token}] APY {agg_str} days moving average")
        # Hide the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("APY")
        ax.set_xlabel("")
    plt.xlabel("date")

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()  # Show all plots in one figure

  def plotly_rates_ts(self, title=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    n_tokens = len(self.tokens)
    
    # Create a subplot figure with n rows and 1 column
    fig = make_subplots(rows=n_tokens, cols=1, shared_xaxes=True, subplot_titles=[f"[{token}] APY 7 days moving average" for token in self.tokens])
    
    for i, token in enumerate(self.tokens, start=1):
        # Aggregate the data
        agg = pd.concat(self.borrow_lend_data[token], names=["datetime"]).unstack(level=0).apyBaseBorrow
        # Calculate the rolling mean
        rolling_mean = agg.rolling("7D").mean()

        # Create a line plot
        for column in rolling_mean.columns:
            fig.add_trace(
                go.Scatter(x=rolling_mean.index, y=rolling_mean[column], mode='lines', name=column),
                row=i, col=1
            )

    # Update layout for the figure
    fig.update_layout(
        height=400 * n_tokens,
        title_text=title or "APY 7 days moving average",
        showlegend=True
    )

    # Update xaxis and yaxis labels
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="APY")
    
    fig.show()
