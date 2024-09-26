import pathlib
import warnings
import numpy as np
import pandas as pd

from dope.names.poolname import PoolName


class BacktestData:

    def __init__(self, token_mkt_data: dict[str, dict[str, pd.DataFrame]]):
        self.token_mkt_data = token_mkt_data
        self._as_block = {}

    def reset_as_block(self):
        self._as_block = {}

    def __getitem__(self, key):
        return self.token_mkt_data[key]

    def keys(self):
        return self.token_mkt_data.keys()

    def values(self):
        return self.token_mkt_data.values()

    def get_markets(self):
        markets = set()
        for token in self.token_mkt_data.keys():
            markets.update(self.token_mkt_data[token].keys())
        return markets

    def items(self):
        return self.token_mkt_data.items()

    def as_block(self, token):
        if token not in self._as_block:
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self._as_block[token] = pd.concat(
                self.token_mkt_data[token], names=["datetime"]
            ).unstack(level=0)
            warnings.simplefilter("default")
        return self._as_block[token]

    def to_block(self, token):
        return self.as_block(token)

    def convert_tvl_from_usd(self, token_price_in_usd_timeseries, inplace=False):
        this = self if inplace else self.copy()

        price_index = token_price_in_usd_timeseries.index
        for token in this.token_mkt_data.keys():
            for mkt in this.token_mkt_data[token].keys():
                mkt_index = this.token_mkt_data[token][mkt].index
                # this.token_mkt_data[token][mkt] = this.token_mkt_data[token][mkt][mkt_index.isin(price_index)].copy()
                mkt_price = token_price_in_usd_timeseries[
                    token_price_in_usd_timeseries.index.isin(mkt_index)
                ].copy()
                for c in ["totalSupplyUsd", "totalBorrowUsd"]:
                    this.token_mkt_data[token][mkt][c] = (
                        this.token_mkt_data[token][mkt][c] / mkt_price.price
                    )

        this.reset_as_block()

        return this

    def copy(self):
        copy_token_mkt_data = {}
        for token, data in self.token_mkt_data.items():
            copy_token_mkt_data[token] = {}
            for mkt, df in data.items():
                copy_token_mkt_data[token][mkt] = df.copy()
        return BacktestData(copy_token_mkt_data)

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
                to_delete = [
                    k for k in self.token_mkt_data[t].keys() if k not in whitelist[t]
                ]
                for k in to_delete:
                    del self.token_mkt_data[t][k]

    def apply_blacklist(self, blacklist):
        for t in self.token_mkt_data.keys():
            if t in blacklist:
                to_delete = [
                    k for k in self.token_mkt_data[t].keys() if k in blacklist[t]
                ]
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

    def dump(
        self, filename, folderpath=pathlib.Path().home() / "s3/fusion/backtest-data"
    ):
        folderpath = pathlib.Path(folderpath)
        pathlib.Path(folderpath / filename).mkdir(parents=True, exist_ok=True)
        # delete data already in folder to avoid overlapping old and new data:
        for file in (folderpath / filename).iterdir():
            if file.is_file():
                file.unlink()

        for token, data in self.token_mkt_data.items():
            for mkt, df in data.items():
                if isinstance(mkt, PoolName):
                    mkt_name = mkt.to_disk_name()
                else:
                    mkt_name = mkt
                df.to_csv(folderpath / filename / f"{token}_{mkt_name}.csv")

    @classmethod
    def load(
        cls,
        filename,
        folderpath=pathlib.Path().home() / "s3/fusion/backtest-data",
        verbose=False,
    ):
        folderpath = pathlib.Path(folderpath)
        token_mkt_data = {}
        for file in pathlib.Path(folderpath / filename).iterdir():
            token_divisor = file.stem.find('_')
            token, mkt_name = file.stem[:token_divisor], file.stem[token_divisor+1:]
            mkt = PoolName.try_from_disk_name(mkt_name, verbose=verbose)
            if token not in token_mkt_data:
                token_mkt_data[token] = {}
            token_mkt_data[token][mkt] = pd.read_csv(file, index_col=0)
            token_mkt_data[token][mkt].index = pd.to_datetime(
                token_mkt_data[token][mkt].index
            )
        return cls(token_mkt_data)