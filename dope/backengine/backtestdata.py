import pathlib
import warnings
import numpy as np
import pandas as pd

from dope.names.poolname import PoolName

from dataclasses  import dataclass

@dataclass
class ColumnsTranslation:
    apyBaseBorrow: str
    apyBase: str
    totalSupplyUsd: str
    totalBorrowUsd: str
    datetime:str = None
    utilizationRate: str = None
    
    def translate(self, df):
        df = df.copy()
        columns={
            self.apyBaseBorrow:"apyBaseBorrow",
            self.apyBase:"apyBase",
            self.totalSupplyUsd:"totalSupplyUsd",
            self.totalBorrowUsd:"totalBorrowUsd"
        }
        if self.utilizationRate is not None:
            columns[self.utilizationRate] = "utilizationRate"
        
        if (self.datetime == "index") or (self.datetime is None):
            # if datetime is not passed, we assume it is the index
            df = df.rename_axis("datetime")
        else:
            columns[self.datetime] = "datetime"
        df = df.rename(columns=columns)
        if self.utilizationRate is None:
            df["utilizationRate"] = df["totalBorrowUsd"] / df["totalSupplyUsd"]
        df.sort_index(inplace=True, ascending=True)
        return df


class BacktestData:

    def __init__(self, token_mkt_data: dict[str, dict[str, pd.DataFrame]]):
        self.token_mkt_data = token_mkt_data

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
        warnings.simplefilter(action="ignore", category=FutureWarning)
        ret_df = pd.concat(
            self.token_mkt_data[token], names=["datetime"]
        ).unstack(level=0)
        warnings.simplefilter("default")

        return ret_df
    
    def get_pool_df(self, pool_name):
        for token in self.token_mkt_data.keys():
            if pool_name in self.token_mkt_data[token]:
                return self.token_mkt_data[token][pool_name]
        raise ValueError(f"Pool {pool_name} not found in data.")

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
                continue
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


class DataCollection:
    
    def __init__(self, name, collection=None):
        self.name = name
        self.collection = collection or {}
    
    def __getitem__(self, key):
        return self.collection[key]
    
    def add(self, key, data):
        self.collection[key] = data
    
    def update(self, data):
        self.collection.update(data)
    
    def keys(self):
        return self.collection.keys()

    def items(self):
        return self.collection.items()

    def copy(self):
        """
        Return a copy of this object
        """
        copy_collection = {}
        for key, df in self.collection.items():
            copy_collection[key] = df.copy()
        return DataCollection(name=self.name, collection=copy_collection)
        

    def id_isin(self, key):
        for k in self.collection.keys():
            if key in k.pool_id:
                return True
        return False

    def as_block(self):
        warnings.simplefilter(action="ignore", category=FutureWarning)
        ret_df = pd.concat(
            self.collection, names=["datetime"]
        ).unstack(level=0)
        warnings.simplefilter("default")
        return ret_df
    
    def convert_tvl_from_usd(self, token_price_in_usd_timeseries, inplace=False):
        this = self if inplace else self.copy()


        price_index = token_price_in_usd_timeseries.index
        for mkt in this.collection.keys():
            mkt_index = this.collection[mkt].index
            mkt_price = token_price_in_usd_timeseries[
                token_price_in_usd_timeseries.index.isin(mkt_index)
            ].copy()
            for c in ["totalSupplyUsd", "totalBorrowUsd"]:
                this.collection[mkt][c] = (
                    this.collection[mkt][c] / mkt_price
                )

        return this

    def __repr__(self):
        keys_str = ", ".join([str(k) for k in self.collection.keys()])
        return f"RatesDataCollection({len(self.collection)} items -- {keys_str[:100]}...)"

    def dump(
        self, filename, folderpath=pathlib.Path().home() / "s3/fusion/backtest-data"
    ):
        folderpath = pathlib.Path(folderpath)
        pathlib.Path(folderpath / filename).mkdir(parents=True, exist_ok=True)
        # delete data already in folder to avoid overlapping old and new data:
        for file in (folderpath / filename).iterdir():
            if file.is_file():
                file.unlink()

        for mkt, df in self.collection.items():
            if isinstance(mkt, PoolName):
                mkt_name = mkt.to_disk_name()
            else:
                mkt_name = mkt
            df.to_csv(folderpath / filename / f"{self.name}_{mkt_name}.csv")

    @classmethod
    def load(
        cls,
        filename,
        folderpath=pathlib.Path().home() / "s3/fusion/backtest-data",
        verbose=False,
    ):
        folderpath = pathlib.Path(folderpath)
        collection = {}
        for file in pathlib.Path(folderpath / filename).iterdir():
            token_divisor = file.stem.find('_')
            name, mkt_name = file.stem[:token_divisor], file.stem[token_divisor+1:]
            mkt = PoolName.try_from_disk_name(mkt_name, verbose=verbose)
            
            collection[mkt] = pd.read_csv(file, index_col=0)
            collection[mkt].index = pd.to_datetime(collection[mkt].index)
        return cls(name=name, collection=collection)

    def translate(self, translation: ColumnsTranslation, inplace=False):
        this = self if inplace else self.copy()
        for mkt in this.collection.keys():
            this.collection[mkt] = translation.translate(this.collection[mkt])
        return this


class PriceRow:
    def __init__(self, row):
        self.row = row

    def __repr__(self):
        line = f"Price @ {self.row.name} = "
        for k, v in self.row.items():
            line += f"{k}: {v:.4f} | "
        line = line[:-2]
        return line

    def get_or_zero(self, token):
        if token in self.row:
            return self.row[token]
        else:
            return 0
    
    def get(self, token):
        return self.row[token]


class PriceCollection(DataCollection):
    
    def __init__(self, name, collection=None, base_token_name="dollar"):
        super().__init__(name=name, collection=collection)
        self.base_token_name = base_token_name
    
    def set_base_token_name(self, base_token_name):
        self.base_token_name = base_token_name
    
    def set_up_price_timeseries(self):
        warnings.simplefilter(action="ignore", category=FutureWarning)
        self._price = (
            pd.concat(self.collection).unstack(level=0).price.reset_index()
        )
        self._price["dollar"] = 1
        warnings.simplefilter("default")
        self._price["date"] = pd.to_datetime(self._price["date"].dt.date)
        self._price = self._price.groupby("date").mean()
        self._price = self._price.apply(lambda x: x/self._price[self.base_token_name])
        return self._price

    def price_row_at(self, date_ix):
        row = self._price.loc[date_ix]
        price_row = PriceRow(row)
        return price_row
