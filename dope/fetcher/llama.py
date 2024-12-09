import requests
import pandas as pd

from dope.names.poolname import PoolName

class Llama:
    def __init__(self):
        self.pools = None
        self.data = None
        self.borrow_lend_data = None

    def get_pools(self):
        url = "https://yields.llama.fi/pools"
        self.pools = pd.DataFrame(requests.get(url).json()["data"])
        return self.pools

    def lend_rate(self, pool_id):
        url = f"https://yields.llama.fi/chart/{pool_id}"
        result = requests.get(url)
        data = pd.DataFrame(result.json()["data"])
        # this line make sure all dates are at midnight
        data["datetime"] = pd.to_datetime(data.timestamp).apply(lambda x: x.date())
        # this gets back to datetime for rolling
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data.set_index("datetime")
        return data

    def borrow_lend(self, pool_id):
        url = f"https://yields.llama.fi/chartLendBorrow/{pool_id}"
        result = requests.get(url)
        df = pd.DataFrame(result.json()["data"])
        # this line make sure all dates are at midnight
        df["datetime"] = pd.to_datetime(df.timestamp).apply(lambda d: d.date())
        # this gets back to datetime for rolling
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")

        df["totalBorrowUsd"] = (
            df["totalBorrowUsd"].astype(float).interpolate(method="linear")
        )
        df["totalSupplyUsd"] = (
            df["totalSupplyUsd"].astype(float).interpolate(method="linear")
        )
        df["utilizationRate"] = df["totalBorrowUsd"] / df["totalSupplyUsd"]

        for col in ["apyReward", "apyRewardBorrow"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col]).fillna(0)

        return df

    def load_data_for_asset(
        self,
        asset_name,
        start_period,
        chain="Ethereum",
        tvl_cut=10_000_000,
    ):
        chain_name = chain.capitalize()
        asset = asset_name.upper()

        if self.pools is None:
            self.get_pools()

        poolids = self.pools[self.pools.symbol == asset].sort_values(
            "tvlUsd", ascending=False
        )  # [:20]
        poolids = poolids[poolids.tvlUsd >= tvl_cut]
        poolids = poolids[poolids.chain == chain_name]
        return self.load_data_from_pool_ids(poolids, start_period)

    def load_data_from_pool_ids(self, poolids, start_period):

        # start_period = pd.to_datetime("2023-06-01")
        start_period = pd.to_datetime(start_period)
        data = {}
        borrow_lend_data = {}
        for _, row in poolids.iterrows():
            _meta = row.poolMeta
            if _meta is not None:
                _meta = f"({_meta})".replace(":", "")
            else:
                _meta = ""
        
            _name = PoolName(row.chain, row.project + f"{_meta}", row.symbol, pool_id=row.pool)
            print(_name, end="\r")
            borrow_lend_data[_name] = self.borrow_lend(row.pool)
            _filter = borrow_lend_data[_name].index >= start_period
            borrow_lend_data[_name] = borrow_lend_data[_name][_filter]
            data[_name] = self.lend_rate(row.pool)
            data[_name] = data[_name][data[_name].index >= start_period]
            print(_name, row.pool, len(data[_name]), len(borrow_lend_data[_name]))

            data[_name]["utilizationRate"] = (
                borrow_lend_data[_name]["utilizationRate"].fillna(0.5).infer_objects()
            )
        self.data = data
        self.borrow_lend_data = borrow_lend_data
        return data, borrow_lend_data
