import pathlib
import pandas as pd
from dataclasses import dataclass

from dope.backengine.maestro import BacktestData


def serialize_ws(row):
  ret_dict = {}
  for token, ddict in row.items():
    ret_dict[token] = {}
    for k, df in ddict.items():
      ret_dict[token][str(k)] = df
  return ret_dict


def serialize(row):
  ret_dict = {}
  for k, df in row.items():
    ret_dict[str(k)] = df
  return ret_dict


@dataclass
class BacktestSummary:
    name: str
    summary: dict[str, pd.DataFrame]
    run_data: BacktestData

    def dump(
        self,
        filename=None,
        folderpath=pathlib.Path().home() / "s3/fusion/backtest-data",
    ):
        folderpath = pathlib.Path(folderpath)
        filename = filename or self.name
        pathlib.Path(folderpath / filename).mkdir(parents=True, exist_ok=True)
        for mkt, df in self.summary.items():
            tmp = df.copy(deep=True)
            tmp["ws"] = tmp.ws.apply(serialize_ws)
            for col in ["mkt_impact", "breakdown"]:
                tmp[col] = tmp.ws.apply(serialize)
            tmp.to_parquet(folderpath / filename / f"{mkt}.parquet")
        self.run_data.dump(f"{filename}/run_data", folderpath)

    @classmethod
    def load(
        cls, filename, folderpath=pathlib.Path().home() / "s3/fusion/backtest-data"
    ):
        folderpath = pathlib.Path(folderpath)
        summary = {}
        for file in pathlib.Path(folderpath / filename).glob("*.parquet"):
            if not file.is_file():
                continue

            strategy_name = file.name.replace(".parquet", "")
            summary[strategy_name] = pd.read_parquet(file)
            summary[strategy_name].index = pd.to_datetime(summary[strategy_name].index)
        run_data = BacktestData.load(f"{filename}/run_data", folderpath)
        return cls(filename, summary, run_data)

    def __getitem__(self, key):
        return self.summary[key]

    def keys(self):
        return self.summary.keys()

    def items(self):
        return self.summary.items()

    def to_block(self):
        return pd.concat(self.summary, names=["datetime"]).unstack(level=0)
