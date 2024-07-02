import pathlib
import pandas as pd
from dataclasses import dataclass

from dope.backengine.maestro import BacktestData

@dataclass
class BacktestSummary:
  name: str
  summary:  dict[str, pd.DataFrame]
  run_data: BacktestData
  
  def dump(self, filename=None, folderpath=pathlib.Path().home() / "s3/fusion/backtest-data"):
    filename = filename or self.name
    pathlib.Path(folderpath / filename).mkdir(parents=True, exist_ok=True)
    for mkt, df in self.summary.items():
      df.to_parquet(folderpath / filename / f"{mkt}.parquet")
    self.run_data.dump(f"{filename}/run_data", folderpath)
  
  @classmethod
  def load(cls, filename, folderpath=pathlib.Path().home() / "s3/fusion/backtest-data"):
    summary = {}
    for file in pathlib.Path(folderpath / filename).glob('*.parquet'):
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