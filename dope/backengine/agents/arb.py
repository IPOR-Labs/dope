import pandas as pd


class ArbStrategy:
  def __init__(self, capital, pay_token, rec_token, pay_protocol=None, rec_protocol=None):
    self.capital = capital
    self.pay_token = pay_token
    self.rec_token = rec_token
    self.engine = None
    self.pay_protocol = pay_protocol
    self.rec_protocol = rec_protocol

  
  def register_engine(self, engine):
    self.engine = engine
  
  @property
  def borrow_lend_data(self):
    return self.engine.data
  
  def act(self, date_ix):

    pay_token = self.pay_token
    rec_token = self.rec_token


    agg_pay = pd.concat(self.borrow_lend_data[pay_token], names=["datetime"]).unstack(level=0)
    agg_pay = agg_pay.loc[:date_ix]
    agg_rec = pd.concat(self.borrow_lend_data[rec_token], names=["datetime"]).unstack(level=0)
    agg_rec = agg_rec.loc[:date_ix]

    whitelist = [self.pay_protocol] if self.pay_protocol else agg_pay.apyBaseBorrow.columns
    borrow_mkt = agg_pay.apyBaseBorrow[whitelist].rolling("1D").mean().idxmin(axis=1).iloc[-1]

    whitelist = [self.rec_protocol] if self.pay_protocol else agg_rec.apyBaseBorrow.columns
    lend_mkt = agg_rec.apyBase[whitelist].rolling("1D").mean().idxmax(axis=1).iloc[-1]
    

    return {
      pay_token:
      {borrow_mkt: -1},
      rec_token:
      {lend_mkt: 1}
      }
