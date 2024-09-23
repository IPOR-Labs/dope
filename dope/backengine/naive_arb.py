import pandas as pd


class OneMktStrategy:
    def __init__(self, pay_token, rec_token, pay_protocol=None, rec_protocol=None):
        self.pay_token = pay_token
        self.rec_token = rec_token
        self.engine = None
        self.pay_protocol = pay_protocol
        self.rec_protocol = rec_protocol

    def set_engine(self, engine):
        self.engine = engine

    @property
    def borrow_lend_data(self):
        return self.engine.borrow_lend_data

    def act(self):

        pay_token = self.pay_token
        rec_token = self.rec_token

        agg = pd.concat(self.borrow_lend_data[pay_token], names=["datetime"]).unstack(
            level=0
        )
        whitelist = (
            [self.pay_protocol] if self.pay_protocol else agg.apyBaseBorrow.columns
        )
        borrow = agg.apyBaseBorrow[whitelist].min(axis=1)
        borrow_pool = agg.apyBaseBorrow.apply(pd.to_numeric, errors="coerce").idxmin(
            axis=1
        )

        agg = pd.concat(self.borrow_lend_data[rec_token], names=["datetime"]).unstack(
            level=0
        )
        whitelist = (
            [self.rec_protocol] if self.pay_protocol else agg.apyBaseBorrow.columns
        )
        lend = agg.apyBase[whitelist].max(axis=1)
        lend_pool = agg.apyBase.apply(pd.to_numeric, errors="coerce").idxmax(axis=1)

        return borrow, lend, borrow_pool, lend_pool


class NaiveArbBacktest:

    def __init__(self, borrow_lend_data):
        self.borrow_lend_data = borrow_lend_data

    def run(self, pay_token, rec_token, pay_protocol=None, rec_protocol=None):
        strategy = OneMktStrategy(pay_token, rec_token, pay_protocol, rec_protocol)
        strategy.set_engine(self)
        self.borrow, self.lend, self.borrow_pool, self.lend_pool = strategy.act()
        self.pnl = self.lend / 100 / 365 - self.borrow / 100 / 365 + 1
