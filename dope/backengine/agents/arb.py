import numpy as np
import pandas as pd


class ArbStrategy:
    def __init__(
        self,
        token,
        capital,
        profitability_threshold=-np.inf,
        past_window_in_minutes=5,
        triggers=None,
    ):
        self.capital = capital
        self.profitability_threshold = profitability_threshold
        self.past_window_in_minutes = past_window_in_minutes
        self.triggers = triggers
        self.token = token
        self.engine = None
        

    def register_engine(self, engine):
        self.engine = engine

    @property
    def borrow_lend_data(self):
        return self.engine.data

    def on_start(self, date_ix):
        # self.engine.deposit_to_wallet(self.token, self.capital)
        self.ws = {}
        return self.ws

    def on_act(self, date_ix):
        if self.triggers is not None:
            if date_ix not in self.triggers:
                return self.ws
        self.now_data = self.engine.block_data_up_to_now(self.token, date_ix)
        
        
        apyBase = self.now_data.apyBase
        apyBase = apyBase.resample(f"{self.past_window_in_minutes}min").mean()
        apyBase = apyBase.iloc[-1]
        apyBase = apyBase[(apyBase>0)&(apyBase.index != "cash")]
        supply_pool = apyBase.idxmax()
        
        apyBaseBorrow = self.now_data.apyBaseBorrow
        apyBaseBorrow = apyBaseBorrow.resample(f"{self.past_window_in_minutes}min").mean()
        apyBaseBorrow = apyBaseBorrow.iloc[-1]
        apyBaseBorrow = apyBaseBorrow[(apyBaseBorrow>0)&(~apyBaseBorrow.index.isin(["cash", supply_pool]))]
        borrow_pool = apyBaseBorrow.idxmin()
        
        supply_rate = apyBase.max()
        borrow_rate = apyBaseBorrow.min()
        

        # if date_ix in [
        #     pd.to_datetime("2025-01-28 11:59:59"),
        #     pd.to_datetime("2025-01-28 12:09:59"),
        #     pd.to_datetime("2025-01-28 12:04:59"),
        # ]:
        #     print(">>>>", date_ix)
        #     print(f"{supply_pool = }", apyBase)
        #     print(f"{borrow_pool = }", apyBaseBorrow)
        #     print()

        
        if supply_rate - borrow_rate <= self.profitability_threshold:
            # not profitable
            self.ws = {self.token:{"cash":1} }
        elif supply_pool == borrow_pool:
            # Arb opportunity in the same market is not supported. 
            # Use the Generic Loop strategy and pools instead.
            self.ws = {self.token:{"cash":1} }
        else:
            # profitable
            self.ws = {self.token:{supply_pool:1, borrow_pool:-1, "cash":1} }
        return self.ws
