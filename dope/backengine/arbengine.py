import numpy as np
import pandas as pd

from dope.market_impact.linear import LinearMktImpactModel
from dope.market_impact.neighborhood import NeighborhoodMktImpactModel


class TokenPortfolio:
    def __init__(self):
        self.allocation = {}

    def __len__(self):
        return len(self.allocation)

    def compound(self, protocol_rate_dict, dt):
        for protocol, rate in protocol_rate_dict.items():
            if protocol in self.allocation:
                self.allocation[protocol] = self.allocation[protocol] * (
                    1 + rate / 100 * dt
                )

    def capital(self):
        # print("@Capital()")
        # print(self.allocation)
        # print(self.allocation.values())
        # print(sum(self.allocation.values()))
        # print("Done with capital()")
        return sum(self.allocation.values())
    
    def deposit_capital(self):
        return sum(capital for capital in self.allocation.values() if capital > 0)
    
    def debt_capital(self):
        return sum(abs(capital) for capital in self.allocation.values() if capital < 0)

    def weights(self):
        total_capital = self.capital()
        return {
            protocol: capital / total_capital
            for protocol, capital in self.allocation.items() if capital != 0
        }

    def rebalance(self, new_weights, capital=None):
        
        capital = capital or self.capital()
        # print(f"{_capital = :,} ")
        # print(f"{new_weights = }")

        for protocol, weight in new_weights.items():
            if weight > 0:
                self.allocation[protocol] = weight * capital
            else:
                self.allocation[protocol] = weight * capital

        # remove the ones not allocated anymore (in case zero weights disapear)
        to_delete = []
        for protocol in list(self.allocation.keys()):
            if protocol not in new_weights:
                to_delete.append(protocol)
                # self.allocation[protocol] = 0

        for protocol in to_delete:
            del self.allocation[protocol]
        
        # print(f"{self.allocation = }")


class AnyDict:
    def __init__(self, value):
        self.value = value
    
    def __getitem__(self, _):
        return self.value


class ArbBacktester:

    def __init__(
        self,
        strategy,
        data,
        borrow_lend_data=None,
        mkt_impact=None,
        tokens=None,
        mkt_impact_mode="past",
    ):
        self.strategy = strategy
        self.data = data
        self.data_blocks = {token: data.as_block(token) for token in data.keys()}
        if borrow_lend_data is None:
            self.borrow_lend_data = self.data
        else:
            self.borrow_lend_data = borrow_lend_data
        if mkt_impact_mode == "past":
            mkt_impact = {token: {mkt:NeighborhoodMktImpactModel() for mkt in data.keys()} for token, data in self.data.items()}
        elif mkt_impact_mode == "linear":
            mkt_impact = {token: {mkt:LinearMktImpactModel() for mkt in data.keys()} for token, data in self.data.items()} 
        elif mkt_impact_mode == "zero":
            mkt_impact = AnyDict(AnyDict(LinearMktImpactModel.zero_instance()))
        else:
            assert mkt_impact is not None, "mkt_impact must be provided if mkt_impact_mode is not 'past', 'linear', or 'zero'"
        self.mkt_impact = mkt_impact
        self.dates = self.get_dates()
        self.summary = None

        self.tokens = tokens or list(self.data.keys())
        self.πs = {token: TokenPortfolio() for token in self.tokens}
    
    def deposit_to_wallet(self, token, amount):
        self.πs[token].rebalance({"cash": 1}, amount)
        #self.πs[token].allocation["cash"] += amount

    def get_dates(self):
        dates = set()
        for token, data in self.data.items():
            for protocol, df in data.items():
                dates.update(df.index)
        dates = list(dates)
        dates.sort()
        return dates

    def get_capital(self, token):
        return self.πs[token].capital()

    def block_data_up_to_now(self, token, date_ix):
        block = self.data_blocks[token]
        block = block[block.index <= date_ix]
        return block

    def prep(self):
        self.strategy.register_engine(self)

        self.data.add_cash_mkt()
        for _token in self.data.keys():
            for mkt in self.data[_token].keys():
                self.mkt_impact[_token][mkt].set_data_ref(self.data[_token][mkt])

    def __call__(self, start_timestamp=None, end_timestamp=None):
        if start_timestamp is not None:
            start_timestamp = pd.to_datetime(start_timestamp)
        else:
            start_timestamp = pd.to_datetime("2000-01-01")
        if end_timestamp is not None:
            end_timestamp = pd.to_datetime(end_timestamp)
        else:
            end_timestamp = pd.to_datetime("2100-01-01")

        self.prep()  # register data, slippage model, strategy

        rows = []
        Ws = []
        _days = len(self.dates)
        print(f"Running Backtest for {_days:,} | token:{self.strategy.token }")
        _tenpct = _days // 10
        started = False
        for i in range(0, len(self.dates[:]) - 1):
            if i % _tenpct == 0:
                print(f"{i:>10,}/{_days:>10,}", end="\r")
            date_prev = self.dates[i]
            date_now = self.dates[i + 1]
            self.date_now = date_now

            if date_now < start_timestamp:
                continue
            if date_now >= end_timestamp:
                break

            if not started:
                self.strategy.on_start(date_now)
                started = True
            
            # Step 1: Position gets Accrued:
            for token, π in self.πs.items():
                r_breakdown = {}
                impacts = {}
                if len(π) > 0:
                    ws_before = π.weights()

                for mkt, capital in π.allocation.items():
                    if mkt not in self.borrow_lend_data[token]:
                        continue
                    df = self.borrow_lend_data[token][mkt]
                    _filter = df.index <= date_now
                    if len(df[_filter]) == 0:
                        continue
                    is_borrow = capital < 0
                    # side_name = "apyBase" if capital >= 0	else "apyBaseBorrow"
                    side_name = "apyBaseBorrow" if is_borrow else "apyBase"
                    if mkt == "cash":
                        assert (
                            side_name != "apyBaseBorrow"
                        ), "Cannot Borrow from own wallet."
                    impacts[mkt] = self.mkt_impact[token][mkt].impact(
                        date_now, π.allocation.get(mkt, 0), is_borrow=is_borrow
                    )
                    # _rate_now = df[_filter][side_name].iloc[-1]
                    if not np.isfinite(impacts[mkt]):
                        _rate = df[_filter][side_name].iloc[-1]
                    else:
                        _rate = df[_filter][side_name].iloc[-1] + impacts[mkt]
                    if not np.isfinite(_rate):
                        continue
                    # if is_print:
                    #     print(mkt,side_name , "_rate", _rate, df[_filter][side_name].iloc[-1], impacts[mkt], π.allocation.get(mkt, 0) )
                    r_breakdown[mkt] = max(0, _rate)

                π.compound(
                    r_breakdown,
                    dt=(date_now - date_prev).total_seconds() / 365 / 24 / 3600,
                )
                raw_r_breakdown = r_breakdown.copy()
                for mkt in r_breakdown.keys():
                    r_breakdown[mkt] = ws_before.get(mkt, 0) * r_breakdown[mkt]
                if len(π) > 0:
                    if len(df[_filter]) == 0:
                        continue
                    timestamp = df[_filter].index[-1]
                    rows.append(
                        [
                            date_now,
                            timestamp,
                            {token: π.weights()},
                            sum(r_breakdown.values()),
                            π.capital(),
                            r_breakdown,
                            raw_r_breakdown,
                            impacts,
                        ]
                    )
                    Ws.append(
                        [
                            date_now,
                            timestamp,
                            self.strategy.token,
                            {token: ws_before},
                            π.capital(),
                        ]
                    )

            # Step 2: Strategy Acts
            ws = self.strategy.on_act(date_now)
            # print("Ws::::",ws)

            # step 3: Rebalance
            for token, _ws in ws.items():
                #print(">", self.πs[token].allocation)
                if len(self.πs[token]) == 0:
                    self.πs[token].rebalance(_ws, self.strategy.capital)
                else:
                    self.πs[token].rebalance(_ws, None)
                #print("<", self.πs[token].allocation)

            # print()
            # print("π:::::",π.allocation)
        strategy = pd.DataFrame(
            rows,
            columns=[
                "datetime",
                "timestamp",
                "ws",
                "rate",
                "capital",
                "breakdown",
                "raw_breakdown",
                "mkt_impact",
            ],
        )
        strategy = strategy.set_index("datetime")
        self.summary = strategy
        self.Ws = pd.DataFrame(
            Ws, columns=["datetime", "timestamp", "token", "ws", "capital"]
        )
        return self.summary, self.Ws
