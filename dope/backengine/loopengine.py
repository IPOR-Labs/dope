import numpy as np
import pandas as pd

from dope.market_impact.linear import LinearMktImpactModel
from dope.token import Token

class PoolAccount:
    def __init__(self, poolname):
        self.poolname = poolname
        self.debt = Token(0, poolname.debt_token)
        self.deposit = Token(0, poolname.deposit_token)

    def __repr__(self):
        return f"PoolAccount({self.poolname}, {self.debt}, {self.deposit})"
    
    def add_deposit(self, capital):
        self.deposit += capital
    
    def add_debt(self, capital):
        self.debt += capital

    def deposit_value(self, price_row):
        return self.deposit.value * price_row.get(self.poolname.deposit_token)

    def debt_value(self, price_row):
        return self.debt.value * price_row.get(self.poolname.debt_token)
        
    def compound(self, protocol_rate_dict, dt):
        for protocol, rate in protocol_rate_dict.items():
            if protocol in self.allocation:
                # print("C::: ",protocol, self.allocation[protocol],  (1 + rate/100*dt), self.allocation[protocol] * (1 + rate/100*dt))
                self.allocation[protocol] = self.allocation[protocol] * (
                    1 + rate / 100 * dt
                )

    def health_factor(self, price_row):

        hf = (
            self.deposit_value(price_row) / self.π.debt.capital(price_row) * self.LT
        )
        return hf


class AccountDict(dict):
    def __missing__(self, key):
        # Create a new object using the key and insert it into the dictionary
        self[key] = PoolAccount(key)
        return self[key]


class TokenPortfolio:
    def __init__(self):
        
        self.pools = AccountDict()
    
    def add_deposit_token(self, pool, capital):
        account = self.pools[pool]        
        account.add_deposit(capital)
        return capital
    
    def take_debt_token(self, pool, capital):
        account = self.pools[pool]
        account.add_debt(capital)
        return capital

    def capital(self, price_row):
        total_capital = 0
        for account in self.pools.values():
            deposit = account.deposit_value(price_row)
            debt = account.debt_value(price_row)
            total_capital += deposit - debt

        return total_capital

    def weights(self, price_row):
        total = self.capital(price_row)
        allocation = {}
        allocation.update(
            {
                protocol: capital / total
                for protocol, capital in self.deposit.allocation_value(
                    price_row
                ).items()
            }
        )
        allocation.update(
            {
                protocol: -capital / total
                for protocol, capital in self.debt.allocation_value(price_row).items()
            }
        )
        allocation.update(
            {
                protocol: capital / total
                for protocol, capital in self.cash.allocation_value(price_row).items()
            }
        )
        return allocation

    def compound(self, protocol_supply_rate_dict, protocol_debt_rate_dict, dt):
        self.deposit.compound(protocol_supply_rate_dict, dt)
        self.debt.compound(protocol_debt_rate_dict, dt)
        # self.cash.compound({}, dt) # cash does not accrue


class LoopBacktester:

    def __init__(
        self,
        strategy,
        data,
        price_data,
        pools,
        mkt_impact=None
    ):
        self.strategy = strategy
        self.data = data
        self.price_data = price_data
        if mkt_impact is None:
            mkt_impact = {
                mkt: LinearMktImpactModel.zero_instance()
                for data in self.data.values()
                for mkt in data.keys()
            }
        self.mkt_impact = mkt_impact
        self.pools = pools
        self.dates = self.get_dates()
        self.summary = None

        self.π = TokenPortfolio()

    def get_dates(self):
        dates = set()
        for protocol, df in self.data.items():
            dates.update(df.index)
        dates = list(dates)
        dates.sort()
        return dates

    def get_capital(self, token):
        return self.π.capital()

    def prep(self):
        self.price_data.set_up_price_timeseries()
        self.strategy.register_engine(self)

        #self.data.add_cash_mkt()
        for mkt in self.data.keys():
            self.mkt_impact[mkt].set_data_ref(self.data[mkt])

    def health_factor(self, price_row):

        hf = (
            self.π.deposit.capital(price_row) / self.π.debt.capital(price_row) * self.LT
        )
        return hf

    def do_loop(
        self,
        date_ix,
        init_capital,
        pair_pool,
        deposit_pool,
        loop_times
    ):
        def total_collateral(n, X, ltv):
            return X * (1-ltv**(n+1)) / (1- ltv)
        def total_borrow(n, X, ltv):
            return X * (1-ltv**(n+1)) / (1- ltv) * ltv
        deposit_pool = self.pools[deposit_pool]
        pair_pool = self.pools[pair_pool]
        
        buffer = 0.1
        # initial deposit
        x = init_capital
        print(total_collateral(loop_times, x, pair_pool.ltv-buffer), 
              total_borrow(loop_times-1, x, pair_pool.ltv-buffer))
        for n in range(loop_times):
            x = self.π.add_deposit_token(pair_pool, x)
            x = self.π.take_debt_token(
                pair_pool,
                Token(x.value * (pair_pool.ltv - buffer), name=pair_pool.debt_token),
            )
            x = self.π.add_deposit_token(deposit_pool, x)
            x = self.swap(
                x.value,
                from_token=deposit_pool.debt_token,
                to_token=pair_pool.deposit_token,
                date_ix=date_ix
            )
        x = self.π.add_deposit_token(pair_pool, x)
    
    def swap(self, token_amount, from_token, to_token, date_ix):
        price_row = self.price_data.price_row_at(date_ix)
        sold_to_usd = token_amount * price_row.get(from_token)
        target_token = sold_to_usd / price_row.get(to_token)
        return Token(target_token, to_token)

    def compound_account(self, account, rate_column, date_now, price_row):
        r_breakdown = {}
        impacts = {}
        for mkt, capital in account.allocation.items():
            for token, _ in self.data.items():
                # print(chain, protocol, token, mkt)
                if mkt not in self.data[token]:
                    continue
                df = self.data[token][mkt]
                _filter = df.index <= date_now
                if len(df[_filter]) == 0:
                    continue
                is_borrow = rate_column == "apyBaseBorrow"
                # side_name = "apyBase" if capital >= 0	else "apyBaseBorrow"
                side_name = rate_column
                if mkt == "cash":
                    assert side_name != "apyBaseBorrow", "Cannot Borrow from own wallet."
                value = account.allocation.get(mkt, 0) * price_row.get_or_zero(mkt)
                impacts[mkt] = self.mkt_impact[mkt].impact(
                    date_now, value, is_borrow=is_borrow
                )
                if not np.isfinite(impacts[mkt]):
                    _rate = df[_filter][side_name].iloc[-1]
                else:
                    _rate = df[_filter][side_name].iloc[-1] + impacts[mkt]
                if not np.isfinite(_rate):
                    continue
                r_breakdown[mkt] = _rate
        return r_breakdown, impacts 

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
        _days = len(self.dates)
        print(f"Running Backtest for {_days:,} | token:{self.strategy.token }")

        did_start = False

        _tenpct = _days // 10
        for i in range(0, len(self.dates[:]) - 1):
            if i % _tenpct == 0:
                print(f"{i:>10,}/{_days:>10,}", end="\r")
            date_prev = self.dates[i]
            date_now = self.dates[i + 1]

            if date_now < start_timestamp:
                continue
            if date_now >= end_timestamp:
                break
            try:
                price_row = self.price_data.price_row_at(date_now)
            except KeyError:
                if did_start:
                    print("Don't have price data for ", date_now, "skipping")
                continue
            if not did_start:
                self.strategy.on_start(date_now)
                did_start = True
                continue
            print("@", date_now, price_row)

            # print(date_now)
            # The agent does not know the future.
            # print(date_now)

            # Step 1: Accounts gets Accrued:


            ws_before = self.π.weights(price_row)

            r_breakdown_deposit, impact_deposit = self.compound_account(
                self.π.deposit, "apyBase", date_now, price_row=price_row
            )
            r_breakdown_debt, impact_debt = self.compound_account(
                self.π.debt, "apyBaseBorrow", date_now, price_row=price_row
            )

            dt = (date_now - date_prev).total_seconds() / 365 / 24 / 3600
            self.π.compound(r_breakdown_deposit, r_breakdown_debt, dt=dt)

            r_breakdown = {}
            for mkt in r_breakdown_deposit.keys():
                r_breakdown[mkt] = ws_before.get(mkt, 0) * r_breakdown_deposit[mkt]
            for mkt in r_breakdown_debt.keys():
                r_breakdown[mkt] = ws_before.get(mkt, 0) * r_breakdown_debt[mkt]
            rate = sum(r_breakdown.values())
            health_factor = self.health_factor(price_row)
            if len(self.π.weights(price_row)) > 0:
                rows.append(
                    [
                        date_now,
                        self.π.weights(price_row),
                        rate,
                        health_factor,
                        self.π.capital(price_row),
                        r_breakdown,
                        r_breakdown_deposit,
                        r_breakdown_debt,
                        impact_deposit,
                        impact_debt,
                    ]
                )
                # Ws.append([date_now, self.strategy.token, {token:ws_before}, self.π.capital(price_row)])

            # Step 2: Strategy Acts
            ws = self.strategy.on_act(date_now)
            # print("Ws::::",ws)

            # step 3: Rebalance
            # for token, _ws in ws.items():
            #   #print(">", self.π[token].allocation)
            #   if len(self.π[token]) ==0:
            #     self.π[token].rebalance(_ws, self.strategy.capital)
            #   else:
            #     self.π[token].rebalance(_ws, None)
            # print("<", self.π.allocation)

        strategy = pd.DataFrame(
            rows,
            columns=[
                "datetime",
                "ws",
                "rate",
                "health_factor",
                "capital",
                "r_breakdown",
                "r_breakdown_deposit",
                "r_breakdown_debt",
                "impact_deposit",
                "impact_debt",
            ],
        )
        strategy = strategy.set_index("datetime")
        self.summary = strategy
        # self.Ws = pd.DataFrame(Ws, columns=["datetime", "timestamp", "token", "ws", "capital"])
        return self.summary
