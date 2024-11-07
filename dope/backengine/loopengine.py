import numpy as np
import pandas as pd

from dope.market_impact.linear import LinearMktImpactModel
from dope.market_impact.neighborhood import NeighborhoodMktImpactModel
from dope.pools.pools import Pool
from dope.token import Token


class PoolAccount:
    def __init__(self, pool: Pool):
        self.pool = pool
        self.debt = Token(0, pool.debt_token)
        self.deposit = Token(0, pool.deposit_token)

    def __repr__(self):
        return f"PoolAccount({self.pool}, {self.debt}, {self.deposit})"
    
    def add_deposit(self, capital):
        self.deposit += capital
    
    def add_debt(self, capital):
        self.debt += capital

    def deposit_value(self, price_row):
        return self.deposit.value * price_row.get(self.pool.deposit_token)

    def debt_value(self, price_row):
        return self.debt.value * price_row.get(self.pool.debt_token)
        
    def compound(self, protocol_rate_dict, dt):
        # print(protocol_rate_dict)
        # print(f"Before Compound", self.debt, self.deposit)
        if self.pool.debt_pool_id is not None:
            debt_rate = np.abs(protocol_rate_dict.get(self.pool.debt_name, 0))
            self.debt = self.debt * (1 + debt_rate / 100 * dt)
        if self.pool.deposit_pool_id is not None:
            deposit_rate = protocol_rate_dict.get(self.pool.deposit_name, 0)
            self.deposit = self.deposit * (1 + deposit_rate / 100 * dt)
        #     print(f"Deposit Rate: {deposit_rate}")
        # print("After Compound", self.debt, self.deposit)
        # print()

    def health_factor(self, price_row):
        debt_value = self.debt_value(price_row)
        if debt_value == 0:
            return np.inf
        hf = (
            self.deposit_value(price_row) / self.debt_value(price_row) * self.pool.LT
        )
        return hf


class AccountDict(dict):
    def __missing__(self, key):
        # Create a new object using the key and insert it into the dictionary
        self[key] = PoolAccount(key)
        return self[key]


class TokenPortfolio:
    def __init__(self):
        
        self.accounts = AccountDict()
    
    def add_deposit_token(self, pool, capital):
        account = self.accounts[pool]        
        account.add_deposit(capital)
        return capital
    
    def take_debt_token(self, pool, capital):
        account = self.accounts[pool]
        account.add_debt(capital)
        return capital

    def capital(self, price_row):
        total_capital = 0
        for account in self.accounts.values():
            deposit = account.deposit_value(price_row)
            debt = account.debt_value(price_row)
            total_capital += deposit - debt

        return total_capital
    
    def weights(self, price_row):
        return self.get_allocation(price_row, normalize=True)
    
    def capital_allocation(self, price_row):
        return self.get_allocation(price_row, normalize=False)
    
    def token_allocation(self, price_row):
        return self.get_allocation(price_row, normalize=False, is_token_amount=True)

    def get_allocation(self, price_row, normalize=True, is_token_amount=False):
        allocation = {}
        total_capital = 0
        for account in self.accounts.values():
            deposit_value = account.deposit_value(price_row) if not is_token_amount else account.deposit.value
            debt_value = account.debt_value(price_row) if not is_token_amount else account.debt.value
            total_capital += deposit_value - debt_value
            debt_key = account.pool.debt_name
            deposit_key = account.pool.deposit_name
            if deposit_key not in allocation:
                allocation[deposit_key] = deposit_value
            else: 
                allocation[deposit_key] += deposit_value
            if debt_key not in allocation:
                allocation[debt_key] = -debt_value
            else:
                allocation[debt_key] -= debt_value

        if normalize:
            allocation = {k: v / total_capital for k, v in allocation.items()}
        return allocation

    def compound(self, protocol_rate_dict, dt):
        for account in self.accounts.values():
            account.compound(protocol_rate_dict, dt)
        #self.deposit.compound(protocol_supply_rate_dict, dt)
        #self.debt.compound(protocol_debt_rate_dict, dt)
        # self.cash.compound({}, dt) # cash does not accrue


class TradeInterface:
    
    def add_deposit_token(self, pool, capital):
        return self.π.add_deposit_token(pool, capital)
    
    def take_debt_token(self, pool, capital):
        return self.π.take_debt_token(pool, capital)
    
    def weights(self, price_row):
        return self.π.weights(price_row)
    
    def capital_allocation(self, price_row):
        return self.π.get_allocation(price_row, normalize=False)
    
    def capital(self, price_row):
        return self.π.capital(price_row)
    
    def price_of_token(self, token_name):
        price_row = self.price_data.price_row_at(self.date_now)
        return price_row.get(token_name)


class LoopBacktester(TradeInterface):

    def __init__(
        self,
        strategy,
        data,
        price_data,
        pools,
        mkt_impact=None,
        add_reward_deposit=False,
        add_reward_borrow=False,
    ):
        self.strategy = strategy
        self.data = data
        self.price_data = price_data
        if mkt_impact is None:
            mkt_impact = {mkt:NeighborhoodMktImpactModel() for mkt in self.data.keys()}
        self.mkt_impact = mkt_impact
        self.pools = pools
        self.dates = self.get_dates()
        self.summary = None
        self.date_now = None

        self.add_reward_deposit = add_reward_deposit
        self.add_reward_borrow = add_reward_borrow
        self.π = TokenPortfolio()

    def get_time(self):
        return self.date_now

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
        self.π = TokenPortfolio()
        self.price_data.set_up_price_timeseries()
        self.strategy.register_engine(self)

        #self.data.add_cash_mkt()
        for mkt in self.data.keys():
            self.mkt_impact[mkt].set_data_ref(self.data[mkt])

    def health_factor(self, price_row):
        hf = {}
        for account in self.π.accounts.values():
            hf[account.pool] = account.health_factor(price_row)
        return hf

    def _swap_at_time(self, token_amount, from_token, to_token, date_ix):
        price_row = self.price_data.price_row_at(date_ix)
        sold_to_usd = token_amount * price_row.get(from_token)
        target_token = sold_to_usd / price_row.get(to_token)
        return Token(target_token, to_token) 

    def swap(self, token_amount, from_token, to_token):
        return self._swap_at_time(token_amount, from_token, to_token, self.date_now)

    def rates_to_compound(self, date_now, price_row):
        r_breakdown = {}
        impacts = {}
        
        for account in self.π.accounts.values():
            # print(">>",account)
            # print("account.pool.debt_pool_id:", account.pool.debt_pool_id)
            
            if account.pool.debt_pool_id is not None:
                df = self.data[account.pool.debt_rate_keyid]
                side_name = "apyBaseBorrow"
                pool_id = account.pool.debt_rate_keyid

                _filter = df.index <= date_now
                if len(df[_filter]) == 0:
                    continue
                debt_value = account.debt_value(price_row)

                impacts[pool_id] = self.mkt_impact[pool_id].impact(
                    date_now, debt_value, is_borrow=True
                )
                if not np.isfinite(impacts[pool_id]):
                    _rate = df[_filter][side_name].iloc[-1]
                else:
                    _rate = df[_filter][side_name].iloc[-1] + impacts[pool_id]
                if not np.isfinite(_rate):
                    continue

                if self.add_reward_borrow:
                    reward_name = "apyRewardBorrow"
                    _rate -= df[_filter][reward_name].iloc[-1]

                r_breakdown[account.pool.debt_name] = -_rate
            # print("account.pool.deposit_pool_id:", account.pool.deposit_pool_id)
            if account.pool.deposit_pool_id is not None:
                df = self.data[account.pool.deposit_rate_keyid]
                side_name = "apyBase"
                pool_id = account.pool.deposit_rate_keyid

                _filter = df.index <= date_now
                if len(df[_filter]) == 0:
                    continue
                deposit_value = account.deposit_value(price_row)
                impacts[pool_id] = self.mkt_impact[pool_id].impact(
                    date_now, deposit_value, is_borrow=False
                )
                if not np.isfinite(impacts[pool_id]):
                    _rate = df[_filter][side_name].iloc[-1]
                else:
                    _rate = df[_filter][side_name].iloc[-1] + impacts[pool_id]
                if not np.isfinite(_rate):
                    continue

                if self.add_reward_deposit:
                    reward_name = "apyReward"
                    _rate += df[_filter][reward_name].iloc[-1]

                r_breakdown[account.pool.deposit_name] = _rate
            # print()
            # print(r_breakdown)
            # print()

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
        print(f"Running Backtest for {_days:,}")

        did_start = False

        _tenpct = _days // 10
        for i in range(0, len(self.dates[:]) - 1):
            if i % _tenpct == 0:
                print(f"{i:>10,}/{_days:>10,}", end="\r")
            date_prev = self.dates[i]
            self.date_now = self.dates[i + 1]

            if self.date_now < start_timestamp:
                continue
            if self.date_now >= end_timestamp:
                break
            try:
                price_row = self.price_data.price_row_at(self.date_now)
            except KeyError:
                if did_start:
                    print("Don't have price data for ", self.date_now, "skipping")
                continue
            if not did_start:
                self.strategy.on_start()
                did_start = True
                continue
            # print("@", self.date_now, price_row, self.π.accounts)
            # print()

            # Step 1: Accounts gets Accrued:
            ws_before = self.π.weights(price_row)

            r_breakdown, impact_breakdown = (
                self.rates_to_compound(self.date_now, price_row=price_row)
            )

            dt = (self.date_now - date_prev).total_seconds() / 365 / 24 / 3600
            self.π.compound(r_breakdown, dt=dt)

            r_by_weight = {}
            for mkt in r_breakdown.keys():
                sign = 1 if r_breakdown[mkt] > 0 else -1
                r_by_weight[mkt] = sign * np.abs(ws_before.get(mkt, 0) * r_breakdown[mkt])
            rate = sum(r_by_weight.values())

            capital = self.π.capital(price_row)
            capital_breakdown = self.π.capital_allocation(price_row)
            token_breakdown = self.π.token_allocation(price_row)
            health_factor = self.health_factor(price_row)
            π_weights = self.π.weights(price_row)
            if len(π_weights) > 0:
                rows.append(
                    [
                        self.date_now,
                        π_weights,
                        rate,
                        health_factor,
                        capital,
                        capital_breakdown,
                        token_breakdown,
                        r_breakdown,
                        r_by_weight,
                        impact_breakdown,
                    ]
                )

            # Step 2: Strategy Acts
            ws = self.strategy.on_act()

            # step 3: Rebalance
            
        strategy = pd.DataFrame(
            rows,
            columns=[
                "datetime",
                "ws",
                "rate",
                "health_factor",
                "capital",
                "capital_breakdown",
                "token_breakdown",
                "r_breakdown",
                "r_by_weight",
                "impact_breakdown",
            ],
        )
        strategy = strategy.set_index("datetime")
        self.summary = strategy
        return self.summary
