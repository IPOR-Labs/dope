import numpy as np
import pandas as pd

from dope.market_impact.linear import LinearMktImpactModel


class TokenAccount:
    def __init__(self, allocation=None, collateral=None):
        self.allocation = allocation or {}
        self.collateral = collateral or {}

    def __len__(self):
        return len(self.allocation)

    def compound(self, protocol_rate_dict, dt):
        for protocol, rate in protocol_rate_dict.items():
            if protocol in self.allocation:
                # print("C::: ",protocol, self.allocation[protocol],  (1 + rate/100*dt), self.allocation[protocol] * (1 + rate/100*dt))
                self.allocation[protocol] = self.allocation[protocol] * (
                    1 + rate / 100 * dt
                )

    def allocation_value(self, price_row):
        allocation_val = {}
        for protocol, capital in self.allocation.items():
            allocation_val[protocol] = price_row.get_or_zero(protocol) * capital
        return allocation_val

    def collateral_value(self, price_row):
        total_collateral = 0
        for protocol, capital in self.collateral.items():
            total_collateral += price_row.get_or_zero(protocol) * capital
        return total_collateral

    def capital(self, price_row):
        # #print("@Capital()")
        # print(self.allocation)
        # print(self.allocation.values())
        # print(sum(self.allocation.values()))
        # print("Done with capital()")
        total = 0
        for protocol, capital in self.allocation.items():
            total += price_row.get_or_zero(protocol) * capital
        return total

    def weights(self):
        total = self.capital()
        return {
            protocol: capital / total for protocol, capital in self.allocation.items()
        }

    def rebalance(self, new_weights, capital=None):
        _capital = capital or self.capital()
        # print(f"{_capital = :,} ")

        for protocol, weight in new_weights.items():
            self.allocation[protocol] = weight * _capital

        # remove the ones not allocated anymore (in case zero weights disapear)
        for protocol in list(self.allocation.keys()):
            if protocol not in new_weights:
                self.allocation[protocol] = 0


class TokenPortfolio:
    def __init__(self):
        self.deposit = TokenAccount()
        self.debt = TokenAccount()
        self.cash = TokenAccount()

    def add_deposit(self, protocol, capital):
        if protocol in self.deposit.allocation:
            self.deposit.allocation[protocol] += capital
        else:
            self.deposit.allocation[protocol] = capital

    def add_debt(self, protocol, capital):
        if protocol in self.debt.allocation:
            self.debt.allocation[protocol] += capital
        else:
            self.debt.allocation[protocol] = capital

    def add_cash(self, protocol, capital):
        if protocol in self.cash.allocation:
            self.cash.allocation[protocol] += capital
        else:
            self.cash.allocation[protocol] = capital

    def capital(self, price_row):
        total_capital = (
            self.cash.capital(price_row)
            + self.deposit.capital(price_row)
            - self.debt.capital(price_row)
        )
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
        self, strategy, data, price_data, liquidation_threshold, mkt_impact=None
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
        self.LT = liquidation_threshold
        self.dates = self.get_dates()
        self.summary = None

        self.π = TokenPortfolio()

    def get_dates(self):
        dates = set()
        for token, data in self.data.items():
            for protocol, df in data.items():
                dates.update(df.index)
        dates = list(dates)
        dates.sort()
        return dates

    def get_capital(self, token):
        return self.π.capital()

    def prep(self):
        self.strategy.register_engine(self)

        self.data.add_cash_mkt()
        for _token in self.data.keys():
            for mkt in self.data[_token].keys():
                self.mkt_impact[mkt].set_data_ref(self.data[_token][mkt])

    def health_factor(self, price_row):

        hf = (
            self.π.deposit.capital(price_row) / self.π.debt.capital(price_row) * self.LT
        )
        return hf

    def compound_account(self, account, rate_column, date_now, price_row):
        r_breakdown = {}
        impacts = {}
        for mkt, capital in account.allocation.items():
            chain, protocol, token = mkt
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
            # print(">>>> impact", mkt, impacts[mkt], value, account.allocation.get(mkt, 0), price_row.get_or_zero(mkt))
            # _rate_now = df[_filter][side_name].iloc[-1]
            if not np.isfinite(impacts[mkt]):
                _rate = df[_filter][side_name].iloc[-1]
            else:
                _rate = df[_filter][side_name].iloc[-1] + impacts[mkt]
            if not np.isfinite(_rate):
                continue
            # print(mkt,"_rate", _rate, df[_filter][side_name].iloc[-1], impacts[mkt], π.allocation.get(mkt, 0) )
            r_breakdown[mkt] = _rate
        return r_breakdown, impacts
        # account.compound(r_breakdown, dt=(date_now - date_prev).total_seconds()/365/24/3600)

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
            # print("@", date_now, price_row)

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
            ws = self.strategy.act(date_now)
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
