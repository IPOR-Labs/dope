class AaveV2Rates:
  def __init__(self, supply_rate, borrow_rate):
    self.supply_rate = supply_rate
    self.borrow_rate = borrow_rate


class AaveV2Market:
  AAVE_PERCENTAGE_FACTOR = 10 ** 4
  ONE_18_DEC = (10 ** 18)
  ONE_27_DEC = 10 ** 27
  ONE_9_DEC = 10 ** 9

  def __init__(
      self,
      total_stable_debt,
      total_variable_debt,
      total_liquidity,
      market_borrow_rate,
      average_stable_borrow_rate,
      optimal_utilization_rate,
      excess_utilization_rate,
      variable_rate_slope1,
      variable_rate_slope2,
      base_variable_borrow_rate,
      reserve_factor):
    self.total_stable_debt = total_stable_debt
    self.total_variable_debt = total_variable_debt
    self.total_liquidity = total_liquidity
    self.market_borrow_rate = market_borrow_rate
    self.average_stable_borrow_rate = average_stable_borrow_rate
    self.optimal_utilization_rate = optimal_utilization_rate
    self.excess_utilization_rate = excess_utilization_rate
    self.variable_rate_slope1 = variable_rate_slope1
    self.variable_rate_slope2 = variable_rate_slope2
    self.base_variable_borrow_rate = base_variable_borrow_rate
    self.reserve_factor = reserve_factor

  def increase_variable_debt(self, additional_variable_debt):
    self.total_variable_debt += additional_variable_debt

  def decrease_variable_debt(self, repay_amount):
    self.total_variable_debt -= repay_amount

  def increase_liquidity(self, liquidity):
    self.total_liquidity += liquidity

  def decrease_liquidity(self, liquidity):
    self.total_liquidity -= liquidity

  def borrow_rate(self, utilization_rate):
    if utilization_rate * (10 ** 9) <= self.optimal_utilization_rate:
      return self.base_variable_borrow_rate + utilization_rate * self.variable_rate_slope1 // self.optimal_utilization_rate
    else:
      excess_utilization_rate_ratio = (((utilization_rate * (10 ** 9) - self.optimal_utilization_rate) // (10 ** 9)) * (
          10 ** 18) // self.excess_utilization_rate) * (10 ** 9)
      return (self.base_variable_borrow_rate + self.variable_rate_slope1 + (
          self.variable_rate_slope2 * excess_utilization_rate_ratio // (10 ** 18))) // (10 ** 9)

  def get_aave_overall_borrow_rate(self, variable_borrow_rate):
    total_debt = self.total_stable_debt + self.total_variable_debt
    if total_debt == 0:
      return 0
    weighted_variable_borrow_rate = self.total_variable_debt * variable_borrow_rate // (10 ** 18)
    weighted_stable_borrow_rate = self.total_stable_debt * self.average_stable_borrow_rate // self.ONE_27_DEC
    return (weighted_variable_borrow_rate + weighted_stable_borrow_rate) * (10 ** 18) // total_debt

  def rates(self):
    total_debt = self.total_stable_debt + self.total_variable_debt
    utilization_rate = total_debt * (10 ** 18) // self.total_liquidity
    borrow_rate = self.borrow_rate(utilization_rate)
    overall_borrow_rate = self.get_aave_overall_borrow_rate(borrow_rate)
    supply_rate = overall_borrow_rate * utilization_rate // (10 ** 18) * (
        self.AAVE_PERCENTAGE_FACTOR - self.reserve_factor) // self.AAVE_PERCENTAGE_FACTOR
    return AaveV2Rates(supply_rate, borrow_rate)


def test():
  market = AaveV2Market(total_stable_debt=3576325734551,
                        total_variable_debt=199188677981386,
                        total_liquidity=249712365461815,
                        market_borrow_rate=100000000000000000000000000,
                        average_stable_borrow_rate=94121248717367022285271907,
                        optimal_utilization_rate=800000000000000000000000000,
                        excess_utilization_rate=200000000000000000000000000,
                        variable_rate_slope1=40000000000000000000000000,
                        variable_rate_slope2=1000000000000000000000000000,
                        base_variable_borrow_rate=0,
                        reserve_factor=2000)

  result = market.rates()
  print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
  print("supply_rate: ", result.supply_rate / (10 ** 16), "%")

  market.increase_liquidity(249712365461815 / 2)
  result = market.rates()
  print('After increase supply')
  print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
  print("supply_rate: ", result.supply_rate / (10 ** 16), "%")


if __name__ == '__main__':
  test()
