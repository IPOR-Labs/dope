class AaveV3Rates:
  def __init__(self, supply_rate, borrow_rate):
    self.supply_rate = supply_rate
    self.borrow_rate = borrow_rate


class AaveV3Market:
  AAVE_PERCENTAGE_FACTOR = 10 ** 4

  def __init__(
      self,
      unbacked,
      total_stable_debt,
      total_variable_debt,
      total_liquidity,
      available_liquidity,
      average_stable_borrow_rate,
      optimal_usage_ratio,
      max_excess_usage_ratio,
      variable_rate_slope1,
      variable_rate_slope2,
      base_variable_borrow_rate,
      reserve_factor):
    self.unbacked = unbacked
    self.total_stable_debt = total_stable_debt
    self.total_variable_debt = total_variable_debt
    self.total_liquidity = total_liquidity
    self.available_liquidity = available_liquidity
    self.average_stable_borrow_rate = average_stable_borrow_rate
    self.optimal_usage_ratio = optimal_usage_ratio
    self.max_excess_usage_ratio = max_excess_usage_ratio
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

  def rates(self) -> AaveV3Rates:
    total_debt = (self.total_stable_debt + self.total_variable_debt) * (10 ** 9)
    current_liquidity_rate = 0
    current_variable_borrow_rate = self.base_variable_borrow_rate
    supply_usage_ratio = 0
    borrow_usage_ratio = 0

    if total_debt > 0:
      available_liquidity_9_dec = self.available_liquidity * (10 ** 9)
      available_liquidity_plus_debt = available_liquidity_9_dec + total_debt
      borrow_usage_ratio = total_debt * (10 ** 27) // available_liquidity_plus_debt
      supply_usage_ratio = total_debt * (10 ** 27) // (available_liquidity_plus_debt + self.unbacked)

    if borrow_usage_ratio > self.optimal_usage_ratio:
      excess_borrow_usage_ratio = (borrow_usage_ratio - self.optimal_usage_ratio) * (
          10 ** 18) // self.max_excess_usage_ratio
      current_variable_borrow_rate += self.variable_rate_slope1 + (
          self.variable_rate_slope2 * excess_borrow_usage_ratio // self.optimal_usage_ratio)
    else:
      current_variable_borrow_rate += self.variable_rate_slope1 * borrow_usage_ratio // self.optimal_usage_ratio

    current_liquidity_rate = self._get_aave_v3_overall_borrow_rate(current_variable_borrow_rate) * supply_usage_ratio // (
        10 ** 18) * (self.AAVE_PERCENTAGE_FACTOR - self.reserve_factor) // self.AAVE_PERCENTAGE_FACTOR

    return AaveV3Rates(current_liquidity_rate // (10 ** 9), current_variable_borrow_rate // (10 ** 9))

  def _get_aave_v3_overall_borrow_rate(self, variable_borrow_rate: int) -> int:
    total_debt = self.total_stable_debt + self.total_variable_debt
    if total_debt == 0:
      return 0

    weighted_variable_borrow_rate = self.total_variable_debt * variable_borrow_rate // (10 ** 27)
    weighted_stable_borrow_rate = self.total_stable_debt * self.average_stable_borrow_rate // (10 ** 27)
    return (weighted_variable_borrow_rate + weighted_stable_borrow_rate) * (10 ** 18) // total_debt


def test():
  # @formatter:off
  market = AaveV3Market(
      unbacked=0,
      total_stable_debt=0,
      total_variable_debt=538809180377,
      total_liquidity=580289933402,
      available_liquidity=0,
      average_stable_borrow_rate=0,
      optimal_usage_ratio=900000000000000000000000000,
      max_excess_usage_ratio=100000000000000000000000000,
      variable_rate_slope1=35000000000000000000000000,
      variable_rate_slope2=600000000000000000000000000,
      base_variable_borrow_rate=0,
      reserve_factor=1000
  )
  # @formatter:on

  result = market.rates()
  print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
  print("supply_rate: ", result.supply_rate / (10 ** 16), "%")

  market.increase_liquidity(538809180377)
  result = market.rates()
  print('After increase supply')
  print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
  print("supply_rate: ", result.supply_rate / (10 ** 16), "%")


if __name__ == '__main__':
  test()
