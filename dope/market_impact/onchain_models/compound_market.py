class CompoundRates:
  def __init__(self, supply_rate, borrow_rate):
    self.supply_rate = supply_rate
    self.borrow_rate = borrow_rate


class CompoundMarket:
  COMPOUND_BLOCKS_PER_YEAR = 5 * 60 * 24 * 365

  def __init__(
      self,
      exchange_rate_stored,
      total_borrows,
      total_supply,
      total_reserves,
      reserve_factor_mantissa,
      multiplier_per_block,
      base_rate_per_block,
      kink,
      jump_multiplier_per_block):
    self.exchange_rate_stored = exchange_rate_stored
    self.total_borrows = total_borrows
    self.total_supply = total_supply
    self.total_reserves = total_reserves
    self.reserve_factor_mantissa = reserve_factor_mantissa
    self.multiplier_per_block = multiplier_per_block
    self.base_rate_per_block = base_rate_per_block
    self.kink = kink
    self.jump_multiplier_per_block = jump_multiplier_per_block

  def increase_borrow(self, additional_borrow):
    self.total_borrows += additional_borrow

  def decrease_borrow(self, repay_amount):
    self.total_borrows -= repay_amount

  def increase_supply(self, supply):
    self.total_supply += supply * 10 ** 18 // self.exchange_rate_stored

  def decrease_supply(self, supply):
    self.total_supply -= supply * 10 ** 18 // self.exchange_rate_stored

  def rates(self):
    total_supply = self.total_supply * self.exchange_rate_stored // (10 ** 18)
    if total_supply == 0:
      return CompoundRates(0, 0)

    utilization_rate = self.total_borrows * (10 ** 18) // total_supply
    borrow_rate = self._calculate_compound_borrow_rate(utilization_rate)
    liquidity_rate = self._calculate_compound_liquidity_rate(utilization_rate)
    return CompoundRates(liquidity_rate, borrow_rate)

  def _calculate_compound_borrow_rate(self, utilization_rate):
    borrow_rate_per_block = self._calculate_compound_borrow_rate_per_block(utilization_rate)
    return self._calculate_rate_apr(borrow_rate_per_block)

  def _calculate_compound_liquidity_rate(self, utilization_rate):
    borrow_rate_per_block = self._calculate_compound_borrow_rate_per_block(utilization_rate)
    one_minus_reserve_factor = (10 ** 18) - self.reserve_factor_mantissa
    rate_to_pool = borrow_rate_per_block * one_minus_reserve_factor // (10 ** 18)
    rate_per_block = utilization_rate * rate_to_pool // (10 ** 18)
    return self._calculate_rate_apr(rate_per_block)

  def _calculate_compound_borrow_rate_per_block(self, utilization_rate):
    if utilization_rate <= self.kink:
      return (utilization_rate * self.multiplier_per_block // (10 ** 18)) + self.base_rate_per_block
    else:
      normal_rate = (self.kink * self.multiplier_per_block // (10 ** 18)) + self.base_rate_per_block
      excess_utilization_rate = utilization_rate - self.kink
      return (excess_utilization_rate * self.jump_multiplier_per_block // (10 ** 18)) + normal_rate

  def _calculate_rate_apr(self, rate_per_block):
    return rate_per_block * self.COMPOUND_BLOCKS_PER_YEAR


def test():
  exchange_rate_stored = 228207543971326
  total_borrows = 228474394749526
  total_supply = 1681658283289354779
  total_reserves = 13740656842428
  reserve_factor_mantissa = 75000000000000000
  multiplier_per_block = 23782343987
  base_rate_per_block = 0
  kink = 800000000000000000
  jump_multiplier_per_block = 518455098934

  market = CompoundMarket(exchange_rate_stored,
                          total_borrows,
                          total_supply,
                          total_reserves,
                          reserve_factor_mantissa,
                          multiplier_per_block,
                          base_rate_per_block,
                          kink,
                          jump_multiplier_per_block)

  result = market.rates()
  print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
  print("supply_rate: ", result.supply_rate / (10 ** 16), "%")

  market.increase_supply(228474394749526 / 2)
  result = market.rates()
  print('After increase supply')
  print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
  print("supply_rate: ", result.supply_rate / (10 ** 16), "%")


if __name__ == '__main__':
  test()
