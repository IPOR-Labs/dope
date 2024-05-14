ONE_18_DEC = 10 ** 18
ONE_27_DEC = 10 ** 27
SEC_IN_YEAR = 365 * 24 * 60 * 60
CURVE_STEEPNESS = 4000000000000000000
ADJUSTMENT_SPEED = 50000000000000000000 // SEC_IN_YEAR
TARGET_UTILIZATION = 900000000000000000
INITIAL_RATE_AT_TARGET = 40000000000000000 // SEC_IN_YEAR
MIN_RATE_AT_TARGET = 1000000000000000 // SEC_IN_YEAR
MAX_RATE_AT_TARGET = 2000000000000000000 // SEC_IN_YEAR
LN_2_INT = 693147180559945309
LN_WEI_INT = -41446531673892822312
WEXP_UPPER_BOUND = 93859467695000404319
WEXP_UPPER_VALUE = 57716089161558943949701069502944508345128422502756744429568


class MorphoBlueRates:
  def __init__(self, supply_rate, borrow_rate):
    self.supply_rate = supply_rate
    self.borrow_rate = borrow_rate


class MorphoBlueMarketParams:
  irm: str
  collateral_token: str
  lltv: int
  market_id: str
  loan_token: str
  oracle: str


class MorphoBlueMarket:
  def __init__(
      self,
      fee: int,
      last_update: int,
      total_borrow_assets: int,
      total_borrow_shares: int,
      total_supply_assets: int,
      total_supply_shares: int):
    self.fee = fee
    self.last_update = last_update
    self.total_borrow_assets = total_borrow_assets
    self.total_borrow_shares = total_borrow_shares
    self.total_supply_assets = total_supply_assets
    self.total_supply_shares = total_supply_shares


class MorphoBlueMarketMarket:
  def __init__(
      self,
      morpho_blue_market_params: MorphoBlueMarketParams,
      morpho_blue_market: MorphoBlueMarket,
      current_timestamp: int,
      rate_at_target: int):
    self.morpho_blue_market_params = morpho_blue_market_params
    self.morpho_blue_market = morpho_blue_market
    self.current_timestamp = current_timestamp
    self.rate_at_target = rate_at_target

  def rates(self) -> MorphoBlueRates:
    utilization = self._w_div_to_zero(self.morpho_blue_market.total_borrow_assets,
                                      self.morpho_blue_market.total_supply_assets) if self.morpho_blue_market.total_supply_assets > 0 else 0

    err_norm_factor = ONE_18_DEC - TARGET_UTILIZATION if utilization > TARGET_UTILIZATION else TARGET_UTILIZATION
    err = self._w_div_to_zero(utilization - TARGET_UTILIZATION, err_norm_factor)
    speed = self._w_mul_to_zero(ADJUSTMENT_SPEED, err)

    elapsed = self.current_timestamp - self.morpho_blue_market.last_update
    linear_adaptation = speed * elapsed

    start_rate_at_target = self.rate_at_target
    end_rate_at_target = self._new_rate_at_target(start_rate_at_target, linear_adaptation)
    mid_rate_at_target = self._new_rate_at_target(start_rate_at_target, linear_adaptation // 2)
    avg_rate_at_target = (start_rate_at_target + end_rate_at_target + mid_rate_at_target * 2) // 4

    borrow_rate = self._curve(avg_rate_at_target, err) * SEC_IN_YEAR
    supply_rate = self._w_mul_to_zero(self._w_mul_to_zero(borrow_rate, utilization),
                                      ONE_18_DEC - self.morpho_blue_market.fee)

    return MorphoBlueRates(supply_rate, borrow_rate)

  def _curve(self, _rate_at_target: int, err: int) -> int:
    coeff = ONE_18_DEC - self._w_div_to_zero(ONE_18_DEC, CURVE_STEEPNESS) if err < 0 else CURVE_STEEPNESS - ONE_18_DEC
    return self._w_mul_to_zero(self._w_mul_to_zero(coeff, err) + ONE_18_DEC, _rate_at_target)

  def _w_div_to_zero(self, x: int, y: int) -> int:
    q = ONE_18_DEC * x // y
    if q < 0:
      if x < 0:
        return -(ONE_18_DEC * -x // y)
      else:
        return -(ONE_18_DEC * x // -y)
    else:
      return ONE_18_DEC * x // y

  def _w_mul_to_zero(self, x: int, y: int) -> int:
    q = x * y
    if q < 0:
      if x < 0:
        return -(-x * y // ONE_18_DEC)
      else:
        return -(x * -y // ONE_18_DEC)
    else:
      return x * y // ONE_18_DEC

  def _new_rate_at_target(self, start_rate_at_target: int, linear_adaptation: int) -> int:
    x = self._w_mul_to_zero(start_rate_at_target, self._w_exp(linear_adaptation))

    if x < MIN_RATE_AT_TARGET:
      return MIN_RATE_AT_TARGET
    if x > MAX_RATE_AT_TARGET:
      return MAX_RATE_AT_TARGET

    return x

  def _w_exp(self, x: int) -> int:
    if x < LN_WEI_INT:
      return 0
    if x >= WEXP_UPPER_BOUND:
      return WEXP_UPPER_VALUE

    rounding_adjustment = LN_2_INT // 2 * (-1 if x < 0 else 1)
    q = (x + rounding_adjustment) // LN_2_INT
    r = x - q * LN_2_INT

    exp_r = ONE_18_DEC + r + r * r // ONE_18_DEC // 2
    return exp_r << q if q >= 0 else exp_r >> -q

  def increase_borrow(self, additional_borrow: int):
    self.morpho_blue_market.total_borrow_assets += additional_borrow

  def decrease_borrow(self, repay_amount: int):
    self.morpho_blue_market.total_borrow_assets -= repay_amount

  def increase_supply(self, supply: int):
    self.morpho_blue_market.total_supply_assets += supply

  def decrease_supply(self, supply: int):
    self.morpho_blue_market.total_supply_assets -= supply


def test():
  # @formatter:off
  morpho_blue_market = MorphoBlueMarket(
      fee=0,
      last_update=1713385655,
      total_borrow_assets=27801738669403,
      total_borrow_shares=27184523571503557254,
      total_supply_assets=35578540296522,
      total_supply_shares=34886692472094027030
  )

  market = MorphoBlueMarketMarket(
      morpho_blue_market=morpho_blue_market,
      current_timestamp=1713396875,
      rate_at_target=1897569358,
      morpho_blue_market_params=None
  )
  # @formatter:on

  result = market.rates()
  print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
  print("supply_rate: ", result.supply_rate / (10 ** 16), "%")

  market.increase_supply(35578540296522 // 10)
  result = market.rates()
  print('After increase supply')
  print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
  print("supply_rate: ", result.supply_rate / (10 ** 16), "%")


if __name__ == '__main__':
  test()
