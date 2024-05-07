class Rates:
    def __init__(self, supply_rate: int, borrow_rate: int):
        self.supply_rate = supply_rate
        self.borrow_rate = borrow_rate


class CompoundV3Market:
    SECONDS_PER_YEAR = 60 * 60 * 24 * 365

    def __init__(self, total_borrows, total_supply, supply_per_second_interest_rate_base, supply_kink,
                 supply_per_second_interest_rate_slope_low, supply_per_second_interest_rate_slope_high,
                 borrow_per_second_interest_rate_base, borrow_kink, borrow_per_second_interest_rate_slope_low,
                 borrow_per_second_interest_rate_slope_high):
        self.total_borrows = total_borrows
        self.total_supply = total_supply
        self.supply_per_second_interest_rate_base = supply_per_second_interest_rate_base
        self.supply_kink = supply_kink
        self.supply_per_second_interest_rate_slope_low = supply_per_second_interest_rate_slope_low
        self.supply_per_second_interest_rate_slope_high = supply_per_second_interest_rate_slope_high
        self.borrow_per_second_interest_rate_base = borrow_per_second_interest_rate_base
        self.borrow_kink = borrow_kink
        self.borrow_per_second_interest_rate_slope_low = borrow_per_second_interest_rate_slope_low
        self.borrow_per_second_interest_rate_slope_high = borrow_per_second_interest_rate_slope_high

    def increase_borrow(self, additional_borrow: int):
        self.total_borrows += additional_borrow

    def decrease_borrow(self, repay_amount: int):
        self.total_borrows -= repay_amount

    def increase_supply(self, supply: int):
        self.total_supply += supply

    def decrease_supply(self, supply: int):
        self.total_supply -= supply

    def calculate_compound_v3_supply_rate(self, utilization_rate: int) -> int:
        if utilization_rate <= self.supply_kink:
            return (utilization_rate * self.supply_per_second_interest_rate_slope_low // (
                    10 ** 18)) + self.supply_per_second_interest_rate_base
        else:
            normal_rate = (self.supply_kink * self.supply_per_second_interest_rate_slope_low // (
                    10 ** 18)) + self.supply_per_second_interest_rate_base
            excess_utilization_rate = utilization_rate - self.supply_kink
            return (excess_utilization_rate * self.supply_per_second_interest_rate_slope_high // (
                    10 ** 18)) + normal_rate

    def calculate_compound_v3_borrow_rate(self, utilization_rate: int) -> int:
        if utilization_rate <= self.borrow_kink:
            return (utilization_rate * self.borrow_per_second_interest_rate_slope_low // (
                    10 ** 18)) + self.borrow_per_second_interest_rate_base
        else:
            normal_rate = (self.borrow_kink * self.borrow_per_second_interest_rate_slope_low // (
                    10 ** 18)) + self.borrow_per_second_interest_rate_base
            excess_utilization_rate = utilization_rate - self.borrow_kink
            return (excess_utilization_rate * self.borrow_per_second_interest_rate_slope_high // (
                    10 ** 18)) + normal_rate

    def rates(self) -> Rates:
        utilization_rate = self.total_borrows * (10 ** 18) // self.total_supply
        borrow_rate_per_second = self.calculate_compound_v3_borrow_rate(utilization_rate)
        supply_rate_per_second = self.calculate_compound_v3_supply_rate(utilization_rate)
        supply_rate = supply_rate_per_second * self.SECONDS_PER_YEAR
        borrow_rate = borrow_rate_per_second * self.SECONDS_PER_YEAR
        return Rates(supply_rate, borrow_rate)


def test():
    market = CompoundV3Market(total_borrows=454299260393634, total_supply=560241660778185,
                              supply_per_second_interest_rate_base=0, supply_kink=900000000000000000,
                              supply_per_second_interest_rate_slope_low=2378234398,
                              supply_per_second_interest_rate_slope_high=114155251141,
                              borrow_per_second_interest_rate_base=475646879, borrow_kink=900000000000000000,
                              borrow_per_second_interest_rate_slope_low=2641425672,
                              borrow_per_second_interest_rate_slope_high=136352105530)

    result = market.rates()
    print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
    print("supply_rate: ", result.supply_rate / (10 ** 16), "%")

    market.increase_supply(560241660778185 / 2)
    result = market.rates()
    print('After increase supply')
    print("borrow_rate: ", result.borrow_rate / (10 ** 16), "%")
    print("supply_rate: ", result.supply_rate / (10 ** 16), "%")


if __name__ == '__main__':
    test()
