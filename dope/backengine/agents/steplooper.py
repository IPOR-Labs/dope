import numpy as np

from dope.backengine.agents.base import BaseAgent
from dope.token import Token

def total_collateral(n, X, ltv):
            return X * (1-ltv**(n+1)) / (1- ltv)
def total_borrow(n, X, ltv):
    return X * (1-ltv**(n+1)) / (1- ltv) * ltv

class StepLooper(BaseAgent):

    def __init__(self, capital, loop_n, buffer, start_pool, loop_pool, *args, **kwargs):
        super().__init__(capital=capital, *args, **kwargs)

        self.start_pool = start_pool
        self.loop_pool = loop_pool
        self.loop_n = loop_n
        self.buffer = buffer
        self.verbose = False
        self.ws = {}
    
    @property
    def leverage(self):
        n = self.loop_n
        A = 1 - (self.start_pool.ltv)**(n + 1)
        B = 1 - self.start_pool.ltv
        return A/B

    @classmethod
    def leverage_from_loops(cls, loops, start_pool):
        LTV = start_pool.ltv
        n = loops
        A = 1 - (LTV)**(n + 1)
        B = 1 - LTV
        return A / B 
    
    @classmethod
    def loops_from_leverage(cls, leverage, start_pool, epsilon=0.1, verbose=False):
        max_leverage = 1/(1 - start_pool.ltv)
        if leverage > max_leverage:
            print(f"Max leverage is {max_leverage: .3f}. Returning at max leverage {max_leverage - epsilon}.")
            L = max_leverage - epsilon
        else:
            L = leverage

        LTV = start_pool.ltv
        loops = int(np.log((1 - L * (1-LTV))) / np.log(LTV) - 1)
        if verbose:
            actual_leverage = cls.leverage_from_loops(loops, start_pool)
            print(f"Loops: {loops}, Actual leverage: {actual_leverage: .3f}")
        return loops
    
    def do_loop(
        self,
        init_capital: Token,
        start_pool,
        loop_pool,
        loop_times
    ):
        loop_pool = self.engine.pools[loop_pool]
        start_pool = self.engine.pools[start_pool]
        
        # initial deposit
        x = init_capital
        if self.verbose:
            print(
                "Loop Algebra Prediction",
                "Deposit:", total_collateral(loop_times, x, start_pool.ltv-self.buffer), 
                "Debt:", total_borrow(loop_times-1, x, start_pool.ltv-self.buffer)
            )

        x = self.engine.add_deposit_token(start_pool, x)
        price_deposit = self.engine.price_of_token(start_pool.deposit_token)
        price_debt = self.engine.price_of_token(start_pool.debt_token)
        price = price_deposit / price_debt

        x = self.engine.take_debt_token(
            start_pool, 
            Token(
                x.value * price * (loop_pool.ltv), 
                name=loop_pool.debt_token
            )
        )

        x = self.engine.swap(
            x.value,
            from_token=start_pool.debt_token,
            to_token=loop_pool.deposit_token,
        ) 

        for _ in range(loop_times):

            x = self.engine.add_deposit_token(loop_pool, x)
            x = self.engine.take_debt_token(
                loop_pool,
                Token(
                    x.value * (loop_pool.ltv - self.buffer), 
                    name=loop_pool.debt_token
                ),
            )
            x = self.engine.swap(
                x.value,
                from_token=loop_pool.debt_token,
                to_token=loop_pool.deposit_token,
            )

        x = self.engine.add_deposit_token(loop_pool, x)
        
        if self.verbose:
            print("Account after loop:")
            print(self.engine.π.accounts)

    def on_start(self):
        self.do_loop(
            self.capital,
            start_pool=self.start_pool,
            loop_pool=self.loop_pool,
            loop_times=self.loop_n
        )
        date_ix = self.engine.get_time()
        self.ws = self.engine.weights(self.engine.price_data.price_row_at(date_ix))
        
        if self.verbose:
            print("ws on_start", self.ws)

        return 

    def on_liquidation(self):
        return {}

    def on_act(self):
        """
        date_ix is the date index NOW.
        One can think as the index of the filtration $\\mathcal{F}_{ix}$, i.e.,
        the increasing sequence of information sets where the agent is acts.

        """
        date_ix = self.engine.get_time()
        if self.verbose:
            print("Acting....", date_ix)
        return self.ws
