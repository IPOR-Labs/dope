
class Pool:
    
    def __init__(
        self,
        chain: str, 
        symbol: str,
        protocol: str,
        deposit_token: str,
        deposit_token_keyid: str, 
        debt_token: str,
        debt_token_keyid: str, 
        deposit_pool_id: str,
        debt_pool_id: str,
        ltv: float,
        LT:float=1,
        deposit_rate_keyid: str = None,
        debt_rate_keyid: str = None,
        meta: str = None, 
    ):
        """
        ltv: loan to value ratio (If pool has x value of collateral, it can borrow x*ltv)
        LT: liquidation threshold (If the value of the collateral falls below x*LT, the position is liquidated)
        in general LT > ltv
        """
        self.chain = chain
        self.symbol = symbol
        self.protocol = protocol
        self.meta = meta or ""
        self.deposit_token = deposit_token
        self.deposit_token_keyid = deposit_token_keyid
        self.debt_token = debt_token
        self.debt_token_keyid = debt_token_keyid
        self.ltv = ltv
        self.LT = LT
        self.deposit_rate_keyid = deposit_rate_keyid
        self.debt_rate_keyid = debt_rate_keyid
        self.deposit_pool_id = deposit_pool_id
        self.debt_pool_id = debt_pool_id
    
    def __repr__(self):
        debt_str = f"{self.debt_token}({str(self.debt_pool_id)[:10]})"
        deposit_str = f"{self.deposit_token}({str(self.deposit_pool_id)[:10]})"
        return f"{self.chain}:{self.protocol}:(debt:{debt_str}, deposit:{deposit_str})"
    
    @property
    def debt_name(self):
        return f"{self.debt_token}({self.debt_pool_id})"

    @property
    def deposit_name(self):
        return f"{self.deposit_token}({self.deposit_pool_id})"

        