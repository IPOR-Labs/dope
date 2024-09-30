
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
        ltv: float,        
        deposit_pool_id: str,
        debt_pool_id: str,
        deposit_rate_keyid: str = None,
        debt_rate_keyid: str = None,
        meta: str = None, 
    ):
        self.chain = chain
        self.symbol = symbol
        self.protocol = protocol
        self.meta = meta or ""
        self.deposit_token = deposit_token
        self.deposit_token_keyid = deposit_token_keyid
        self.debt_token = debt_token
        self.debt_token_keyid = debt_token_keyid
        self.ltv = ltv
        self.deposit_rate_keyid = deposit_rate_keyid
        self.debt_rate_keyid = debt_rate_keyid
        self.deposit_pool_id = deposit_pool_id
        self.debt_pool_id = debt_pool_id
    
    def __repr__(self):
        return f"Pool(debt:{self.debt_rate_keyid}, deposit:{self.deposit_rate_keyid})"
        
        