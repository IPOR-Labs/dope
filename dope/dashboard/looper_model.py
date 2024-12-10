import pandas as pd

from dope.backengine.backtestdata import DataCollection, PriceCollection


class LoopModel:
    
    def get_df_and_caps(
        self,
        pool_dict,
        rates_data_collection: DataCollection,
        price_data_collection: PriceCollection,
        ):
        dfs = {}
        level_caps = {}
        
        add_rewards = True
        for k, p in pool_dict.items():
    
            debt = rates_data_collection[p.debt_rate_keyid].apyBaseBorrow
            debt_token_ret = ( (
                price_data_collection._price[p.debt_token]
                / price_data_collection._price[p.debt_token].shift()
            ) - 1 ) * 365 * 100

            if add_rewards:
                debt = (
                    rates_data_collection[p.debt_rate_keyid].apyBaseBorrow
                    + rates_data_collection[p.debt_rate_keyid].apyRewardBorrow
                    + debt_token_ret
                )
            else:
                debt = debt + debt_token_ret

            
            deposit = rates_data_collection[p.deposit_rate_keyid].apyBase
            deposit_token_ret = ( (
                price_data_collection._price[p.deposit_token]
                / price_data_collection._price[p.deposit_token].shift()
            ) - 1 ) * 365 * 100
            
            
            if add_rewards:
                deposit = (
                    rates_data_collection[p.deposit_rate_keyid].apyBase
                    + rates_data_collection[p.deposit_rate_keyid].apyReward
                    + deposit_token_ret
                )
            else:
                deposit = deposit + deposit_token_ret
                
            df = pd.DataFrame()
            df["debt"] = debt
            df["deposit"] = deposit
            
            # plotly complains if datetime has no timezone
            df.index = df.index.tz_localize('UTC')
            dfs[k] = df.dropna()
            level_caps[k] = 1/(1 - p.ltv)
        
        return dfs, level_caps
    

    def __call__(self, **params):

        df = params["df"]
        leverage = params["leverage"]
        timewindow = int(params["mavg"])

        debt_ratio = 1 - (1 / leverage) if leverage > 1 else 0
        apyLoop = (df.deposit - (df.debt * debt_ratio)) * leverage if leverage > 1 else df.deposit
        data = df.copy()
        data.index = pd.to_datetime(data.index)
        data["apyLoop"] = apyLoop
        data["Mvg. Avg."] = (apyLoop).rolling(timewindow).mean()
    
        return data.ffill()
