import pandas as pd
import numpy as np

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
            price_data_collection.set_base_token_name(p.debt_token)
            price_data_collection.set_up_price_timeseries()
    
            debt = rates_data_collection[p.debt_rate_keyid].apyBaseBorrow
            debt_token_price = price_data_collection._price[p.debt_token]
            debt_token_ret = ( (
                debt_token_price
                / debt_token_price.shift()
            ) - 1 )  * 100 * 365

            if add_rewards:
                debt = (
                    rates_data_collection[p.debt_rate_keyid].apyBaseBorrow
                    + rates_data_collection[p.debt_rate_keyid].apyRewardBorrow
                )            
            debt = (((1+debt/100/365) * (1+debt_token_ret/100/365))-1)*100*365

            deposit = rates_data_collection[p.deposit_rate_keyid].apyBase
            deposit_token_price = price_data_collection._price[p.deposit_token]
            deposit_token_ret = ( (deposit_token_price / deposit_token_price.shift()) - 1 ) * 100 * 365
            
            
            if add_rewards:
                deposit = (
                    rates_data_collection[p.deposit_rate_keyid].apyBase
                    + rates_data_collection[p.deposit_rate_keyid].apyReward
                )
            deposit = (((1+deposit/100/365) * (1+deposit_token_ret/100/365))-1)*100*365

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
        
        dt = data.index.diff().mean().total_seconds()
        dt = dt / (60 * 60 * 24)
        
        window_size = int(30 * dt)

        data["cumLoopApy"] = (np.exp(np.log(1+data['apyLoop']/100 * 1/365 ).cumsum()) - 1)*100
        data['roll_max'] = data['cumLoopApy'].rolling(window=window_size).max()
        data['drawdown'] = (data['cumLoopApy'] - data['roll_max'])
    
        return data.ffill()
