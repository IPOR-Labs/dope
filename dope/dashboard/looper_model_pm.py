import pandas as pd

from dope.backengine.backtestdata import DataCollection


class LoopModel:
    
    def get_df_and_caps(self, pool_dict, rates_data_collection: DataCollection):
        dfs = {}
        level_caps = {}
        
        for k, p in pool_dict.items():
            add_rewards = True
            debt = rates_data_collection[p.debt_rate_keyid].apyBaseBorrow
            if add_rewards:
                debt = (
                    rates_data_collection[p.debt_rate_keyid].apyBaseBorrow
                    + rates_data_collection[p.debt_rate_keyid].apyRewardBorrow
                )

            
            deposit = rates_data_collection[p.deposit_rate_keyid].apyBase
            if add_rewards:
                deposit = (
                    rates_data_collection[p.deposit_rate_keyid].apyBase
                    + rates_data_collection[p.deposit_rate_keyid].apyReward
                )
                
            df = pd.DataFrame()
            df["debt"] = debt
            df["deposit"] = deposit

            dfs[str(k)] = df
            level_caps[str(k)] = 1/(1 - p.ltv)
        
        
        
        return dfs, level_caps
    

    def __call__(self, **params):
        
        # print("params",params)
        
        df = params["df"]
        leverage = params["leverage"]
        threshold = params["threshold"]

        w_long = int(params["turn_on_off"])
        w_short = int(params["short_moving_avg"])

        debt_ratio = 1 - (1 / leverage) if leverage > 1 else 0
        apyLoop = (df.deposit - (df.debt * debt_ratio)) * leverage if leverage > 1 else df.deposit
        data = pd.DataFrame(index=df.index)
        data.index = pd.to_datetime(data.index)
        data["apyLoop"] = apyLoop
        data["apyLoop/leverage"] = apyLoop/leverage
        data["apyLoop/leverage short mavg"] = (apyLoop/leverage).rolling(w_short).mean()
        data["Turn Off Threshold"] = (apyLoop/leverage).rolling(w_long).mean() -  threshold
        data["Turn On Mvg. Avg."] = (apyLoop/leverage).rolling(w_long).mean()
    
        return data.ffill()
