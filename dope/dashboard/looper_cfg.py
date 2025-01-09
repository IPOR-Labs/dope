from dataclasses import dataclass
from builtins import dict

@dataclass
class Config:
  show_upper_band: bool = False
  show_lower_band: bool = False
  show_ipor: bool = True
  show_rec: bool = True
  show_pay: bool = True
  show_accipor: bool = True
  show_mavg_low_vol: bool = False
  show_mavg_high_vol: bool = False
  show_mavg_dynamic_cap: bool = False


def get_plot_cfg():
  
  plot_cfg = {

    

    "apyLoop": {
      "df_col":"apyLoop", 
      "scatter_cfg":{
        "mode":'lines',
        "line_shape":'hv',
        "name":"apyLoop",
        "line":dict(color='blue', width=1)
      },
      "row":1, "col":1,
      "should_show_col": "show_pay"
    },
    
    "Mvg. Avg.": {
      "df_col":"Mvg. Avg.", 
      "scatter_cfg":{
        "mode":'lines',
        "line_shape":'hv',
        "name":"Mvg. Avg.",
        "line":dict(color='grey', width=1)
      },
      "row":1, "col":1,
      "should_show_col": "show_ipor"
    },
 
    "Debt": {
      "df_col":"debt", 
      "scatter_cfg":{
        "mode":'lines',
        "line_shape":'hv',
        "name":"Debt",
        "line":dict(color='red', width=1)
      },
      "row":4, "col":1,
      #"should_show_col": "show_lower_band"
    },
    
    "Deposit": {
      "df_col":"deposit", 
      "scatter_cfg":{
        "mode":'lines',
        "line_shape":'hv',
        "name":"Deposit",
        "line":dict(color='green', width=1)
      },
      "row":4, "col":1,
      #"should_show_col": "show_lower_band"
    },
    
    "30 Days Drawdown": {
      "df_col":"drawdown", 
      "scatter_cfg":{
        "mode":'lines',
        "line_shape":'hv',
        "name":"30 Days Drawdown",
        "line":dict(color='grey', width=2)
      },
      "row":3, "col":1,
      #"should_show_col": "show_lower_band"
    },
    
    "Cum. Returns": {
      "df_col":"cumLoopApy", 
      "scatter_cfg":{
        "mode":'lines',
        "line_shape":'hv',
        "name":"Cum. Returns",
        "line":dict(color='black', width=2)
      },
      "row":2, "col":1,
      #"should_show_col": "show_lower_band"
    },
    
    
    
  }

  return plot_cfg