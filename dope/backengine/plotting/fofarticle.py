# fund of fund article misc functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
  def __init__(self, summary):
    self.summary = summary

  def summary_view(self, summary=None, title_2="",  xlim=None, ylim=None):
    summary = summary or self.summary
    
    plt.figure(figsize=(16,4))
    ax = plt.gca()
    summary.rate.resample("1h").last().interpolate().plot(ax=ax)
    _ = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Strategy")
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optionally, you can also move the left and bottom spines
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlabel("")
    ax.set_ylabel("Returns (in %)")
    if xlim is not None:
        _ = plt.xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))
    plt.show()

    r_t = ((summary.capital/summary.capital.apply(lambda col: col.dropna().iloc[0]) - 1)*100)
    r_t = r_t.resample("1h").last().interpolate()
    T = 365.25 * 24 * 60 * 60
    annualized_returns = r_t.apply(
      lambda col: col.dropna().iloc[-1] * T / (col.dropna().index[-1] - col.dropna().index[0]).total_seconds()
    )
    annualized_vol = r_t.apply(
      lambda col: 
        col.dropna().std() 
        * np.sqrt(T / (col.dropna().index[-1] - col.dropna().index[0]).total_seconds())
    )
    plt.figure(figsize=(16,4))
    ax = plt.gca()
    annualized_rt = r_t.apply(
      lambda col: col.dropna() * T / (col.dropna().index[-1] - col.dropna().index[0]).total_seconds()
    )
    annualized_rt.plot(ax=ax, linewidth=4)
    ax.set_title(title_2, fontsize=22)
    ax.set_xlabel("")
    ax.set_ylabel("Returns (in %)")
    _ = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Strategy")
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optionally, you can also move the left and bottom spines
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    if xlim is not None:
        _ = plt.xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))
    if ylim is not None:
        _ = plt.ylim(ylim[0], ylim[1])
    plt.show()

    plt.figure(figsize=(6,6))
    returns = annualized_returns
    vols = annualized_vol

    colors = plt.cm.tab10(np.linspace(0, 1, len(returns))[::])
    COLS = returns.sort_values(ascending=False).index
    for i in range(len(COLS)):
      c = COLS[i]
      if "arb" in c:
        marker = "x"
      elif "jack" in c:
        marker = "s"
      else:
        marker = "o"
      plt.scatter(vols[c], returns[c], label=c, color=colors[i], marker=marker)
    _ = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Strategy (Higher PnL First)")

    plt.title(f"Risk vs Return")
    plt.show()
    print(returns)
    
  def plot_pools(self, borrow_lend_data, xlim=None, title_prefix=""):
    color_cls = {"DAI": plt.cm.Oranges, "USDC":plt.cm.Blues, "USDT":plt.cm.Greens}

    for token in borrow_lend_data.keys():
      if "cash" in borrow_lend_data[token]:
        del borrow_lend_data[token]["cash"]
      n_pools = len(borrow_lend_data[token])
      colors = color_cls[token](np.linspace(0, 0.8, n_pools))
      plt.figure(figsize=(16,4))
      ax = plt.gca()
      borrow_lend_data.to_block(token).apyBase.resample("3D").mean().plot(color=colors, legend=False, ax=ax)
      #_ = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
      if xlim is not None:
        _ = plt.xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))
      plt.title(f"{title_prefix}Pool Daily Yield. Token:{token} ({n_pools:,} pools)")

      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.spines['left'].set_position(('outward', 10))
      ax.spines['bottom'].set_position(('outward', 10))
      ax.set_xlabel("")
      ax.set_ylabel("Pool Yield (in %)")
      plt.show()
  
  def plot_arb_routes(self, run, xlim=None):
    color_counter = {"aave-v3":0, "aave-v2":0, "compound-v2":0}
    for token in run.run_data.keys():
      for k, df in run.run_data[token].items():
        if k == "cash": 
          continue
        for protocol in color_counter.keys():
          if protocol in k.split(":")[1]:
            color_counter[protocol] +=1
    color_counter

    colors = {
      "aave-v3": plt.cm.Blues(np.linspace(0, 0.5, color_counter["aave-v3"])[::-1]),
      "aave-v2": plt.cm.Reds(np.linspace(0, 0.5, color_counter["aave-v2"])[::-1]),
      "compound-v2": plt.cm.Greens(np.linspace(0, 0.5, color_counter["compound-v2"])[::-1])
    }
    plt.figure(figsize=(20,7))

    color_counter = {"aave-v3":0, "aave-v2":0, "compound-v2":0}
    for token in run.run_data.keys():
      for k, df in run.run_data[token].items():
        if k == "cash": 
          continue
        for protocol in color_counter.keys():
          if protocol in k.split(":")[1]:
            color_counter[protocol] +=1
    color_counter

    c_left, c_right =  0.4, 0.5
    colors = {
      "aave-v3": plt.cm.Blues(np.linspace(c_left, c_right, color_counter["aave-v3"]+1)[::1]),
      "aave-v2": plt.cm.Reds(np.linspace(c_left, c_right, color_counter["aave-v2"]+1)[::1]),
      "compound-v2": plt.cm.Greens(np.linspace(c_left, c_right, color_counter["compound-v2"]+1)[::1])
    }

    did_label = {"aave-v3":False, "aave-v2":False, "compound-v2":False}

    for token in run.run_data.keys():
      plt.figure(figsize=(20,7))
      for k, df in run.run_data[token].items():
        if k == "cash": 
          continue
        label = k.split(":")[1]
        _label = label
        if did_label[label]:
          _label = None
        else:
          did_label[label] = True
        plt.plot(df.apyBase, label=_label, color=colors[label][color_counter[label]], alpha=0.3)
        color_counter[label] -= 1

      best_triangle = run.run_data.to_block("USDC").apyBase.mean().sort_values().index[-1]
      plt.plot(run.run_data[token][best_triangle].apyBase, "--", color="black", label="Best Route")  
      _ = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Starting Protocol")

      if xlim is not None:
        _ = plt.xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))
        #_ = plt.xlim(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-07-01"))
      ax = plt.gca()
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      ax.spines['left'].set_position(('outward', 10))
      ax.spines['bottom'].set_position(('outward', 10))
      ax.set_xlabel("")
      ax.set_ylabel("Pool Yield (in %)")

    plt.show()
    r_t = (((run.run_data[token][best_triangle].apyBase/100/365 + 1).cumprod()-1)*100)
    plt.plot(r_t)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlabel("")
    ax.set_ylabel("Pool Yield (in %)")
    plt.title(f"[{best_triangle}]")
    plt.show()
