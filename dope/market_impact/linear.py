import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def zero_func(*args, **kwargs):
  return 0


class ZeroLinearMktImpactModel:
  def __(self, *args, **kwargs):
    pass
  
  def impact(self,*args, **kwargs):
    return 0
  
  def set_data_ref(self, *args, **kwargs):
    pass

class LinearMktImpactModel:
  def __init__(self, utilization_rate, apy):
    self.x = utilization_rate
    self.y = apy
    self.slopes = {}
    self._data_ref = None
  
  @classmethod
  def zero_instance(cls):
    model = ZeroLinearMktImpactModel()
    model.slopes = defaultdict(zero_func)
    return model

  def set_data_ref(self, data_ref):
    self._data_ref = data_ref
  
  @property
  def data_ref(self):
    return self._data_ref

  def get_slope(self, level):
    x = level
    # initializing right to a value that will never be reached unless enter loop
    right = x+1 
    for (left, right), v in self.slopes.items():
      if x <= left:
        return v
      if left <= x < right:
        return v
    if x >= right:
      return self.slopes[(left, right)]
      
    return self.slopes[(0,0)]
  
  def impact_simple(self, timestamp, capital, is_borrow):
    """
    is_borrow is a boolean that is true if the borrow rate should be used. otherwise use lend/supply rate.
    """

    _filter = self.data_ref.index <= timestamp
    row = self.data_ref[_filter].iloc[-1]
    ur0 = row.utilizationRate
    #totalSupplyUsd	totalBorrowUsd
    tvl_name = "totalBorrowUsd" if is_borrow else "totalSupplyUsd"

    if is_borrow:
      ur = (row["totalBorrowUsd"] + capital)/(row["totalSupplyUsd"])
    else:
      ur = row["totalBorrowUsd"]/(capital + row["totalSupplyUsd"])

    slope = self.get_slope(ur0)
    
    # the following line cancels out the bias (we do not keep the bias)
    impact = slope * (ur - ur0)

    return impact
  
  def impact(self, timestamp, capital, is_borrow):
    """
    is_borrow is a boolean that is true if the borrow rate should be used. otherwise use lend/supply rate.
    """

    _filter = self.data_ref.index <= timestamp
    row = self.data_ref[_filter].iloc[-1]
    #ur0 = row.utilizationRate
    #totalSupplyUsd	totalBorrowUsd
    tvl_name = "totalBorrowUsd" if is_borrow else "totalSupplyUsd"

    if is_borrow:
      ur1 = (row["totalBorrowUsd"] + capital)/(row["totalSupplyUsd"])
    else:
      ur1 = row["totalBorrowUsd"]/(capital + row["totalSupplyUsd"])
    ur0 = row["totalBorrowUsd"]/(row["totalSupplyUsd"])

    # N.B.: assumption that intervals are ordered
    intervals = self.slopes.keys()
    
    is_counting = False
    deltas = {}
    ur_left = min(ur0, ur1)
    ur_right = max(ur0, ur1)
    for (left, right) in intervals:
      
      if (left <= ur_left < right) and (left <= ur_right < right):
        # same interval
        deltas[(left, right)] = ur_left - ur_right
        break
      
      if left <= ur_left < right:
        deltas[(left, right)] = ur_left - right
        is_counting = True
      elif left <= ur0 < right:
        # got to initial ur interval
        deltas[(left, right)] = left - ur_right
        is_counting = False # stop counting
      elif is_counting:
        # between ur1 and ur0
        deltas[(left, right)] = left - right
      else:
        # done conting or not yeat conting
        deltas[(left, right)] = 0
    # print(ur1, ur0)
    # print(ur_left, ur_right)
    # print(deltas)
    # print(self.slopes)
    # print([self.slopes[(left, right)] * deltas[(left, right)] for (left, right) in deltas.keys()])
    impact = sum([self.slopes[(left, right)] * deltas[(left, right)] for (left, right) in deltas.keys()])
    sign = 1 if ur1 < ur0 else -1
    #print(f"sign: {sign}")
    # # the following line cancels out the bias (we do not keep the bias)
    # impact = slope * (ur - ur0)
    
    return sign * impact
  
  def fit(self, kinks=None, should_plot=False):
    self.slopes = {}
    # Example data
    x = self.x
    y = self.y
    if kinks is None:
      _slope = self._fit_one(x, y, should_plot=should_plot)
      self.slopes = defaultdict(lambda: _slope)
    else:
      _kinks = [0] + kinks + [1]
      for i in range(len(_kinks)-1):
        start, end = _kinks[i], _kinks[i+1]
        x_i = x[(x>=start) & (x<end)]
        y_i = y[(x>=start) & (x<end)]
        if len(x_i) == 0:
          continue
        _slope = self._fit_one(x_i, y_i, should_plot=should_plot)
        self.slopes[(start, end)] = max(0, _slope) # no negative slopes
    if should_plot:
      plt.show()

    return self

  def _fit_one(self, x, y, should_plot=False):
    if np.all(y == 0):
      slope, bias = 0, 0  
    else:
      try:
        slope, bias = np.polyfit(x, y, 1)
      except ValueError as e:
        print(e)
        slope, bias = 0, 0
      #print(slope, bias)
    # Plot the data and the line
    if should_plot:
      self.plot(x,y, slope, bias)
    return slope
  
  def plot(self, x, y, slope, bias):
    y_pred = slope * x + bias
    
    plt.plot(x, y, 'o', color="tab:blue")
    plt.plot(x, y_pred, '-', label=f'Fitted line ($\\alpha$={slope:.3f})')

    plt.xlabel("Utilization Rate")
    plt.ylabel("Borrow Rates")

    plt.legend()
    plt.title(f"Slope: {slope:,.3f}, Bias: {bias:,.3f}")
 
    
    
