import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class LinearMktImpactModel:
  def __init__(self, utilization_rate, apy):
    self.x = utilization_rate
    self.y = apy
    self.slopes = {}
  
  @classmethod
  def zero_instance(cls):
    model = cls([],[])
    model.slopes = defaultdict(lambda: 0)
    return model

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
  
  def impact(self, level, delta):
    slope = self.get_slope(level)
    return slope * delta
  
  def fit(self, kinks=None, should_plot=False):
    self.slopes = {}
    # Example data
    x = self.x

    y = self.y
    if kinks is None:
      _slope = self._fit_one(x, y)
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

    return self

  def _fit_one(self, x, y, should_plot=False):
    line_fit = np.polyfit(x, y, 1)
  
    slope, bias = np.poly1d(line_fit)    
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
 
    
    
