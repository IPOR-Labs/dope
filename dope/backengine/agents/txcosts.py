import numpy as np
import pandas as pd
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

import numpy as np
import cvxopt as opt
import sympy

from dope.backengine.agents.base import BaseAgent

@dataclass
class LenderQuadraticParams:
  columns: list
  sigmas: np.array
  mus: np.array
  cov: np.array
  inv_cov: np.array
  corr_matrix: np.array
  iota: np.array
  c_deposit: np.array
  capital: float
  
  def __iter__(self):
    for attr in dir(self):
      if not attr.startswith("__") and not callable(getattr(self, attr)):
        yield getattr(self, attr)
  

class VolType:
  up = "upvol"
  down = "downvol"
  justvol = "justvol"

class LenderMIQP(BaseAgent):
  vol_types = VolType

  def __init__(self, token, capital, risk_aversion=1, mean_window=1, cov_window=7, triggers=None, vol_type=VolType.justvol):
    """
    Mixed Integer Quadratic Programming Agent
    """
    self.token = token
    self.capital = capital
    self.engine = None
    # assert risk_aversion >= 0, "Risk aversion must be non negative"
    self.risk_aversion = risk_aversion
    self.mean_window = int(mean_window)
    self.cov_window = int(cov_window)
    self.triggers = triggers
    self.ws = {}
    
    self.rate_column = "apyBase"
    self.vol_type = vol_type
  
  def _get_optimizer_params(self, date_ix):
    rate_column = "apyBase"
    df = pd.concat(self.data[self.token], names=["datetime"]).unstack(level=0)[rate_column]
    #del df["cash"]
    df = df[df.index <= date_ix]
    mean_filter = df.index >= date_ix - pd.Timedelta(f"{self.mean_window}D")
    cov_filter = df.index >= date_ix - pd.Timedelta(f"{self.cov_window}D")
    if len(df) == 0:
      return None
    
    if self.vol_type == VolType.down:
      cov = np.matrix(df[cov_filter].diff().clip(upper=0).cumsum().cov())
    elif self.vol_type == VolType.up:
      cov = np.matrix(df[cov_filter].diff().clip(lower=0).cumsum().cov())
    else:
      cov = np.matrix(df[cov_filter].cov()) 

    sigma = np.matrix(df[cov_filter].std()).reshape(-1,1)
    mus = df[mean_filter].mean().values
    
    inv_cov = cov # np.linalg.inv(cov)
    corr_matrix = cov # np.matrix(df[cov_filter].corr())
    
    # deposit costs
    #c_deposit = self._get_depth_cost(date_ix)

    iota = np.matrix(np.ones((len(cov),1))) # len(cov) = 8
    
    my_capital = self.engine.get_capital(token=self.token)
    
    return LenderQuadraticParams(
      columns=df.columns, 
      sigmas=sigma,
      mus=mus,
      cov=cov,
      inv_cov=inv_cov,
      corr_matrix=corr_matrix,
      iota=iota,
      c_deposit=None, #c_deposit,
      capital=my_capital
      )

  def _get_depth_cost(self, date_ix):
    deposit_cost = [] 
    for protocol, df in self.data[self.token].items():
      # if protocol == "cash":
      #   continue
      # extra interest rate per percentage point of utilization rate
      _filter = df.index <= date_ix
      if len(df[_filter]) == 0:
        continue
      
      impact_1 = self.engine.mkt_impact[protocol].impact(
            date_ix,
            0,
            is_borrow=False
          )
      impact_2 = self.engine.mkt_impact[protocol].impact(
            date_ix,
            self.capital/2,
            is_borrow=False
          )
      slope = (impact_2 - impact_1)/(self.capital/2)
      if protocol == "cash":
        c = 0
      else:
        c = self.capital/df[_filter][self.rate_column].resample("1D").mean()[-1] * slope 
      deposit_cost.append(c)
  
    return np.array(deposit_cost)
    #return np.ones(n) *  deposit_cost

  def optimum_allocation(self, opt_params, date_ix, prec=None):
    """
    Notes about this functions: 
    https://github.com/IPOR-Labs/latex-papers/blob/main/yield-optimization/main.pdf
    """
    
    num_assets = len(self.opt_params.sigmas)
    
    r = self.opt_params.mus
    Q = self.opt_params.cov
    CAP = self.opt_params.capital if self.opt_params.capital > 0 else self.capital

    w0 = np.zeros(num_assets)


    fplus = np.array([10/CAP, 50/CAP, 0])  # Gas fees
    fminus = np.array([10/CAP, 50/CAP, 0])  # Gas fees

    num_assets = len(r)

    # Define the function to be minimized
    def f(params, Q, r):
        w = params[:num_assets]
        wplus = params[num_assets:2*num_assets]
        wminus = params[2*num_assets:3*num_assets]
        mkts = self.opt_params.columns
        cp = []
        cm = []
        for ix in range(len(mkts)):
          cp.append(self.engine.mkt_impact[mkts[ix]].impact(date_ix, capital=CAP*wplus[ix], is_borrow=True))
          cm.append(self.engine.mkt_impact[mkts[ix]].impact(date_ix, capital=CAP*wminus[ix], is_borrow=True))

        cm = np.array(cm)/100
        cp = np.array(cp)/100
        result = (
            
            + w.T @ (
              r
              - cp
              + cm
            )
        )
        if np.abs(self.risk_aversion) > 0.01:
          result += self.risk_aversion * (- w.T @ Q @ w )
        
        return - result

    # Initial guess for the parameters
    initial_guess = np.concatenate([
      w0, # w
      np.zeros(num_assets),  # wplus
      np.zeros(num_assets),  # wmins
    ])

    bounds = [(0, 1)] * num_assets + [(0, 1)] * (2 * num_assets)

    # Constraint: w + w_minus - w_plus == w0
    def constraint_func(params):
        w = params[:num_assets]
        wplus = params[num_assets:2*num_assets]
        wminus = params[2*num_assets:3*num_assets]
        return w + wminus - wplus - w0

    # Constraint: sum(w) == 1
    def sum_to_one_constraint(params):
        w = params[:num_assets]
        return np.sum(w) - 1

      
    # Constraint: w <= 1
    def w_less_equal_one_constraint(params):
        w = params[:num_assets]
        return 1 - w

    # Constraint: w_minus <= w0
    def w_minus_less_equal_w0_constraint(params):
        wminus = params[2*num_assets:3*num_assets]
        return w0 - wminus

    # Constraint: w_plus <= 1 - w0
    def w_plus_less_equal_one_minus_w0_constraint(params):
        wplus = params[num_assets:2*num_assets]
        return (1 - w0) - wplus

    constraints = [
        {
            'type': 'eq',
            'fun': constraint_func
        },
        {
            'type': 'eq',
            'fun': sum_to_one_constraint
        },
        {
            'type': 'ineq',
            'fun': w_less_equal_one_constraint
        },
        {
            'type': 'ineq',
            'fun': w_minus_less_equal_w0_constraint
        },
        {
            'type': 'ineq',
            'fun': w_plus_less_equal_one_minus_w0_constraint
        },

    ]


    best_result = minimize(
        f,
        initial_guess, 
        args=(Q, r), 
        constraints=constraints, 
        bounds=bounds, 
        method='SLSQP'
    )
    # Get the optimal parameters
    if best_result is not None:
        optimal_params = best_result.x
        w_opt = optimal_params[:num_assets]
        #wplus_opt = optimal_params[num_assets:2*num_assets]
        #wminus_opt = optimal_params[2*num_assets:3*num_assets]

        # print("Optimal parameters:")
        # print("w0", w0)
        # print("w =", w_opt)
        # print("wplus =", wplus_opt)
        # print("wminus =", wminus_opt)
        # print("E[r]", -best_result.fun)
    else:
        # print("No valid solution found.")
        return []

    if prec is not None:
      return np.round(w_opt, prec)
    return w_opt

  def act(self, date_ix):
    
    """
    date_ix is the date index NOW. 
    One can think as the index of the filtration \mathcal{F}_{ix}, i.e.,
    the increasing sequence of information sets where the agent acts.
    
    """
    if self.triggers is not None:
      if date_ix not in self.triggers:
        return self.ws
    print("Acting....", date_ix)
    
    self.opt_params = self._get_optimizer_params(date_ix)
    ws_list = self.optimum_allocation(self.opt_params, date_ix=date_ix, prec=None)
    #print("ws:",ws_list)
    ws = {self.opt_params.columns[i]:ws_list[i] for i in range(len(ws_list))}
    
    self.ws = {self.token:ws}
    # print(self.ws)
    return self.ws
