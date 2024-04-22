import plotly.graph_objects as go
import plotly.express as px
import numpy as np

import numpy as np
import cvxopt as opt
from cvxopt import matrix
from cvxopt.solvers import qp, options
options['show_progress'] = False

import sympy

from dope.backengine.agents.base import BaseAgent

class MktImpactLender(BaseAgent):

  def __init__(self, capital, risk_aversion=0):
    self.capital = capital
    self.engine = None
    self._lambda = risk_aversion


  def act(self, date_ix):
    _data = {}
    for protocol, df in self.data.items():
      _filter = df.index <= date_ix
      if len(df[_filter]) == 0:
        continue
      _data[protocol] = df[_filter]

    return self.mktImpactOptimizer(_data)

  def mktImpactOptimizer(self, data):

    def corr(pool_1, pool_2):
      df_1 = data[pool_1].apy.copy()
      df_2 = data[pool_2].apy.copy()
      df_1 = df_1[df_1.index.isin(df_2.index)]
      df_2 = df_2[df_2.index.isin(df_1.index)]

      return np.corrcoef(df_1, df_2)[0,1]


    def cov(pool_1, pool_2):
      df_1 = data[pool_1].apy.copy()
      df_2 = data[pool_2].apy.copy()
      df_1 = df_1[df_1.index.isin(df_2.index)]
      df_2 = df_2[df_2.index.isin(df_1.index)]

      return np.cov(df_1, df_2)[0,1]

    def mu(pool_1):
      return data[pool_1].apy.rolling("7D").mean().iloc[-1]


    rows_corr = []
    sigmas = []
    mus = []
    for p1 in data.keys():
      #if p1 == "morpho-aave": continue
      #print(p1)
      rows = []
      for p2 in data.keys():
        #if p2 == "morpho-aave": continue
        rows.append(corr(p1,p2))
        if p1 == p2:
          sigmas.append(np.sqrt(cov(p1, p2)))
          mus.append(mu(p1))
      rows_corr.append(rows)    


    sigma = np.matrix(sigmas).reshape(-1,1)
    mu = np.matrix(mus).reshape(-1,1)

    corr_matrix = np.matrix(rows_corr)
    cov = np.matrix(np.diag(sigmas)) * corr_matrix * np.matrix(np.diag(sigmas))
    inv_cov = np.linalg.inv(cov)

    iota = np.matrix(np.ones((len(cov),1))) # len(cov) = 8

    n = len(data)
    max_weight = 1

    # slope = 4.8335110905540715#/100
    # slope = 19.633
    # slope = 6.571
    CAP = self.capital
    # deposit less money, more rates
    slope = {key:self.engine.mkt_impact[key].get_slope(data[key].utilizationRate.iloc[-1]) for key in data.keys()}
    cplus = -np.ones(n) * [CAP/data[key].tvlUsd.resample("7D").mean()[-1] * slope[key] for key in data.keys()]
    # We are not modeling withdraw impact (because we do not have money in the market)
    cplus = -np.ones(n) * [0 * slope[key] for key in data.keys()] 
    # deposit more money, lower rates
    cminus = np.ones(n) * [CAP/data[key].tvlUsd.resample("7D").mean()[-1] * slope[key] for key in data.keys()]
    #cminus = np.ones(n) * [0 * slope for key in data.keys()] 

    r_t =  np.ones(n) * mus
    #print(sympy.Matrix(np.matrix(r_t)))
    #return r_t, cminus

    max_weight = 1
    r = opt.matrix(np.block([r_t - cminus]))

    A = opt.matrix(
      np.block(
        [
          [np.ones(n)],
          #[np.ones(len(mus)) * mus],

        ]
      )
    )
    sorted_indices = sorted(range(len(sigma)), key=lambda i: sigma[i], reverse=True)
    second_hiest_sigma_return = (mus[sorted_indices[0]] + mus[sorted_indices[0]])/2
    B = opt.matrix(np.block(
      [1.0,
      # second_hiest_sigma_return
      ]))

    Q = opt.matrix(
      np.block(
        [cov]
      )
    )
    #print(sympy.Matrix(np.matrix(Q)).__repr__())
    #print(np.matrix(Q).shape)

    # Create constraint matrices
    G = opt.matrix(
      np.block(
        [
          [-np.eye(n)],
          [np.eye(n)]
        ]
      )
    )
    h = opt.matrix(
      np.block(
        [
          [np.zeros(n), max_weight*np.ones(n)],
        ]
      ).T
    )
    sol = qp(self._lambda / 2 * Q, -r, G, h, A, B)
    #sympy.Matrix(np.matrix(sol["x"]))
  
    return {k:w for k,w in zip(data.keys(),list(sol["x"]))}