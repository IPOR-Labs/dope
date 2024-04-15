import numpy as np
import cvxopt as opt
from cvxopt import matrix
from cvxopt.solvers import qp, options



def weights_with_cost():#r_array, Q, c, max_weight, turnover):

  # Loss Function
  n_assets_r = 20
  n = 200
  c = 0.001
  max_weight = 0.05
  r = opt.matrix(np.block([np.random.sample(n), -c * np.ones(2*n)]))
  # Constraint matrices and vectors

  A = opt.matrix(
    np.block(
      [
        [np.ones(n), c * np.ones(n), -c * np.ones(n)],
        [np.eye(n), np.eye(n), -np.eye(n)]
      ]
    )
  )
  old_x = np.zeros(n)
  old_x[np.random.choice(n, n_assets_r, replace=False)] = max_weight
  b = opt.matrix(np.block([1.0, old_x]))

  max_weight = 0.05
  turnover = 2.0
  # Q matrix such that it resembles the covariance matrix of returns
  T = np.random.randn(n,100)
  Q = np.cov(T)
  Q = opt.matrix(
    np.block(
      [
        [Q, np.zeros((n,n)), np.zeros((n,n))],
        [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))],
        [np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n))]
      ]
    )
  )
  # Create constraint matrices
  G = opt.matrix(0.0, (6 * n + 1, 3 * n))
  h = opt.matrix(0.0, (6 * n + 1, 1))
  for k in range(3 * n):
      # wi > 0 constraint
      G[k, k] = -1
      # wi > max_weight
      G[k + 3 * n, k] = 1
      h[k + 3 * n] = max_weight
  for k in range(2 * n):
      # sum dwi+ + dwi- < turnover
      G[6 * n, k + n] = 1
      
  h[6 * n] = turnover

  # Compute random portfolios in order to have a baseline
  n_random = 100
  random_returns = np.zeros(n_random)
  random_risks = np.zeros(n_random)

  options['show_progress'] = False
  for i in range(n_random):
      w0 = np.zeros(n)
      w0[np.random.choice(n, n_assets_r, replace=False)] = 1 / n_assets_r
      random_returns[i] = np.dot(w0, r[:n])
      random_risks[i] = np.dot(w0, np.dot(Q[:n,:n], w0))
  # Compute the optimal portfolio for different values of lambda, the risk aversion
  lmbdas = [10 ** (5.0 * t / n - 1.0) for t in range(n)]
  sol = [qp(lmbda / 2 * Q, -r, G, h, A, b)['x'] for lmbda in lmbdas]
  optimal_returns = np.array([opt.blas.dot(x, r) for x in sol])
  optimal_risks = np.array([opt.blas.dot(x, Q * x) for x in sol])
  
  return optimal_returns, optimal_risks, random_returns, random_risks