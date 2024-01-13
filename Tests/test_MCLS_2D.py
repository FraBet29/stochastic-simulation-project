import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import solve_ivp
from question2 import *

def ff(x, y):
    return f(x) * f(y)

def MCLS_2D_test(a_samples, b_samples, ff, n):
    M = len(a_samples)
    # compute the 2D Vandermonde matrix
    x = np.random.uniform(0, 1, M)
    y = np.random.uniform(0, 1, M)
    V = np.polynomial.legendre.legvander2d(shifted(x, 0, 1), shifted(y, 0, 1), (n, n))
    cond = np.linalg.cond(V)
    # solve the least squares problem
    c = np.linalg.lstsq(V, ff(x, y), rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    # compute the estimator
    estim = np.sum(ff(a_samples, b_samples) - np.polynomial.legendre.legval2d(shifted(a_samples, 0, 1), shifted(b_samples, 0, 1), c)) / M + c[0, 0]
    return estim, cond

def MCLS_prime_2D_test(a_samples, b_samples, ff, n):
    M = len(a_samples)
    # compute the 2D Vandermonde matrix
    V = np.polynomial.legendre.legvander2d(shifted(a_samples, 0, 1), shifted(b_samples, 0, 1), (n, n))
    cond = np.linalg.cond(V)
    # solve the least squares problem
    c = np.linalg.lstsq(V, ff(a_samples, b_samples), rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    # compute the estimator
    estim = c[0, 0]
    return estim, cond