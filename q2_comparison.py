import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def f(x):
    return 1/(25*x**2+1)
def LS_Vandermonde(n, M):
    x = np.random.uniform(0, 1, M)
    x2 = 2 *x - 1
    # compute the Vandermonde matrix
    V = np.polynomial.legendre.legvander(x2, n)
    # rescale to have orthonormal polynomials
    V = V * np.tile(np.sqrt(2 * np.arange(0, n + 1) + 1), (M, 1))
    # solve the least squares problem
    c = np.linalg.lstsq(V, f(x2), rcond=None)[0]
    return c

def LS_fit(n, M):
    x = np.random.uniform(0, 1, M)
    c = np.polynomial.legendre.Legendre.fit(x, f(x), n, domain=[0, 1])
    c = c.coef
    return c / np.sqrt(2 * np.arange(0, n+1) + 1)

def MCLS_V_V(samples, f, n):

    # compute the Vandermonde matrix
    samples2 = samples*2 - 1
    c = LS_Vandermonde(n, len(samples))
    V = np.polynomial.legendre.legvander(samples2, n)
    # rescale to have orthonormal polynomials
    V = V * np.tile(np.sqrt(2 * np.arange(0, n + 1) + 1), (len(samples), 1))
    # compute the estimator
    estim = np.sum(f(samples) - c @ V.T) / len(samples) + c[0]
    return estim

def MCLS_V_legval(samples, f, n):
    samples2 = samples*2 - 1
    c = LS_Vandermonde(n, len(samples))
    # rescale the coefficients
    c = c * np.sqrt(2 * np.arange(0, n + 1) + 1)
    # compute the estimator
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]
    return estim

def MCLS_fit_legval(samples, f, n):
    samples2 = samples*2 - 1
    c = LS_fit(n, len(samples))
    # rescale the coefficients
    c = c * np.sqrt(2 * np.arange(0, n + 1) + 1)
    # compute the estimator
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]
    return estim

def MCLS_fit_V(samples, f, n):
    samples2 = samples*2 - 1
    c = LS_fit(n, len(samples))
    V = np.polynomial.legendre.legvander(samples2, n)
    # rescale to have orthonormal polynomials
    V = V * np.tile(np.sqrt(2 * np.arange(0, n + 1) + 1), (len(samples), 1))
    # compute the estimator
    estim = np.sum(f(samples) - c @ V.T) / len(samples) + c[0]
    return estim


