import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def f(x):
    return 1/(25*x**2+1)

def LS_Vandermonde(n, x):
    x2 = 2 * x - 1
    # compute the Vandermonde matrix
    V = np.polynomial.legendre.legvander(x2, n)
    # rescale to have orthonormal polynomials
    V = V * np.tile(np.sqrt(2 * np.arange(0, n + 1) + 1), (len(x), 1))
    # solve the least squares problem
    c = np.linalg.lstsq(V, f(x), rcond=None)[0]
    return c

def LS_fit(n, x):
    c = np.polynomial.legendre.Legendre.fit(x, f(x), n, domain=[0, 1])
    c = c.coef
    return c / np.sqrt(2 * np.arange(0, n+1) + 1)

def MCLS_V_V(samples, samples_fit, f, n):
    # compute the Vandermonde matrix
    samples2 = samples*2 - 1
    c = LS_Vandermonde(n, samples_fit)
    V = np.polynomial.legendre.legvander(samples2, n)
    # rescale to have orthonormal polynomials
    V = V * np.tile(np.sqrt(2 * np.arange(0, n + 1) + 1), (len(samples), 1))
    # compute the estimator
    estim = np.sum(f(samples) - c @ V.T) / len(samples) + c[0]
    return estim

def MCLS_V_legval(samples, samples_fit, f, n):
    samples2 = samples*2 - 1
    c = LS_Vandermonde(n, samples_fit)
    # rescale the coefficients
    c = c * np.sqrt(2 * np.arange(0, n + 1) + 1)
    # compute the estimator
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]
    return estim

def MCLS_fit_legval(samples, samples_fit, f, n):
    samples2 = samples*2 - 1
    c = LS_fit(n, samples_fit)
    # rescale the coefficients
    c = c * np.sqrt(2 * np.arange(0, n + 1) + 1)
    # compute the estimator
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]
    return estim

def MCLS_fit_V(samples, samples_fit, f, n):
    samples2 = samples*2 - 1
    c = LS_fit(n, samples_fit)
    V = np.polynomial.legendre.legvander(samples2, n)
    # rescale to have orthonormal polynomials
    V = V * np.tile(np.sqrt(2 * np.arange(0, n + 1) + 1), (len(samples), 1))
    # compute the estimator
    estim = np.sum(f(samples) - c @ V.T) / len(samples) + c[0]
    return estim

def MCLS_nonorm(samples, samples_fit, f, n):
    # compute the Vandermonde matrix
    samples_fit2 = samples_fit * 2 - 1
    V = np.polynomial.legendre.legvander(samples_fit2, n)
    c = np.linalg.lstsq(V, f(samples_fit), rcond=None)[0]
    # compute the estimator
    samples2 = samples * 2 - 1
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]
    return estim

def MCLS_new_nonorm(samples, samples_fit, f, n):
    # solve the least squares problem
    c = np.polynomial.legendre.Legendre.fit(samples_fit, f(samples_fit), n, domain=[0, 1])
    c = c.coef
    samples2 = samples*2 - 1
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]
    return estim