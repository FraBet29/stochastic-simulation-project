import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt

#######################################################
# Question 2: estimating integral I with CMC and MCLS #
#######################################################

def f(x):
    """
    Evaluate the function to be integrated at point x.
    args : points x where f will be evaluated
    return : value of the function
    """
    return 1 / (25 * x ** 2 + 1)


def crude_MC(samples, f, alpha=0.05):
    """
    Provide a crude Monte Carlo estimator for the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
           alpha, significance level
    return : estim, crude MC estimator based on these samples
             CI, confidence interval for the crude MC estimator based on these samples
    """
    estim = np.sum(f(samples)) / len(samples)
    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    CI = estim + np.array([-1, 1]) * np.sqrt(np.var(f(samples)) / len(samples)) * quantile
    return estim, CI


def MCLS(samples, f, n, alpha=0.05):
    """
    Compute Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
           n, maximal exponential of the Legendre polynomials
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
             CI, confidence interval
    """
    # compute the Vandermonde matrix
    x = np.random.uniform(0, 1, len(samples))
    x2 = 2 * x - 1 # shift the Legendre polynomials from [-1, 1] to [0, 1]
    V = np.polynomial.legendre.legvander(x2, n)
    
    # compute the condition number
    cond = np.linalg.cond(V)
    
    # solve the least squares problem
    c = np.linalg.lstsq(V, f(x), rcond=None)[0]
    
    # compute the estimator
    samples2 = 2 * samples - 1
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]

    # compute the confidence interval
    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    std_ls = np.sqrt(np.sum((f(samples) - np.polynomial.legendre.legval(samples2, c)) ** 2) / len(samples))
    CI = estim + np.array([-1, 1]) * quantile * std_ls / np.sqrt(len(samples))
    
    return estim, cond, CI


def MCLS_prime(samples, f, n, alpha=0.05):
    """
    Compute an alternative Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
           n, maximal exponential of the Legendre polynomials
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
             CI, confidence interval
    """
    # compute the Vandermonde matrix
    samples2 = 2 * samples - 1
    V = np.polynomial.legendre.legvander(samples2, n)

    # compute the condition number
    cond = np.linalg.cond(V)
    
    # solve the least squares problem
    c = np.linalg.lstsq(V, f(samples), rcond=None)[0]
    
    # compute the estimator
    estim = c[0]

    # compute the confidence interval
    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    std_ls = np.sqrt(np.sum((f(samples) - np.polynomial.legendre.legval(samples2, c)) ** 2) / len(samples))
    CI = estim + np.array([-1, 1]) * quantile * std_ls / np.sqrt(len(samples))
    
    return estim, cond, CI


def MCLS_old(samples, f, n):
    """
    Compute Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
           n, maximal exponential of the legendre polynomials
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    # solve the least squares problem
    x = np.random.uniform(0, 1, len(samples))
    c, diagnostics = np.polynomial.legendre.Legendre.fit(x, f(x), n, domain=[0, 1], full=True) # diagnostics = [resid, rank, sv, rcond]
    c = c.coef

    # compute the condition number
    cond = np.max(diagnostics[2]) / np.min(diagnostics[2])
    
    # compute the estimator
    samples2 = 2 * samples - 1
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]
    
    return estim, cond


def MCLS_prime_old(samples, f, n):
    """
    Compute an alternative Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    # solve the least squares problem
    c, diagnostics = np.polynomial.legendre.Legendre.fit(samples, f(samples), n, domain=[0, 1], full=True) # diagnostics = [resid, rank, sv, rcond]
    c = c.coef

    # compute the condition number
    cond = np.max(diagnostics[2]) / np.min(diagnostics[2])
    
    # compute the estimator
    estim = c[0]
    
    return estim, cond