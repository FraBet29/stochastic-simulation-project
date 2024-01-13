import numpy as np 
import math
import matplotlib.pyplot as plt

########################################
# Question 4 : MCLS estimators with IS #
########################################

def phi_squared(x, n):
    c_vector = np.zeros(n+1)
    c_vector[n] = 1
    return np.square(np.polynomial.legendre.legval(2*x-1, c_vector))*(2*n+1)


def g(x):
    return 1 / (np.pi * np.sqrt(x * (1 - x)))


def G(x):
    return 2 * np.arcsin(np.sqrt(x)) / np.pi


def G_inv(x):
    return np.sin(np.pi * x / 2) ** 2


def h(x, n):
    sum_of_squares = 0
    c_vector = np.eye(n+1, n+1)
    for j in range(n + 1):
        sum_of_squares += np.square(np.polynomial.legendre.legval(2*x-1, c_vector[j,:]))*(2*j+1)
    return sum_of_squares / (n + 1)


def sample_from_g_new(M):
    """
    Sample from pdf g by Inverse transform method.
    args: M, number of samples to draw
    return: g_samples, array of the M samples from pdf g
    """
    unif_samples = np.random.uniform(0, 1, M)
    return G_inv(unif_samples)


def sample_from_h_new(M, ite_max, n, print_rate=False):
    """
    Sample from pdf h by composition method + Acceptance-Rejection method.
    args: M, number of samples to draw
          ite_max, maximal number of iterations allowed (in case rejectance rate would be too high)
          n, Legendre polynomials up to degree n will be used
    return: h_samples, array of the M samples from pdf g
    """
    h_samples = np.zeros(M)
    C = 4 * math.e
    
    tot_ite = 0
    
    for m in range(M):
        
		# Sample an index from 0 to n
        j = np.random.randint(0, n + 1)
        
		# Apply the AR method to sample from phi_j^2
        ite = 0
        while ite < ite_max:
            Y_sample = sample_from_g_new(1)
            U_sample = np.random.uniform(0, 1, 1)
            if U_sample <= phi_squared(Y_sample, j) / (C * g(Y_sample)):
                h_samples[m] = Y_sample
                break
            ite += 1
        
        if ite >= ite_max:
            raise Exception("Acceptance rate too low")
        tot_ite += ite

    if print_rate:
        print('Acceptance rate: ', M / tot_ite)
    
    return h_samples


def weighted_least_squares_new(f, n, M, w):
    """
    Compute n + 1 coefficients of weighted least squares based on M samples drawn from h(x) = 1 / w(x)
    args : f, function to approximate
           n, st n+1 is the number of coefficients
           M, number of samples to use
           w, weights
    return : c, vector containing the n+1 optimal coefficients
             cond, condition number of the Vandermonde matrix
    """
	# generate M samples from h
    x = sample_from_h_new(M, 1000, n)
    y = f(x)
	# solve the least squares problem
    c, diagnostics = np.polynomial.legendre.Legendre.fit(x, y, n, domain=[0, 1], full=True, w=np.sqrt(w)) # diagnostics = [resid, rank, sv, rcond]
    cond = np.max(diagnostics[2]) / np.min(diagnostics[2])
    return c.coef, cond


def IS_MCLS_new(samples, f, n):
    """
    Compute Monte Carlo Least Square estimator of the integral of f between 0 and 1 with importance sampling.
    args : samples, samples x drawn from h(x) = 1 / w(x)
           f, the function to integrate
           n, maximal exponential of the legendre polynomials
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    weights = 1 / h(samples, n)
    c, cond = weighted_least_squares_new(f, n, len(samples), weights)
    samples2 = samples*2 - 1
    estim = np.sum(np.dot(f(samples) - np.polynomial.legendre.legval(samples2, c), weights)) / np.sum(weights) + c[0]
    return estim, cond


def IS_MCLS_prime_new(samples, f, n):
    """
    Compute an alternative Monte Carlo Least Square estimator of the integral of f between 0 and 1 with importance sampling.
    args : samples, samples x drawn from h(x) = 1 / w(x)
           f, the function to integrate
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    weights = 1 / h(samples, n)
    c, cond = weighted_least_squares_new(f, n, len(samples), weights)
    estim = c[0]
    return estim, cond