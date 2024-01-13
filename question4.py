import numpy as np 
import math
import matplotlib.pyplot as plt
from question2 import f

#############################################
# Question 4: estimating integral I with IS #
#############################################

def pdf_h(x, n):
    """
    Evaluate pdf h = 1 / w at x.
    args: x, where to evaluate (float or array)
    return: value of h at x (float or array)
    """
    sum_of_squared = 0
    c_vector = np.zeros(n + 1)
    for j in range(n + 1):
        c_vector[j] = 1
        # 2j+1 is a normalizing factor on Legendre polynomials
        sum_of_squared += np.square(np.polynomial.legendre.legval(2 * x - 1, c_vector)) * (2 * j + 1)
        c_vector[j] = 0

    return sum_of_squared / (n + 1)


def pdf_phi_squared(x, n):
    """
    Evaluate pdf phi_squared (nodmalized Legendre polynomial of degree n) at x.
    args: x, where to evaluate (float or array)
    return: value of h at x (float or array)
    """
    c_vector = np.zeros(n + 1)
    c_vector[n] = 1
    # 2n+1 is a normalizing factor on Legendre polynomials
    phi_squared = np.square(np.polynomial.legendre.legval(2 * x - 1, c_vector)) * (2 * n + 1)
    return phi_squared


def pdf_g(x):
    """
    Evaluate pdf g (bound on h) at x.
    args: x, where to evluate
    return: value of pdf g at x
    """
    return 1 / (np.pi * np.sqrt(x * (1 - x)))


def cdf_G(x):
    """
    Evaluate G (cdf of g) at x. We verify easily that dG / dx = g.
    args: x, where to evaluate
    return: Gx, value of cdf G at x
    """
    Gx = 2 * np.arcsin(np.sqrt(x)) / np.pi
    return Gx


def cdf_Ginv(x) :
    """
    Evaluate Ginv (inverse cdf of g) at x.
    args: x, where to evaluate
    return: Ginvx, value of cdf Ginv at x
    """
    Ginvx = np.sin(np.pi * x / 2)**2
    return Ginvx


def sample_from_g(M):
    """
    Sample from pdf g (bound on pdf h = 1 / w). This is done by Inverse transform method.
    args: M, number of samples to draw
    return: g_samples, array of the M samples from pdf g
    """
    unif_samples = np.random.uniform(0, 1, M)
    return cdf_Ginv(unif_samples)


def sample_from_h_old(M, ite_max, n, print_rate=False):
    """
    Sample from pdf h = 1 / w. This is done by Acceptance-Rejection method.
    WARNING: this method is inefficient. Use the method sample_from_h instead.
    args: M, number of samples to draw
          ite_max, maximal number of iterations allowed (in case rejectance rate would be too high)
          n, Legendre polynomials up to degree n will be used
    return: h_samples, array of the M samples from pdf g
    """
    h_samples = np.zeros(M)
    C = 4 * math.e
    k = 0
    ite = 0

    while k < M and ite < ite_max:
        U_sample = np.random.uniform(0, 1, 1)
        Y_sample = sample_from_g(1)

        if U_sample <= pdf_h(Y_sample, n) / (C * pdf_g(Y_sample)):
            h_samples[k] = Y_sample
            k = k+1
        
        ite = ite+1

    if ite >= ite_max:
        raise Exception("Acceptance rate too low.")

    if print_rate:
        print('Acceptance rate : ', M / ite)
    
    return h_samples


def sample_from_h(M, ite_max, n, print_rate=False):
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
        
		# sample an index from 0 to n
        j = np.random.randint(0, n + 1)
        
		# apply the AR method to sample from pdf phi_squared
        ite = 0
        
        while ite < ite_max:
            Y_sample = sample_from_g(1)
            U_sample = np.random.uniform(0, 1, 1)
            
            if U_sample <= pdf_phi_squared(Y_sample, j) / (C * pdf_g(Y_sample)):
                h_samples[m] = Y_sample
                break
            
            ite += 1
        
        if ite >= ite_max:
            raise Exception("Acceptance rate too low.")

        tot_ite += ite

    if print_rate:
        print('Acceptance rate: ', M / tot_ite)
    
    return h_samples


def visualize_cdf_from_samples(samples):
    """
    Plot a graph of empirical cdf, based on the received samples.
    args: samples, array containing the sample values
    return: /
    """
    M = len(samples)

    # sort the samples
    sorted_samples = np.sort(samples)

    # calculate the empirical CDF
    cdf = np.arange(1, M + 1) / M

    # plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_samples, cdf, label='Empirical CDF')
    plt.title('Empirical CDF')
    plt.xlabel('Samples')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.show()


def IS_MCLS(samples, f, n):
    """
    Compute Monte Carlo Least Square estimator of the integral of f between 0 and 1 with importance sampling.
    args : samples, samples x drawn from h(x) = 1 / w(x)
           f, the function to integrate
           n, maximal exponential of the legendre polynomials
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    # solve the least squares problem
    x = sample_from_h(len(samples), 1000, n)
    weights = 1 / pdf_h(x, n)
    c, diagnostics = np.polynomial.legendre.Legendre.fit(x, f(x), n, domain=[0, 1], full=True, w=np.sqrt(weights)) # diagnostics = [resid, rank, sv, rcond]
    c = c.coef

    # compute the condition number
    cond = np.max(diagnostics[2]) / np.min(diagnostics[2])

    # compute the estimator
    weights = 1 / pdf_h(samples, n)
    samples2 = 2 * samples - 1
    estim = np.sum(np.dot(f(samples) - np.polynomial.legendre.legval(samples2, c), weights)) / np.sum(weights) + c[0]
    return estim, cond


def IS_MCLS_prime(samples, f, n):
    """
    Compute an alternative Monte Carlo Least Square estimator of the integral of f between 0 and 1 with importance sampling.
    args : samples, samples x drawn from h(x) = 1 / w(x)
           f, the function to integrate
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    # solve the least squares problem
    weights = 1 / pdf_h(samples, n)
    c, diagnostics = np.polynomial.legendre.Legendre.fit(samples, f(samples), n, domain=[0, 1], full=True, w=np.sqrt(weights)) # diagnostics = [resid, rank, sv, rcond]
    c = c.coef

    # compute the condition number
    cond = np.max(diagnostics[2]) / np.min(diagnostics[2])
    
    # compute the estimator
    estim = c[0]
    
    return estim, cond










def visualize_bound_g_on_h(n_value) :
    # Define the range of x values
    x_values = np.linspace(0.0001, 0.9999, 500)  # Adjust the range as needed

    # Calculate the corresponding y values for pdf_h and bound_pdf_g
    y_bound_pdf_g = bound_pdf_g(x_values)

    # Plotting
    plt.figure(figsize=(8, 6))

    for n in range(0, n_value+1) :
        y_pdf_h = pdf_h(x_values, n)
        plt.plot(x_values, y_pdf_h, label='pdf_h')
    plt.plot(x_values, 4*math.e*y_bound_pdf_g, label='bound_pdf_g')

    plt.title('Comparison of pdf_h and bound_pdf_g')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()