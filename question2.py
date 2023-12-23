import numpy as np 
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

######################################
# Question 2 : estimating integral I #
######################################

def f(x):
    """
    Evaluate the function to be integrated at point x.
    args : points x where f will be evaluated
    return : value of the function defined in question 2
    """
    return 1/(25*x**2+1)

def crude_MC(samples, f):
    """
    Provide a crude Monte Carlo estimator for the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
    return : estim, crude MC estimator based on these samples
    """
    estim = np.sum(f(samples)) / len(samples)
    return estim


def loglog_graph(nb_samples, MC_estims, ref_value):
    """
    Plot the graph of the error in absolute value of the MC_estims for the 'true' value ref_value.
    args : nb_samples, vector containing the number of samples used to compute each estimator
           MC_estims, the estimators
           ref_value, the reference 'true' value
    return : /
    """
    # compute absolute values of the error between estimators and ref_value
    absolute_errors = np.abs(MC_estims - ref_value)

    # create log-log plot
    log_nb_samples = np.log(nb_samples)
    log_errors = np.log(absolute_errors)

    plt.figure(figsize=(8, 6))
    plt.scatter(log_nb_samples, log_errors, label='Absolute error')

    # Fit a linear regression line
    regression = LinearRegression()
    regression.fit(log_nb_samples.reshape(-1, 1), log_errors)
    pred = regression.predict(log_nb_samples.reshape(-1, 1))
    plt.plot(log_nb_samples, pred, color='red', label='Linear regression')

    # Get the coefficients of the linear regression
    slope = regression.coef_[0]
    intercept = regression.intercept_
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(0.5, -8, equation, fontsize=10, color='red')

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def MCLS (samples, f, n):
    """
    Compute Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
           n, maximal exponential of the legendre polynomials
    return : estim, MCLS estimator based on these samples
    """
    # compute the Vandermonde matrix
    x = np.random.uniform(0, 1, len(samples))
    x2 = x*2 - 1
    V = np.polynomial.legendre.legvander(x2, n)
    """
    # compute the Vandermonde matrix by hand
    coef = [1] * (n + 1)
    L = np.polynomial.legendre.Legendre(coef, domain=[0, 1])
    x = np.repeat(samples, n, axis=0).reshape(-1, n)
    V = L(x)
    """
    print("n = ", n)
    # print the condition number of the Vandermonde matrix
    print("Condition number of the Vandermonde matrix : ", np.linalg.cond(V))
    # compute the coefficients of the estimator using a QR decomposition
    Q, R = np.linalg.qr(V.T @ V)
    y = Q.T @ (V.T @ f(x))
    c = np.linalg.solve(R, y)
    # compute the estimator
    estim = np.sum(f(samples) - c @ V.T) / len(samples) + c[0]
    return estim

def MCLS_prime (samples, f):
    """
    Compute an alternative Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
    return : estim, MCLS estimator based on these samples
    """
    x = np.random.uniform(0, 1, len(samples))
    # compute the Vandermonde matrix
    V = np.polynomial.legendre.legvander(x, 0)
    # compute the coefficients of the estimator
    c0 = np.linalg.inv(V.T @ V) @ V.T @ f(x)
    # compute the estimator
    estim = c0
    return estim


def least_squares(n, M): 
    """
    Compute n + 1 coefficients of least squares based on M samples
    args : n, st n+1 is the number of coefficients
           M, number of samples to use
    return : c, vector containing the n+1 optimal coefficients
    """
	# generate M samples
    x = np.random.uniform(0, 1, M)
    y = f(x)
	# solve the least squares problem
    c = np.polynomial.legendre.Legendre.fit(x, y, n, domain=[0, 1]).coef
    return c

def MCLS_new(samples, f, n):
    c = least_squares(n, len(samples))
    samples2 = samples*2 - 1
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]
    return estim

def MCLS_prime_new(samples, f):
    c = least_squares(0, len(samples))
    estim = c
    return estim

def multiple_loglog_graph(nb_samples, MC_estims_list, ref_value):
    """
    Plot the graph of the error in absolute value of the MC_estims for the 'true' value ref_value.
    args : nb_samples, number of samples used to compute each estimator
           MC_estims_list, list of the estimators
           ref_value, the reference 'true' value
    return : /
    """
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', len(MC_estims_list))  # Get distinct colors

    for i, MC_estims in enumerate(MC_estims_list):
        # compute absolute values of the error between estimators and ref_value
        absolute_errors = np.abs(MC_estims - ref_value)

        # create log-log plot
        log_nb_samples = np.log(nb_samples)
        log_errors = np.log(absolute_errors)

        plt.scatter(log_nb_samples, log_errors, label=f'Series {i+1}', color=colors(i))

        # Fit a linear regression line
        regression = LinearRegression()
        regression.fit(log_nb_samples.reshape(-1, 1), log_errors)
        pred = regression.predict(log_nb_samples.reshape(-1, 1))
        plt.plot(log_nb_samples, pred, color=colors(i))

        # Get the coefficients of the linear regression
        slope = regression.coef_[0]
        intercept = regression.intercept_
        equation = f'y = {slope:.2f}x + {intercept:.2f}'
        plt.text(0.5, -8 - i * 0.5, equation, fontsize=10, color=colors(i))

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return