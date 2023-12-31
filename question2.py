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
    log_nb_samples = np.log10(nb_samples)
    log_errors = np.log10(absolute_errors)

    plt.figure(figsize=(8, 6))
    plt.scatter(log_nb_samples, log_errors, label='Absolute error')

    # fit a linear regression line
    regression = LinearRegression()
    regression.fit(log_nb_samples.reshape(-1, 1), log_errors)
    pred = regression.predict(log_nb_samples.reshape(-1, 1))
    plt.plot(log_nb_samples, pred, color='red', label='Linear regression')

    # get the coefficients of the linear regression
    slope = regression.coef_[0]
    intercept = regression.intercept_
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    plt.text(1.5, -3, equation, fontsize=10, color='red')

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def loglog_average_error_graph(nb_samples, MC_estims, ref_value):
    """
    Plot the graph of the error in absolute value of the MC_estims for the 'true' value ref_value.
    args : nb_samples, vector containing the number of samples used to compute each estimator
           MC_estims, list of lists of the estimators
           ref_value, the reference 'true' value
    return : /
    """
    # compute absolute values of the error between estimators and ref_value
    absolute_errors = np.abs(MC_estims - ref_value)
    
    averages = 1/len(MC_estims[0])*np.sum(absolute_errors, axis=1)

    # create log-log plot
    log_nb_samples = np.log10(nb_samples)
    log_errors = np.log10(averages)


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
    plt.text(3, -1.5, equation, fontsize=10, color='red')

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def MCLS(samples, f, n):
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
    # print("n = ", n)
    # print the condition number of the Vandermonde matrix
    # print("Condition number of the Vandermonde matrix : ", np.linalg.cond(V))
    # compute the coefficients of the estimator using a QR decomposition
    Q, R = np.linalg.qr(V.T @ V)
    y = Q.T @ (V.T @ f(x))
    c = np.linalg.solve(R, y)
    # compute the estimator
    estim = np.sum(f(samples) - c @ V.T) / len(samples) + c[0]
    return estim

def MCLS_prime(samples, f):
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
             cond, condition number of the Vandermonde matrix
    """
	# generate M samples
    x = np.random.uniform(0, 1, M)
    y = f(x)
	# solve the least squares problem
    c, diagnostics = np.polynomial.legendre.Legendre.fit(x, y, n, domain=[0, 1], full=True) # diagnostics = [resid, rank, sv, rcond]
    cond = np.max(diagnostics[2]) / np.min(diagnostics[2])
    return c.coef, cond

def MCLS_new(samples, f, n):
    """
    Compute Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
           n, maximal exponential of the legendre polynomials
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    c, cond = least_squares(n, len(samples))
    samples2 = samples*2 - 1
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]
    return estim, cond

def MCLS_prime_new(samples, f):
    """
    Compute an alternative Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    c, cond = least_squares(0, len(samples))
    estim = c
    return estim, cond

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
        # discard values equal to -1 (not valid)
        valid_idx = np.where(MC_estims != -1)[0]
        MC_estims = MC_estims[valid_idx]
        
        # compute absolute values of the error between estimators and ref_value
        absolute_errors = np.abs(MC_estims - ref_value)

        # discard values for which the error is of the order of epsilon machine (cannot compute log)
        eps_idx = np.where(absolute_errors > 1e-16)[0]
        absolute_errors = absolute_errors[eps_idx]

        # create log-log plot
        log_nb_samples = np.log10(nb_samples[valid_idx][eps_idx])
        log_errors = np.log10(absolute_errors)

        plt.scatter(log_nb_samples, log_errors, label=f'Series {i+1}', color=colors(i), s=5)       

        # fit a linear regression line
        regression = LinearRegression()
        regression.fit(log_nb_samples.reshape(-1, 1), log_errors)
        pred = regression.predict(log_nb_samples.reshape(-1, 1))
        plt.plot(log_nb_samples, pred, color=colors(i))

        # get the coefficients of the linear regression
        slope = regression.coef_[0]
        intercept = regression.intercept_
        equation = f'y = {slope:.2f}x + {intercept:.2f}'
        plt.text(max(log_nb_samples)+1, -1 - i * 0.3, equation, fontsize=10, color=colors(i))

    plt.plot(np.log10(nb_samples), np.log10(1 / np.sqrt(nb_samples)), '--', label='$1 / \sqrt{M}$')

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def multiple_cond_loglog_graph(nb_samples, cond_list):
    """
    Plot the graph of the condition number of the Vandermonde matrix.
    args : nb_samples, number of samples used to compute the least squares fit
           cond_list, list of the condition numbers
    return : /
    """    

    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', len(cond_list))  # Get distinct colors

    for i, cond in enumerate(cond_list):
        # discard values equal to -1 (not valid)
        valid_idx = np.where(cond != -1)[0]
        cond_filtered = cond[valid_idx]
        nb_samples_filtered = nb_samples[valid_idx]
        
        # create log-log plot
        log_nb_samples = np.log10(nb_samples_filtered)
        log_cond = np.log10(cond_filtered - 1)
    
        plt.plot(log_nb_samples, log_cond, '--*', label=f'Series {i+1}', color=colors(i))

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Condition number - 1)')
    plt.title('Log-log plot of the condition number \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return