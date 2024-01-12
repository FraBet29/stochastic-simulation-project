import numpy as np 
import math
from scipy.stats import norm
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

def crude_MC(samples, f, alpha):
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

    plt.plot(np.log10(nb_samples), np.log10(1 / np.sqrt(nb_samples)), '--', label='$1 / \sqrt{M}$', color='grey')

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def plot_CI(nb_samples, MC_estims, CI, exact_val, alpha):
    """
    Plot the confidence interval of the crude MC estimator.
    args : nb_samples, vector containing the number of samples used to compute each estimator
           MC_estims, the estimators
           CI, confidence interval for the crude MC estimator based on these samples
           exact_val, true value
    return : /
    """
    plt.figure(figsize=(8, 6))
    plt.plot(nb_samples, MC_estims, label='Estimator', color='red')
    plt.plot(nb_samples, CI[:, 0], linestyle='--', label='Estimated $ \\alpha $ = '+ str(alpha)+' confidence interval', color='orange')
    plt.plot(nb_samples, CI[:, 1], linestyle='--', color='orange')
    plt.axhline(y=exact_val, color='blue', linestyle='-.', label='True value')

    plt.xscale('log')  # Set x-axis to log scale
    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Value')
    plt.title('Confidence interval for the crude MC estimator')
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
    plt.text(2.8, -1.2, equation, fontsize=10, color='red')

    plt.plot(np.log10(nb_samples), np.log10(1 / np.sqrt(nb_samples)), '--', label='$1 / \sqrt{M}$', color='grey')

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
           n, maximal exponential of the Legendre polynomials
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    # compute the Vandermonde matrix
    x = np.random.uniform(0, 1, len(samples))
    x2 = x*2 - 1
    V = np.polynomial.legendre.legvander(x2, n)
    cond = np.linalg.cond(V)
    """
    # compute the coefficients of the estimator using a QR decomposition
    Q, R = np.linalg.qr(V.T @ V)
    y = Q.T @ (V.T @ f(x))
    c = np.linalg.solve(R, y)
    """
    c = np.linalg.lstsq(V, f(x), rcond=None)[0]
    # compute the estimator
    samples2 = samples*2 - 1
    estim = np.sum(f(samples) - np.polynomial.legendre.legval(samples2, c)) / len(samples) + c[0]

    return estim, cond

def CI_MCLS(samples, f, n, estim, alpha):
    """
    Compute the confidence interval of the MCLS estimator.
    args : samples
           f, the function to integrate
           estim, MCLS estimator based on these samples
           alpha, significance level of the confidence interval
    return : CI, confidence interval for the MCLS estimator based on these samples
    """
    samples2 = samples * 2 - 1
    V = np.polynomial.legendre.legvander(samples2, n)
    c = np.linalg.lstsq(V, f(samples), rcond=None)[0]
    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    CI = estim + np.array([-1, 1]) * quantile * np.sqrt(np.sum((f(samples) - np.polynomial.legendre.legval(samples2, c))**2) / len(samples))
    return CI
def MCLS_prime(samples, f, n):
    """
    Compute an alternative Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
           n, maximal exponential of the Legendre polynomials
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    # compute the Vandermonde matrix
    samples2 = samples*2 - 1
    V = np.polynomial.legendre.legvander(samples2, n)
    cond = np.linalg.cond(V)
    # compute the coefficients of the estimator
    c = np.linalg.lstsq(V, f(samples), rcond=None)[0]
    # compute the estimator
    estim = c[0]
    return estim, cond

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

def MCLS_prime_new(samples, f, n):
    """
    Compute an alternative Monte Carlo Least Square estimator of the integral of f between 0 and 1.
    args : samples, samples x drawn from uniform distribution U([0, 1])
           f, the function to integrate
    return : estim, MCLS estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    c, cond = least_squares(n, len(samples))
    estim = c[0]
    return estim, cond

def multiple_loglog_graph(nb_samples, MC_estims_list, ref_value, legend_series):
    """
    Plot the graph of the error in absolute value of the MC_estims for the 'true' value ref_value.
    args : nb_samples, number of samples used to compute each estimator
           MC_estims_list, list of the estimators
           ref_value, the reference 'true' value
    return : /
    """    
    
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('Set2', len(MC_estims_list))  # Get distinct colors

    for i, MC_estims in enumerate(MC_estims_list):
        # discard values equal to -1 (not valid)
        valid_idx = np.where(MC_estims != -1)[0]
        MC_estims = MC_estims[valid_idx]
        
        # compute absolute values of the error between estimators and ref_value
        absolute_errors = np.abs(MC_estims - ref_value)

        # discard values for which the error is of the order of epsilon machine (cannot compute log)
        eps_idx = np.where(absolute_errors < 1e-16)[0]
        absolute_errors[eps_idx] = 1e-16

        # create log-log plot
        log_nb_samples = np.log10(nb_samples[valid_idx])
        log_errors = np.log10(absolute_errors)

        legend_M = legend_series[i]
        if legend_M == -2 or legend_M == -1 or legend_M == -3:
            if legend_M == -2:
                legend_M = 'n = M/2'
                color = 'blue'  # Blue color for legend_series -2
            elif legend_M == -1:
                legend_M = 'n = $\sqrt{M}$'
                color = 'red'  # Red color for legend_series -1
            elif legend_M == -3:
                legend_M = '$ M = n \log{n}$'
                color = 'green'

            plt.plot(log_nb_samples, log_errors, label=f'Serie ' + str(legend_M), color=color, linestyle='--', marker='o', markersize=3, linewidth=1)
        else:
            # Colors for other legend_series values
            color = colors(i)

            plt.scatter(log_nb_samples, log_errors, label=f'Serie n = ' + str(legend_M), color=color, s=5)

            if len(log_nb_samples[log_nb_samples > 2.5]) != 0:
                # Fit a linear regression line
                regression = LinearRegression()
                regression.fit(log_nb_samples[log_nb_samples > 2.5].reshape(-1, 1), log_errors[log_nb_samples > 2.5])
                pred = regression.predict(log_nb_samples[log_nb_samples > 2.5].reshape(-1, 1))
                plt.plot(log_nb_samples[log_nb_samples > 2.5], pred, color=color)

                slope = regression.coef_[0]
                intercept = regression.intercept_
                if intercept>=0:
                    equation = f'y = {slope:.2f}x + {intercept:.2f}'
                else:
                    equation = f'y = {slope:.2f}x - {np.abs(intercept):.2f}'
                plt.text(max(log_nb_samples) + 0.5, -1 - i * 0.5, equation, fontsize=10, color=color)

    plt.plot(np.log10(nb_samples), np.log10(1 / np.sqrt(nb_samples)), '--', label='$1 / \sqrt{M}$', color='grey')

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Absolute error)')
    plt.title('Log-log plot of absolute error \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

def multiple_cond_loglog_graph(nb_samples, cond_list, legend_series):
    """
    Plot the graph of the condition number of the Vandermonde matrix.
    args : nb_samples, number of samples used to compute the least squares fit
           cond_list, list of the condition numbers
    return : /
    """    

    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('Set2', len(cond_list))  # Get distinct colors

    for i, cond in enumerate(cond_list):
        color=colors(i)
        legend_M = "n =" + str(legend_series[i])
        if legend_series[i] == -1:
            color = 'red'
            legend_M = '$n = \sqrt{M}$'
        elif legend_series[i] == -2:
            color = 'blue'
            legend_M = 'n = M/2'
        elif legend_series[i] == -3:
            legend_M = '$ M = n \log{n}$'
            color = 'green'


        # discard values equal to -1 (not valid)
        valid_idx = np.where(cond != -1)[0]
        cond_filtered = cond[valid_idx]
        nb_samples_filtered = nb_samples[valid_idx]
        
        # create log-log plot
        log_nb_samples = np.log10(nb_samples_filtered)
        log_cond = np.log10(cond_filtered - 1)
    
        plt.plot(log_nb_samples, log_cond, label=f'Serie '+ legend_M, color=color, linestyle='--', marker='o', markersize=3, linewidth=1)

    plt.xlabel('Log(Number of samples M)')
    plt.ylabel('Log(Condition number - 1)')
    plt.title('Log-log plot of the condition number \n as a function of the number of samples M')
    plt.legend()
    plt.grid(True)
    plt.show()

    return