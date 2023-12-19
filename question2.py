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

