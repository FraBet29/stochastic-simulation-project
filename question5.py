import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import solve_ivp

#################################################################
# Question 5: application to the Fitzhugh-Nagumo system of ODES #
#################################################################

def solve_FHN(epsilon, I, v0, w0, t0, T, Nt, a, b):
    """
    Solve the Fitzhugh-Nagumo model system of ODEs with the forward Euler scheme.
    args : epsilon, parameter of the model
           I, parameter of the model
           v0, initial value of v
           w0, initial value of w
           t0, initial time
           T, final time
           Nt, number of time steps
           a, parameter of the model
           b, parameter of the model
    """
    # time step
    dt = (T - t0) / Nt
    # time vector
    t = np.linspace(t0, T, Nt + 1)
    # initial values
    v = np.zeros(Nt + 1)
    w = np.zeros(Nt + 1)
    v[0] = v0
    w[0] = w0
    # forward Euler scheme
    for i in range(Nt):
        v[i+1] = v[i] + dt * (v[i] - v[i] ** 3 / 3 - w[i] + I)
        w[i+1] = w[i] + dt * epsilon * (v[i] + a - b * w[i])
    return v, w, t


def solve_FHN_scipy(epsilon, I, v0, w0, t0, T, Nt, a, b):
    """
    Solve the Fitzhugh-Nagumo model system of ODEs with scipy.integrate.solve_ivp.
    args : epsilon, parameter of the model
           I, parameter of the model
           v0, initial value of v
           w0, initial value of w
           t0, initial time
           T, final time
           Nt, number of time steps
           a, parameter of the model
           b, parameter of the model
    """
    # time vector
    t = np.linspace(t0, T, Nt + 1)
    # initial values
    y0 = [v0, w0]
    # function to solve
    def fitzhugh_nagumo(t, y):
        v, w = y
        dvdt = v - v ** 3 / 3 - w + I
        dwdt = epsilon * (v + a - b * w)
        return [dvdt, dwdt]
    # solve the system of ODEs
    sol = solve_ivp(fitzhugh_nagumo, [t0, T], y0, t_eval=t)
    return sol.y[0], sol.y[1], sol.t


def calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a, b):
    """
    Calculate the quantity of interest Q for the Fitzhugh-Nagumo model system of ODEs.
    args : epsilon, parameter of the model
           I, parameter of the model
           v0, initial value of v
           w0, initial value of w
           t0, initial time
           T, final time
           Nt, number of time steps
           a, parameter of the model
           b, parameter of the model
    """
    v, w, t = solve_FHN(epsilon, I, v0, w0, t0, T, Nt, a, b)
    dt = (T - t0) / Nt
    Q = (np.sum(v ** 2) - v[0] ** 2 - v[len(v)-1] ** 2) * dt
    return Q


def shifted(x, a, b):
    """
    Shift the evaluation of the Legendre polynomials from [-1, 1] to [a, b].
    args : x, evaluation points
           a, 1st extreme of the interval
           b, 2nd extreme of the interval
    """
    return 2 * (x - a) / (b - a) - 1


def crude_MC_2D(a_samples, b_samples, epsilon, I, v0, w0, t0, T, Nt, alpha):
    """
    Provide a crude Monte Carlo estimator for the integral of Q.
    args : a_samples, samples x drawn from uniform distribution U([0.6, 0.8])
           b_samples, samples y drawn from uniform distribution U([0.7, 0.9])
           epsilon, I, v0, w0, t0, T, Nt, parameters of the Fitzhugh-Nagumo model
    return : estim, crude MC estimator based on these samples
             CI, confidence interval for the crude MC estimator based on these samples
    """
    M = len(a_samples)
    Q = np.zeros(M)
    for i in range(M):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a_samples[i], b_samples[i])
    estim = (0.2 ** 2) * np.sum(Q) / M
    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    CI = estim + np.array([-1, 1]) * quantile * np.std(Q) / M
    return estim, CI


def MCLS_2D(a_samples, b_samples, n, epsilon, I, v0, w0, t0, T, Nt, alpha):
    """
    Compute Monte Carlo Least Square estimator of for the integral of Q.
    args : a_samples, samples x drawn from uniform distribution U([0.6, 0.8])
           b_samples, samples y drawn from uniform distribution U([0.7, 0.9])
           n, maximal exponential of the Legendre polynomials
           epsilon, I, v0, w0, t0, T, Nt, parameters of the Fitzhugh-Nagumo model
           alpha, significance level of the confidence interval
    return : estim, crude MC estimator based on these samples
             cond, condition number of the Vandermonde matrix
             CI, confidence interval for the crude MC estimator based on these samples
    """
    M = len(a_samples)
    
    # compute the 2D Vandermonde matrix
    x = np.random.uniform(0.6, 0.8, M)
    y = np.random.uniform(0.7, 0.9, M)
    V = np.polynomial.legendre.legvander2d(shifted(x, 0.6, 0.8), shifted(y, 0.7, 0.9), (n, n))

    # compute the condition number
    cond = np.linalg.cond(V)
    
    # compute Q for the least squares problem
    Q = np.zeros(M)
    for i in range(M):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, x[i], y[i])
    
    # solve the least squares problem
    c = np.linalg.lstsq(V, Q, rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    
    # compute Q for the estimator
    for i in range(M):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a_samples[i], b_samples[i])
    
    # compute the estimator
    estim = (0.2 ** 2) * (np.sum(Q - np.polynomial.legendre.legval2d(shifted(a_samples, 0.6, 0.8), shifted(b_samples, 0.7, 0.9), c)) / M + c[0, 0])
    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    err = np.sqrt(np.sum((Q - np.polynomial.legendre.legval2d(shifted(a_samples, 0.6, 0.8),
                                                              shifted(b_samples, 0.7, 0.9), c)) ** 2) / M) / np.sqrt(M)
    CI = estim + np.array([-1, 1]) * quantile * err
    return estim, cond, CI


def MCLS_prime_2D(a_samples, b_samples, n, epsilon, I, v0, w0, t0, T, Nt):
    """
    Compute an alternative Monte Carlo Least Square estimator of for the integral of Q.
    args : a_samples, samples x drawn from uniform distribution U([0.6, 0.8])
           b_samples, samples y drawn from uniform distribution U([0.7, 0.9])
           n, maximal exponential of the Legendre polynomials
           epsilon, I, v0, w0, t0, T, Nt, parameters of the Fitzhugh-Nagumo model
    return : estim, crude MC estimator based on these samples
             cond, condition number of the Vandermonde matrix
    """
    M = len(a_samples)

    # compute the 2D Vandermonde matrix
    V = np.polynomial.legendre.legvander2d(shifted(a_samples, 0.6, 0.8), shifted(b_samples, 0.7, 0.9), (n, n))

    # compute the condition number
    cond = np.linalg.cond(V)

    # compute Q
    Q = np.zeros(M)
    for i in range(M):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a_samples[i], b_samples[i])

    # solve the least squares problem
    c = np.linalg.lstsq(V, Q, rcond=None)[0]
    c = c.reshape((n + 1, n + 1))

    # compute the estimator
    estim = (0.2 ** 2) * c[0, 0]

    return estim, cond


def calculate_error(a_samples, b_samples, n, epsilon, I, v0, w0, t0, T, Nt):
    """
    Compute an estimator for the absolute error of MCLS.
    args : a_samples, samples x drawn from uniform distribution U([0.6, 0.8])
           b_samples, samples y drawn from uniform distribution U([0.7, 0.9])
           n, maximal exponential of the Legendre polynomials
           epsilon, I, v0, w0, t0, T, Nt, parameters of the Fitzhugh-Nagumo model
    return : err, error estimator
    """
    M = len(a_samples)
    
    # compute the 2D Vandermonde matrix
    x = np.random.uniform(0.6, 0.8, M)
    y = np.random.uniform(0.7, 0.9, M)
    V = np.polynomial.legendre.legvander2d(shifted(x, 0.6, 0.8), shifted(y, 0.7, 0.9), (n, n))
    
    # compute Q for the least squares problem
    Q = np.zeros(M)
    for i in range(M):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, x[i], y[i])
    
    # solve the least squares problem
    c = np.linalg.lstsq(V, Q, rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    
    # compute Q for the error estimator
    for i in range(M):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a_samples[i], b_samples[i])

    # compute the error estimator
    err = np.sqrt(np.sum((Q - np.polynomial.legendre.legval2d(shifted(a_samples, 0.6, 0.8), shifted(b_samples, 0.7, 0.9), c)) ** 2)) / M
    return err


def CI_MCLS_2D(a_samples, b_samples, n, epsilon, I, v0, w0, t0, T, Nt, estim, alpha):
    """
    Compute the confidence interval of the MCLS estimator.
    args : a_samples, samples x drawn from uniform distribution U([0.6, 0.8])
            b_samples, samples y drawn from uniform distribution U([0.7, 0.9])
            n, maximal exponential of the Legendre polynomials
            epsilon, I, v0, w0, t0, T, Nt, parameters of the Fitzhugh-Nagumo model
    return : CI, confidence interval
    """
    M = len(a_samples)
    
    # compute the 2D Vandermonde matrix
    x = np.random.uniform(0.6, 0.8, M)
    y = np.random.uniform(0.7, 0.9, M)
    V = np.polynomial.legendre.legvander2d(shifted(x, 0.6, 0.8), shifted(y, 0.7, 0.9), (n, n))
    
    # compute Q for the least squares problem
    Q = np.zeros(M)
    for i in range(M):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, x[i], y[i])
    
    # solve the least squares problem
    c = np.linalg.lstsq(V, Q, rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    
    # compute Q for the estimator
    for i in range(M):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a_samples[i], b_samples[i])
    
    # compute the confidence interval
    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    std_ls = np.sqrt(np.sum((Q - np.polynomial.legendre.legval2d(shifted(a_samples, 0.6, 0.8), shifted(b_samples, 0.7, 0.9), c)) ** 2) / M)
    CI = estim + np.array([-1, 1]) * quantile * std_ls / np.sqrt(M)
    
    return CI