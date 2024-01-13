import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import solve_ivp

#######################################################################
# Question 5: application to the Fitzhugh-Nagumo model system of ODES #
#######################################################################

def solve_FHN(epsilon, I, v0, w0, t0, T, Nt, a, b):
    """
    Solve the Fitzhugh-Nagumo model system of ODEs with the forward Euler scheme.
    args: epsilon, parameter of the model
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
    args: epsilon, parameter of the model
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
    args: epsilon, parameter of the model
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
    args: x, evaluation points
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
        Compute the error
        args : a_samples, samples x drawn from uniform distribution U([0.6, 0.8])
               b_samples, samples y drawn from uniform distribution U([0.7, 0.9])
               n, maximal exponential of the Legendre polynomials
               epsilon, I, v0, w0, t0, T, Nt, parameters of the Fitzhugh-Nagumo model
        return : err, error
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
    # compute the estimator
    err = np.sqrt(np.sum((Q - np.polynomial.legendre.legval2d(shifted(a_samples, 0.6, 0.8),
                                                              shifted(b_samples, 0.7, 0.9), c)) ** 2) / M) / np.sqrt(M)
    return err


# TO CHECK

def MCLS_multiple(a_samples, b_samples, n, epsilon, I, v0, w0, t0, T, Nt):
    # I have some doubts about the range of a and b (this sampling is only for solving the LS problem)
    a = np.random.uniform(0.6, 0.8, len(a_samples))
    #a = np.random.uniform(0, 1, len(a_samples))
    x = a
    y=np.zeros(len(a))
    for i in range(len(a)):
        y[i] = MCLS_interior(b_samples, n, epsilon, I, v0, w0, t0, T, Nt, a[i])
    # solve the least squares problem
    c = np.polynomial.legendre.Legendre.fit(x, y, n, domain=[0, 1]).coef
    # use the obtained coefficients to compute the estimator
    samples2 = a_samples*2 - 1
    interior_integral= np.zeros(len(a_samples))
    for i in range(len(a_samples)):
        interior_integral[i] = MCLS_interior(b_samples, n, epsilon, I, v0, w0, t0, T, Nt, a_samples[i])
    # in this case, the function we want to approximate is the interior integral
    estim = 0.2*(np.sum(interior_integral - np.polynomial.legendre.legval(samples2, c)) / len(a_samples) + c[0])
    return estim

def MCLS_interior (b_samples, n, epsilon, I, v0, w0, t0, T, Nt, a):
    b = np.random.uniform(0.7, 0.9, len(b_samples))
    #b = np.random.uniform(0, 1, len(b_samples))
    x = b
    y=np.zeros(len(b))
    for i in range(len(b)):
        y[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a, b[i])
    # solve the least squares problem
    c = np.polynomial.legendre.Legendre.fit(x, y, n, domain=[0, 1]).coef
    # use the obtained coefficients to compute the estimator
    samples2 = b_samples*2 - 1
    Q = np.zeros(len(b_samples))
    for i in range(len(b_samples)):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a, b_samples[i])
    estim = 0.2*(np.sum(Q - np.polynomial.legendre.legval(samples2, c)) / len(b_samples) + c[0])
    return estim

def IS_MCLS_multiple(a_samples, b_samples, n, epsilon, I, v0, w0, t0, T, Nt):
    weights = 1 / h(a_samples, n)
    x = sample_from_h_new(len(a_samples), 1000, n)
    y=np.zeros(len(x))
    for i in range(len(x)):
        y[i] = IS_MCLS_interior(b_samples, n, epsilon, I, v0, w0, t0, T, Nt, x[i])
    # solve the weighted least squares problem
    c, diagnostics= np.polynomial.legendre.Legendre.fit(x, y, n, domain=[0, 1], full=True, w=np.sqrt(weights))
    c = c.coef
    # use the obtained coefficients to compute the estimator
    samples2 = a_samples*2 - 1
    interior_integral= np.zeros(len(a_samples))
    for i in range(len(a_samples)):
        interior_integral[i] = IS_MCLS_interior(b_samples, n, epsilon, I, v0, w0, t0, T, Nt, a_samples[i])
    estim = np.sum(np.dot(interior_integral - np.polynomial.legendre.legval(samples2, c), weights)) / np.sum(weights) + c[0]
    return estim

def IS_MCLS_interior (b_samples, n, epsilon, I, v0, w0, t0, T, Nt, a):
    weights = 1 / h(b_samples, n)
    x = sample_from_h_new(len(b_samples), 1000, n)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a, x[i])
    # solve the weighted least squares problem
    c, diagnostics = np.polynomial.legendre.Legendre.fit(x, y, n, domain=[0, 1], full=True, w=np.sqrt(weights))
    c = c.coef
    # use the obtained coefficients to compute the estimator
    samples2 = b_samples*2 - 1
    Q = np.zeros(len(b_samples))
    for i in range(len(b_samples)):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a, b_samples[i])
    estim = np.sum(np.dot(Q - np.polynomial.legendre.legval(samples2, c), weights)) / np.sum(weights) + c[0]

    return estim

def ff(x, y):
    return f(x) * f(y)

def MCLS_2D_test(a_samples, b_samples, ff, n):
    M = len(a_samples)
    # compute the 2D Vandermonde matrix
    x = np.random.uniform(0, 1, M)
    y = np.random.uniform(0, 1, M)
    V = np.polynomial.legendre.legvander2d(shifted(x, 0, 1), shifted(y, 0, 1), (n, n))
    cond = np.linalg.cond(V)
    # solve the least squares problem
    c = np.linalg.lstsq(V, ff(x, y), rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    # compute the estimator
    estim = np.sum(ff(a_samples, b_samples) - np.polynomial.legendre.legval2d(shifted(a_samples, 0, 1), shifted(b_samples, 0, 1), c)) / M + c[0, 0]
    return estim, cond

def MCLS_prime_2D_test(a_samples, b_samples, ff, n):
    M = len(a_samples)
    # compute the 2D Vandermonde matrix
    V = np.polynomial.legendre.legvander2d(shifted(a_samples, 0, 1), shifted(b_samples, 0, 1), (n, n))
    cond = np.linalg.cond(V)
    # solve the least squares problem
    c = np.linalg.lstsq(V, ff(a_samples, b_samples), rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    # compute the estimator
    estim = c[0, 0]
    return estim, cond

def CI_MCLS_2D(a_samples, b_samples, n, epsilon, I, v0, w0, t0, T, Nt, estim, alpha):
    """
    Compute the confidence interval of the MCLS estimator.
    args : a_samples, samples x drawn from uniform distribution U([0.6, 0.8])
            b_samples, samples y drawn from uniform distribution U([0.7, 0.9])
            n, maximal exponential of the Legendre polynomials
            epsilon, I, v0, w0, t0, T, Nt, parameters of the Fitzhugh-Nagumo model
    return : CI, confidence interval for the MCLS estimator based on these samples
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
    # compute the estimator
    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    err = np.sqrt(np.sum((Q - np.polynomial.legendre.legval2d(shifted(a_samples, 0.6, 0.8), shifted(b_samples, 0.7, 0.9),c))**2)/M)/np.sqrt(M)
    CI = estim + np.array([-1, 1]) * quantile * err
    return CI, err