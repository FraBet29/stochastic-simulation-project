import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import solve_ivp

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