import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

########################################################################
# Question 5 : Application to the Fitzhugh-Nagumo model system of ODES #
########################################################################

def solve_FHN(epsilon, I, v0, w0, t0, T, Nt, a, b):
    """
    Solve the Fitzhugh-Nagumo model system of ODEs with forward Euler scheme.
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
    dt = (T-t0)/Nt
    # time vector
    t = np.linspace(t0, T, Nt+1)
    # initial values
    v = np.zeros(Nt+1)
    w = np.zeros(Nt+1)
    v[0] = v0
    w[0] = w0
    # forward Euler scheme
    for i in range(Nt):
        v[i+1] = v[i] + dt*(v[i] - np.power(v[i], 3)/3 - w[i] + I)
        w[i+1] = w[i] + dt*epsilon*(v[i] + a - b*w[i])
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
    t = np.linspace(t0, T, Nt+1)
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
    dt = (T-t0)/Nt
    return (np.sum(np.square(v))-v[0]**2-v[len(v)-1]**2)*dt


def MCLS_multiple(a_samples, b_samples, n, epsilon, I, v0, w0, t0, T, Nt):
    a = np.random.uniform(0.6, 0.8, len(a_samples))
    x = a
    y=np.zeros(len(a))
    for i in range(len(a)):
        y[i] = MCLS_interior(b_samples, n, epsilon, I, v0, w0, t0, T, Nt, a[i])
    c = np.polynomial.legendre.Legendre.fit(x, y, n, domain=[0, 1]).coef
    samples2 = a_samples*2 - 1
    interior_integral= np.zeros(len(a_samples))
    for i in range(len(a_samples)):
        interior_integral[i] = MCLS_interior(b_samples, n, epsilon, I, v0, w0, t0, T, Nt, a_samples[i])
    estim = np.sum(interior_integral - np.polynomial.legendre.legval(samples2, c)) / len(a_samples) + c[0]
    return estim

def MCLS_interior (b_samples, n, epsilon, I, v0, w0, t0, T, Nt, a):
    b = np.random.uniform(0.7, 0.9, len(b_samples))
    x = b
    y=np.zeros(len(b))
    for i in range(len(b)):
        y[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a, b[i])
    c = np.polynomial.legendre.Legendre.fit(x, y, n, domain=[0, 1]).coef
    samples2 = b_samples*2 - 1
    Q = np.zeros(len(b_samples))
    for i in range(len(b_samples)):
        Q[i] = calculate_Q(epsilon, I, v0, w0, t0, T, Nt, a, b_samples[i])
    estim = np.sum(Q - np.polynomial.legendre.legval(samples2, c)) / len(b_samples) + c[0]
    return estim