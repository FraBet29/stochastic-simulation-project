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