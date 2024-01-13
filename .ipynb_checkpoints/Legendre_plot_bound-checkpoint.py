import numpy as np 
import matplotlib.pyplot as plt

# Evaluate the square of the Legendre polynomial of degree n at some points
def eval_phi_squared(x, n):
    c_vector = np.zeros(n+1)
    c_vector[n] = 1
    return np.square(np.polynomial.legendre.legval(2 * x - 1, c_vector) * np.sqrt(2 * n + 1)) # need to add the normalization constant

# Arcsine density
def g(x):
    return 1 / (np.pi * np.sqrt(x * (1 - x)))

C = 4 * np.e
n = 20
x = np.linspace(0.001, 0.999, 1000)
plt.plot(x, eval_phi_squared(x, n))
plt.plot(x, C * g(x))
print(np.all(eval_phi_squared(x, n) < C * g(x)))
plt.show