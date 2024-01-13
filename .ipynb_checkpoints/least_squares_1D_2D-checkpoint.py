from question2 import f
from question5 import shifted
import numpy as np
import matplotlib.pyplot as plt


def ff(x, y):
    return f(x) * f(y)


def ls_test_2D(n, M):

    x_samples = np.random.uniform(0, 1, M)
    y_samples = np.random.uniform(0, 1, M)
    
    x = np.random.uniform(0, 1, M)
    y = np.random.uniform(0, 1, M)
    V = np.polynomial.legendre.legvander2d(shifted(x, 0, 1), shifted(y, 0, 1), [n, n])
    
    c = np.linalg.lstsq(V, ff(x, y), rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    
    ff_values = ff(x_samples, y_samples)
    ls_estimate = np.polynomial.legendre.legval2d(shifted(x_samples, 0, 1), shifted(y_samples, 0, 1), c)
    
    # Compare values element-wise
    comparison = np.column_stack((ff_values, ls_estimate))
    print("Original Function vs. Least Squares Estimate:")
    print("x_samples   y_samples   Original Function   LS Estimate")
    for i in range(M):
        print(f"{x_samples[i]:.4f}      {y_samples[i]:.4f}      {ff_values[i]:.6f}       {ls_estimate[i]:.6f}")


def ls_test_1D(n, M):
    
    x_samples = np.random.uniform(0, 1, M)
    V = np.polynomial.legendre.legvander(shifted(x_samples, 0, 1), n)
    
    c = np.linalg.lstsq(V, f(x_samples), rcond=None)[0]
    
    f_values = f(x_samples)
    ls_estimate = np.polynomial.legendre.legval(shifted(x_samples, 0, 1), c)
    
    # Compare values element-wise
    comparison = np.column_stack((f_values, ls_estimate))
    print("Original Function vs. Least Squares Estimate:")
    print("x_samples   Original Function   LS Estimate")
    for i in range(M):
        print(f"{x_samples[i]:.4f}      {f_values[i]:.6f}       {ls_estimate[i]:.6f}")