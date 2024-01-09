from question2 import f
from question5 import shifted
import numpy as np
import matplotlib.pyplot as plt


def ff(x, y):
    return f(x) * f(y)


def ls_test_2D(n, M):
    """
    x_samples = np.random.uniform(0, 1, np.floor(np.sqrt(M).astype(int)))
    y_samples = np.random.uniform(0, 1, np.floor(np.sqrt(M).astype(int)))

    X, Y = np.meshgrid(shifted(x_samples, 0, 1), shifted(y_samples, 0, 1), (n, n))
    
    #x = np.random.uniform(0, 1, M)
    #y = np.random.uniform(0, 1, M)
    V = np.polynomial.legendre.legvander2d(X.flatten(), Y.flatten(), [n, n])
    
    c = np.linalg.lstsq(V, ff(X, Y), rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    
    ff_values = ff(X, Y)
    ls_estimate = np.polynomial.legendre.legval2d(X.flatten(), Y.flatten(), c)
    
    # Compare values element-wise
    comparison = np.column_stack((ff_values, ls_estimate))
    print("Original Function vs. Least Squares Estimate:")
    print("x_samples   y_samples   Original Function   LS Estimate")
    for i in range(M):
        print(f"{x_samples[i]:.4f}      {y_samples[i]:.4f}      {ff_values[i]:.6f}       {ls_estimate[i]:.6f}")
    """
    
    x_samples = np.random.uniform(0, 1, np.floor(np.sqrt(M)).astype(int))
    y_samples = np.random.uniform(0, 1, np.floor(np.sqrt(M)).astype(int))

    X, Y = np.meshgrid(x_samples, y_samples)
    X_shifted, Y_shifted = np.meshgrid(shifted(x_samples, 0, 1), shifted(y_samples, 0, 1))

    ff_ls = ff(X, Y)
    
    X_flat, Y_flat, X_shifted_flat, Y_shifted_flat = X.flatten(), Y.flatten(), X_shifted.flatten(), Y_shifted.flatten()
    ff_ls = ff_ls.flatten()
    
    #x = np.random.uniform(0, 1, M)
    #y = np.random.uniform(0, 1, M)
    V = np.polynomial.legendre.legvander2d(X_shifted_flat, Y_shifted_flat, [n, n])
    
    c = np.linalg.lstsq(V, ff_ls, rcond=None)[0]
    c = c.reshape((n + 1, n + 1))
    
    ff_values = ff(X, Y).reshape(-1)
    ls_estimate = np.polynomial.legendre.legval2d(X_shifted, Y_shifted, c).reshape(-1)
    
    # Compare values element-wise
    comparison = np.column_stack((ff_values, ls_estimate))
    print("Original Function vs. Least Squares Estimate:")
    print("x_samples   y_samples   Original Function   LS Estimate")
    for i in range(X_flat.size):
        print(f"{X_flat[i]:.4f}      {Y_flat[i]:.4f}      {ff_values[i]:.9f}       {ls_estimate[i]:.9f}")


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