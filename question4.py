import numpy as np 
import math
import matplotlib.pyplot as plt

########################################
# Question 4 : MCLS estimators with IS #
########################################

def pdf_h(x, n) :
    """
    Evaluate pdf h =1/w (the one we wnat) at x
    args: x, where to evaluate (float or array)
    return: value of h at x (float or array)
    """
    sum_of_squared = 0
    c_vector = np.zeros(n+1)
    for j in range(n+1) :
        c_vector[j] = 1
        sum_of_squared += np.square(np.polynomial.legendre.legval(2*x-1, c_vector))*(2*j+1)
        c_vector[j] = 0

    return sum_of_squared/(n+1)

def bound_pdf_g(x) :
    """
    Evaluate pdf g (bound on h) at x
    args: x, where to evluate
    return: value of pdf g at x
    """
    denominator = np.pi*np.sqrt(x*(1-x))
    return 1/denominator

def cdf_G(x) :
    """
    Evaluate G (cdf of probability density function g) at x
    We verify easily that the derivative dG/dx = g
    args: x, where to evaluate
    return: Gx, value of cdf G at x
    """
    Gx = 2*np.arcsin(np.sqrt(x))/np.pi
    return Gx

def inverse_cdf_Ginv(x) :
    """
    Evaluate Ginv (inverse cdf of the probability density function g) at x
    args: x, where to evaluate
    return: Ginv_x, value of cdf Ginv at x
    """
    Ginv_x = np.sin(np.pi*x/2)**2
    return Ginv_x

def sample_from_g(M) :
    """
    Sample from pdf g (pdf g being a bound on pdf h = 1/w).
    This is done by Inverse transform method.
    args: M, number of samples to draw
    return: g_samples, array of the M samples from pdf g
    """
    unif_samples = np.random.uniform(0, 1, M)
    return inverse_cdf_Ginv(unif_samples)

def sample_from_h(M, ite_max, n) :
    """
    Sample from pdf h (pdf h=1/w, the one we really want).
    This is done by Acceptance-Rejection method.
    args: M, number of samples to draw
          ite_max, maximal number of iterations allowed (in case rejectance rate would be too high)
          n, Legendre polynomials up to degree n will be used
    return: h_samples, array of the M samples from pdf g
    """
    h_samples = np.zeros(M)
    cst_C = 4*math.e
    k = 0
    ite = 0

    while k<M and ite < ite_max:
        U_sample = np.random.uniform(0, 1, 1)
        Y_sample = sample_from_g(1)

        if U_sample <= pdf_h(Y_sample, n)/(cst_C*bound_pdf_g(Y_sample)) :
            h_samples[k] = Y_sample
            k = k+1
        
        ite = ite+1

    if ite >= ite_max:
        raise Exception("Acceptance rate quite low, ite_max did not suffice :( ")
    
    print('Acceptance rate : ', M/ite)
    
    return h_samples


def visualize_cdf_from_samples(samples):
    """
    Plot a graph of empirical cdf, based on the received samples.
    args: samples, array containing the sample values
    return: /
    """
    M = len(samples)

    # Sort the samples
    sorted_samples = np.sort(samples)

    # Calculate the empirical CDF
    cdf = np.arange(1, M + 1) / M

    # Plot the empirical CDF
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_samples, cdf, label='Empirical CDF')
    plt.title('Empirical CDF')
    plt.xlabel('Samples')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_bound_g_on_h(n_value) :
    # Define the range of x values
    x_values = np.linspace(0.0001, 0.9999, 500)  # Adjust the range as needed

    # Calculate the corresponding y values for pdf_h and bound_pdf_g
    y_bound_pdf_g = bound_pdf_g(x_values)

    # Plotting
    plt.figure(figsize=(8, 6))

    for n in range(0, n_value+1) :
        y_pdf_h = pdf_h(x_values, n)
        plt.plot(x_values, y_pdf_h, label='pdf_h')
    plt.plot(x_values, 4*math.e*y_bound_pdf_g, label='bound_pdf_g')

    plt.title('Comparison of pdf_h and bound_pdf_g')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()


def IS_MCLS(samples, f) :

    return