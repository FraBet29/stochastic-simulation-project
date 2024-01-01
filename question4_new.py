import numpy as np 
import math
import matplotlib.pyplot as plt

########################################
# Question 4 : MCLS estimators with IS #
########################################

def eval_phi_squared(x, n):
    c_vector = np.zeros(n+1)
    c_vector[n] = 1
    return np.square(np.polynomial.legendre.legval(2*x-1, c_vector))*(2*n+1)


def g(x):
    return 1 / (np.pi * np.sqrt(x * (1 - x)))


def G(x):
    return 2 * np.arcsin(np.sqrt(x)) / np.pi


def G_inv(x):
    return np.sin(np.pi * x / 2) ** 2


def sample_from_g_new(M):
    """
    Sample from pdf g by Inverse transform method.
    args: M, number of samples to draw
    return: g_samples, array of the M samples from pdf g
    """
    unif_samples = np.random.uniform(0, 1, M)
    return G_inv(unif_samples)


def sample_from_h_new(M, ite_max, n):
    """
    Sample from pdf h by composition method + Acceptance-Rejection method.
    args: M, number of samples to draw
          ite_max, maximal number of iterations allowed (in case rejectance rate would be too high)
          n, Legendre polynomials up to degree n will be used
    return: h_samples, array of the M samples from pdf g
    """
    h_samples = np.zeros(M)
    C = 4 * math.e
    
    tot_ite = 0
    
    for m in range(M):
        
		# Sample an index from 0 to n
        j = np.random.randint(0, n + 1)
        
		# Apply the AR method to sample from phi_j^2
        ite = 0
        while ite < ite_max:
            Y_sample = sample_from_g_new(1)
            U_sample = np.random.uniform(0, 1, 1)
            if U_sample <= eval_phi_squared(Y_sample, j) / (C * g(Y_sample)):
                h_samples[m] = Y_sample
                break
            ite += 1
        
        if ite >= ite_max:
            raise Exception("Acceptance rate too low")
        tot_ite += ite
    
    print('Acceptance rate: ', M / tot_ite)
    
    return h_samples