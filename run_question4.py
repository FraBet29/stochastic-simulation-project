from question4 import *
from question2 import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time

#############################################
# Question 4: estimating integral I with IS #
#############################################

# test sampling from bound g, based on inverse transform method
Y_samples = sample_from_g(1000)
visualize_cdf_from_samples(Y_samples)

# test sampling from h = 1 / w, done by composition method + AR method
H_samples = sample_from_h(1000, 100, 5, True)
visualize_cdf_from_samples(H_samples)

# exact value of the integral (reference)
ref_value = (1/5) * np.arctan(5)

N = 100
nb_samples = np.logspace(np.log10(10), np.log10(2000), num=N, dtype=int) # evenly spaced values on a logarithmic scale

trials_n = [5, 10, 20, -1, -2, -3]
nb_trials_n = len(trials_n)

# MCLS estimators
IS_MCLS_estims = [-1 * np.ones(N) for _ in range(nb_trials_n)]
IS_MCLS_prime_estims = [-1 * np.ones(N) for _ in range(nb_trials_n)]

# condition number of the Vandermonde matrix
IS_MCLS_cond = [-1 * np.ones(N) for _ in range(nb_trials_n)]
IS_MCLS_prime_cond = [-1 * np.ones(N) for _ in range(nb_trials_n)]

tic = time.time()

for M in range(N):
    for i in range(nb_trials_n):

        if (M * nb_trials_n + i) % 50 == 0:
            print(f'Iteration {M * nb_trials_n + i} of {N * nb_trials_n}')
        
        n = trials_n[i]

        if n == -1:
            n = np.ceil(np.sqrt(nb_samples[M])).astype(int)
            if nb_samples[M] > 5000: # too computationally expensive
                break
        if n == -2:
            n = np.ceil((nb_samples[M])/2).astype(int)
            if nb_samples[M] > 5000: # too computationally expensive
                break
        if n == -3:
            n = np.rint(np.real(nb_samples[M] / scipy.special.lambertw(nb_samples[M]))).astype(int)
            if nb_samples[M] > 5000:
                break

        if n < nb_samples[M]:
            h_samples = sample_from_h(nb_samples[M], 1000, n)
            IS_MCLS_estims[i][M], IS_MCLS_cond[i][M] = IS_MCLS(h_samples, f, n)
            IS_MCLS_prime_estims[i][M], IS_MCLS_prime_cond[i][M] = IS_MCLS_prime(h_samples, f, n)

toc = time.time()
print(f'Elapsed time: {toc - tic}s')

# plot log-log graph to see the order of the error
multiple_loglog_graph(nb_samples, IS_MCLS_estims, ref_value, trials_n)
multiple_loglog_graph(nb_samples, IS_MCLS_prime_estims, ref_value, trials_n)

# plot log-log graph to see the evolution of the condition number
multiple_cond_loglog_graph(nb_samples, IS_MCLS_cond, trials_n)
multiple_cond_loglog_graph(nb_samples, IS_MCLS_prime_cond, trials_n)