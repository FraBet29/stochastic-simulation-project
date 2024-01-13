from question2 import *
import numpy as np
import matplotlib.pyplot as plt

#######################################################
# Question 2: estimating integral I with CMC and MCLS #
#######################################################

### CMC ###

# exact value of the integral (reference)
ref_value = (1/5) * np.arctan(5)

N = 100
nb_samples = np.logspace(np.log10(10), np.log10(2000), num=N, dtype=int) # evenly spaced values on a logarithmic scale

# crude Monte Carlo estimator
CMC_estims = np.zeros(N)

# confidence interval
alpha = 0.05
CMC_conf = np.zeros((N, 2))

for M in range(N):
    unif_samps = np.random.uniform(0, 1, nb_samples[M])
    CMC_estims[M], CMC_conf[M] = crude_MC(unif_samps, f, alpha)

# plot log-log graph to see the order of the error
loglog_graph(nb_samples, CMC_estims, ref_value)

# plot semi-log graph to see the confidence intervals
plot_CI(nb_samples, CMC_estims, CMC_conf, ref_value, alpha)

# averaging the absolute error, i.e. doing 'averaging' experiments for each value of M (number of samples)
averaging = 20
CMC_estims_av = np.zeros((N, averaging))
for M in range(N):
    CMC_estims_M = np.zeros(averaging)
    for i in range(averaging):
        unif_samps = np.random.uniform(0, 1, nb_samples[M])
        CMC_estims_M[i], _ = crude_MC(unif_samps, f, alpha)
    CMC_estims_av[M][:] = CMC_estims_M

loglog_average_error_graph(nb_samples, CMC_estims_av, ref_value)

### MCLS ###

# maximum degree of the Legendre polynomials
trials_n = [5, 10, 20, 30, -1, -2]
nb_trials_n = len(trials_n)

# MCLS estimators
MCLS_estims = [-1 * np.ones(N) for _ in range(nb_trials_n)]
MCLS_prime_estims = [-1 * np.ones(N) for _ in range(nb_trials_n)]

# condition number of the Vandermonde matrix
MCLS_cond = [-1 * np.ones(N) for _ in range(nb_trials_n)]
MCLS_prime_cond = [-1 * np.ones(N) for _ in range(nb_trials_n)]

for M in range(N):
    for i in range(nb_trials_n):
        
        n = trials_n[i]

        if n == -1:
            n = np.ceil(np.sqrt(nb_samples[M])).astype(int)
            if nb_samples[M] > 5000: # too computationally expensive
                break
        if n == -2:
            n = np.ceil((nb_samples[M])/2).astype(int)
            if nb_samples[M] > 5000: # too computationally expensive
                break

        if n < nb_samples[M]: # avoid underdetermined least squares
            unif_samps = np.random.uniform(0, 1, nb_samples[M])
            MCLS_estims[i][M], MCLS_cond[i][M] = MCLS(unif_samps, f, n)
            MCLS_prime_estims[i][M], MCLS_prime_cond[i][M] = MCLS_prime(unif_samps, f, n)

# plot log-log graph to see the order of the error
multiple_loglog_graph(nb_samples, MCLS_estims, ref_value, trials_n)
multiple_loglog_graph(nb_samples, MCLS_prime_estims, ref_value, trials_n)

# plot log-log graph to see the evolution of the condition number
multiple_cond_loglog_graph(nb_samples, MCLS_cond, trials_n)