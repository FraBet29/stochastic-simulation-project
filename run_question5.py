from question5 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from plots import *
import time

#################################################################
# Question 5: application to the Fitzhugh-Nagumo system of ODES #
#################################################################

### solve the system of ODEs ###

# problem data
epsilon = 0.08
I = 1.0
v0 = 0
w0 = 0
t0 = 0
T = 10
Nt = 1000

# uniform sampling of a and b
a = np.random.uniform(0.6, 0.8)
b = np.random.uniform(0.7, 0.9)

v, w, t = solve_FHN(epsilon, I, v0, w0, t0, T, Nt, a, b)
v_scipy, w_scipy, t_scipy = solve_FHN_scipy(epsilon, I, v0, w0, t0, T, Nt, a, b)

# plot the solution
plt.plot(t, v)
plt.plot(t, w)
plt.plot(t_scipy, v_scipy)
plt.plot(t_scipy, w_scipy)
plt.legend(['v', 'w', 'v_scipy', 'w_scipy'])
plt.show()

### estimate the integral of Q ###

compute_ref = False

# compute a reference value for CMC by using high number of sample points
# WARNING: it takes a long time to run
if compute_ref:
    N = int(1e5)
    a_unif = np.random.uniform(0.6, 0.8, N)
    b_unif = np.random.uniform(0.7, 0.9, N)
    ref_value_2D = crude_MC_2D(a_unif, b_unif, epsilon, I, v0, w0, t0, T, Nt)

ref_value_2D = 1.174996467515269

N = 100
nb_samples = np.logspace(np.log10(10), np.log10(1000), num=N, dtype=int) # evenly spaced values on a logarithmic scale

# crude Monte Carlo estimator
CMC_estims = np.zeros(N)

# confidence interval
alpha = 0.05
CMC_conf = np.zeros((N, 2))

for M in range(N):
    a_samples = np.random.uniform(0.6, 0.8, nb_samples[M])
    b_samples = np.random.uniform(0.7, 0.9, nb_samples[M])
    CMC_estims[M], CMC_conf[M] = crude_MC_2D(a_samples, b_samples, epsilon, I, v0, w0, t0, T, Nt, alpha)

# plot log-log graph to see the order of the error
loglog_graph(nb_samples, CMC_estims, ref_value_2D)

# plot semi-log graph to see the confidence intervals
plot_CI(nb_samples, CMC_estims, CMC_conf, ref_value_2D, alpha)

# compute a reference value for MCLS by using high number of sample points
# WARNING: it takes a long time to run
if compute_ref:
    N = int(1e4)
    n = 3
    a_unif = np.random.uniform(0.6, 0.8, N)
    b_unif = np.random.uniform(0.7, 0.9, N)
    ref_value_2D_MCLS, _, _ = MCLS_2D(a_unif, b_unif, n, epsilon, I, v0, w0, t0, T, Nt)
    ref_value_2D_MCLS_prime, _, _ = MCLS_prime_2D(a_unif, b_unif, n, epsilon, I, v0, w0, t0, T, Nt)

ref_value_2D_MCLS = 1.1749791860591807
ref_value_2D_MCLS_prime = 1.174979186071431

# maximum degree of the Legendre polynomials
trials_n = [0, 1, 2, 3]
nb_trials_n = len(trials_n)

# MCLS estimators
MCLS_estims_2D = [-1 * np.ones(N) for _ in range(nb_trials_n)]
MCLS_prime_estims_2D = [-1 * np.ones(N) for _ in range(nb_trials_n)]

# confidence intervals
alpha = 0.05
MCLS_CI_2D = [-1 * np.ones((N,2)) for _ in range(nb_trials_n)]

# error estimators
err_MCLS_2D = [-1 * np.ones(N) for _ in range(nb_trials_n)]

tic = time.time()

for M in range(N):
    for i in range(nb_trials_n):

        if (M * nb_trials_n + i) % 10 == 0:
            print(f'Iteration {M * nb_trials_n + i} of {N * nb_trials_n}')
        
        n = trials_n[i]

        if n < nb_samples[M]:
            a_unif_samps = np.random.uniform(0.6, 0.8, nb_samples[M])
            b_unif_samps = np.random.uniform(0.7, 0.9, nb_samples[M])
            MCLS_estims_2D[i][M], _, MCLS_CI_2D[i][M]= MCLS_2D(a_unif_samps, b_unif_samps, n, epsilon, I, v0, w0, t0, T, Nt, alpha)
            err_MCLS_2D[i][M] = calculate_error(a_samples, b_samples, n, epsilon, I, v0, w0, t0, T, Nt)
            MCLS_prime_estims_2D[i][M], _, _ = MCLS_prime_2D(a_unif_samps, b_unif_samps, n, epsilon, I, v0, w0, t0, T, Nt)

toc = time.time()
print(f'Elapsed time: {toc - tic}s')

# plot log-log graph to see the order of the error
multiple_loglog_graph(nb_samples, MCLS_estims_2D, ref_value_2D_MCLS, trials_n)
multiple_loglog_graphp(nb_samples, MCLS_prime_estims_2D, ref_value_2D_MCLS_prime, trials_n)

# plot the confidence interval for a fixed n
i = 3 # fix n to plot the CI for MCLS
plot_CI(nb_samples, MCLS_estims_2D[i], MCLS_CI_2D[i], ref_value_2D_MCLS, alpha)

# plot the error estimator vs the true error
cut = 0
plt.figure()
for i in range(nb_trials_n):
    plt.plot(nb_samples[cut:], np.abs(MCLS_estims_2D[i][cut:] - ref_value_2D_MCLS), label=f'true n={trials_n[i]}')
    plt.plot(nb_samples[cut:], err_MCLS_2D[i][cut:], label=f'estimated n={trials_n[i]}')
plt.title('Error estimator vs true error for different values of n')
plt.xscale('log')
plt.legend(['true error', 'estimated error'])