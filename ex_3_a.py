import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import choice

y = np.genfromtxt('OMXLogReturns.csv')
T = len(y)

phi_grid = np.hstack((np.arange(0.1, 1, 0.05), np.arange(0.95, 1, 0.01)))
N_phi = len(phi_grid)

sigma = 0.16
beta = 0.7

M = 10
N = 100

def state_rvs(x, phi):
    return norm.rvs(loc=phi*x, scale=sigma, size=len(x))

def weights_pdf(y, x):
    return norm.pdf(y, loc=0, scale=beta*np.exp(x/2))

loglikelihood = np.zeros((M, N_phi))
for phi_idx in range(N_phi):
    phi = phi_grid[phi_idx]
    for mc in range(M):
        print('phi=', phi, ' mc=', mc)
        loglikelihood_ack_t = 0
        x_bpf = norm.rvs(loc=0, scale=sigma, size=N)
        w_tilde = weights_pdf(y[0], x_bpf)
        w = w_tilde/w_tilde.sum()
        for t in range(1, T):
            a = choice(N, size=N, replace=1, p=w)
            x_bpf = state_rvs(x_bpf[a], phi)
            w_tilde = weights_pdf(y[t], x_bpf)
            w = w_tilde/w_tilde.sum()
            loglikelihood_ack_t += np.log(np.sum(w_tilde)) - np.log(N)
        loglikelihood[mc, phi_idx] = loglikelihood_ack_t

plt.figure()
w = 0.025
plt.boxplot(loglikelihood)
plt.xlim(phi_grid[0]-w, phi_grid[-1]+w)
plt.xticks(np.arange(1, N_phi+2, 2), np.round(phi_grid[0:N_phi+2:2], 3))
plt.xlabel(r'$\phi$')
plt.ylabel('log-likelihood')
plt.show()
        
