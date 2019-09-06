"""
 Implement importance sampler of target cauchy gamma = 1 with proposal N(0,1)
 
"""

import numpy as np
from scipy.stats import cauchy, norm
import matplotlib.pyplot as plt

g = 1

## to control the cauchy is what we think it is haha
# x = np.linspace(cauchy.ppf(0.01), cauchy.ppf(0.99), 1000)
# fig, ax = plt.subplots(1,1)
# ax.plot(x, cauchy.pdf(x, scale=g), label='scipy')
# ax.plot(x, g/(np.pi*(g**2 + x**2)), '--', label='custom')
# plt.legend()
# plt.show()

# sample N x's

N = 10000
mc_runs = 10000
W_tilde = np.zeros((mc_runs, N))
for mc in range(mc_runs):
    x = norm.rvs(size=N, loc=0, scale=1)
    q_pdf = norm.pdf(x, loc=0, scale=1)
    pi_pdf = 1/(1+x**2)
    w_tilde = pi_pdf/q_pdf
    w = w_tilde/w_tilde.sum()
    W_tilde[mc, :] = w_tilde


plt.hist(np.mean(W_tilde[:, 0:100], axis=1), range=(2, 5), bins=100, density=True, alpha=0.6, label='N=100')
plt.hist(np.mean(W_tilde[:, 0:1000], axis=1), range=(2, 5), bins=100, density=True, alpha=.6, label='N=1000')
plt.hist(np.mean(W_tilde[:, 0:10000], axis=1), range=(2, 5), bins=100, density=True, alpha=.6, label='N=10000')
plt.vlines(np.mean(W_tilde[:, 0:100]), 0, 5, 'C0', linestyles='--')
plt.vlines(np.mean(W_tilde[:, 0:1000]), 0, 5, 'C1', linestyles='--')
plt.vlines(np.mean(W_tilde[:, 0:10000]), 0, 5, 'C2', linestyles='--')
plt.legend()
# plt.xticks([, np.pi, 3], ('2.0','Z=pi','3.0'))
plt.vlines(np.pi,0,5)
plt.title('Estimates of normalizing constant Z for different N')
plt.show()
