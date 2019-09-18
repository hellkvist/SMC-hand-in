import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma
from scipy.special import gamma
from numpy.random import choice


def state_rvs(x, phi, sigma): # state propagation function
    return norm.rvs(loc=phi*x, scale=sigma, size=len(x))


def weights_pdf(y, x, beta): # weight computation function
    return norm.pdf(y, loc=0, scale=beta*np.exp(x/2))


def logweights_pdf(y, x, beta): # computation of the logweights for numerical robustness
    return norm.logpdf(y, loc=0, scale=beta*np.exp(x/2))


def gauss_rw_rvs(x0, tau):# the random walk is done in beta, and sigma, and then squared to get beta^2 and sigma^2
    return norm.rvs(loc=x0, scale=tau, size=1)


def gauss_rw_pdf(x_, x0, tau): 
    return norm.pdf(x_, loc=x0, scale=tau)


def prior_logpdf(x_, a, b): # this is the prior for beta^2 and sigma^2
    return np.log(b**a/gamma(a) * x_**(-a-1)) - b/x_


def bpf(sigma, beta):
    N = 10
    loglikelihood_ack_t = 0
    x_bpf = norm.rvs(loc=0, scale=sigma, size=N)
    logw_tilde = logweights_pdf(y[0], x_bpf, beta)
    max_logw = np.max(logw_tilde)
    w = np.exp(logw_tilde - max_logw)/(np.exp(logw_tilde - max_logw)).sum()
    for t in range(1, T):
        a = choice(N, size=N, replace=1, p=w)
        x_bpf = state_rvs(x_bpf[a], phi, sigma)

        logw_tilde = logweights_pdf(y[t], x_bpf, beta)
        max_logw = np.max(logw_tilde)
        logw_diffmax = logw_tilde - max_logw
        w = np.exp(logw_diffmax)/(np.exp(logw_diffmax)).sum()
        loglikelihood_ack_t += np.log(np.sum(np.exp(logw_diffmax))) + max_logw - np.log(N)

    return loglikelihood_ack_t


y = np.genfromtxt('seOMXlogreturns2012to2014.csv')
T = len(y)

phi = 0.985

M = 10000
a = 0.01
b = 0.01
tau =  0.1
beta2 = np.zeros(M)
sigma2 = np.zeros(M)
# intialize
beta2[0] = 0.7
sigma2[0] = 0.1
z = bpf(sigma2[0]**.5, beta2[0]**.5)
for m in range(1, M):
    beta2_prev = beta2[m-1]
    sigma2_prev = sigma2[m-1]
    if m % 100 == 0:
        print(m, '/', M)
    beta_ = gauss_rw_rvs(beta2_prev**.5, tau)
    sigma_ = gauss_rw_rvs(sigma2_prev**.5, tau)
    if beta_ > 0 and sigma_ > 0:
        beta2_ = beta_**2
        sigma2_ = sigma_**2
        z_ = bpf(sigma2_**.5, beta2_**.5)  # is the log-likelihood

        # take logarithms of the densities
        a_num = z_ + prior_logpdf(beta2_, a, b) + prior_logpdf(sigma2_, a, b)

        a_den = z + prior_logpdf(beta2_prev, a, b) + prior_logpdf(sigma2_prev, a, b)

        # remember to take exponential of the logarithmic expressions
        alpha = np.min((1, np.exp(a_num-a_den)))
        if np.random.uniform() < alpha:
            beta2[m] = beta2_
            sigma2[m] = sigma2_
            z = z_
        else:
            beta2[m] = beta2_prev
            sigma2[m] = sigma2_prev
    else:
        beta2[m] = beta2_prev
        sigma2[m] = sigma2_prev

plt.subplot(121)
plt.hist(beta2, bins=100, density=1)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma2, bins=100, density=1)
plt.xlabel(r'$\sigma^2$')


plt.subplot(121)
plt.hist(beta2[1500::], bins=100, density=1)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma2[1500::], bins=100, density=1)
plt.xlabel(r'$\sigma^2$')

plt.show()
