import numpy as np
from scipy.stats import norm
from scipy.special import gamma
import matplotlib.pyplot as plt

def log_invgamma(x, a, b):
    return np.log(b**a/gamma(a) * x**(-a-1)) - b/x

def bpf_propagate(x_prev, phi, sigma):
    return np.random.normal(loc=phi*x_prev, scale=sigma)

def log_weights(y, x, beta):
    variance = beta**2*np.exp(x)
    return np.log(1/np.sqrt(2*np.pi * variance)) - y**2/(2*variance)

def log_lik_bpf(y, N, phi, sigma, beta):
    log_z = 0

    T = len(y)
    log_N = np.log(N)

    x = np.random.normal(loc=0, scale=sigma, size=N)

    # set initial weights 
    w = 1/N * np.ones(N)

    for t in range(T):
        # resample with linear weights
        idx = np.random.choice(N, size=N, replace=True, p=w)

        # propagate particles
        x = bpf_propagate(x[idx], phi, sigma)

        # compute logweights
        log_w = log_weights(y[t], x, beta)
        c = np.max(log_w)
        v = log_w - c

        # compute normalized linear weights
        exp_v = np.exp(v)
        w = exp_v/exp_v.sum()

        # add to the ackumulator for log likelihood
        log_z += c + np.log(np.sum(np.exp(v))) - log_N

    return log_z

def pmh(y, M, N, phi, step_size_beta, step_size_sigma):
    beta = np.zeros(M)
    sigma = np.zeros(M)
    log_z = np.zeros(M)

    # initialize parameters beta and sigma
    beta[0] = 10
    sigma[0] = 10

    beta_prev = beta[0]
    sigma_prev = sigma[0]
    log_z[0] = log_lik_bpf(y, N, phi, sigma_prev, beta_prev)

    # compute the logarithm of the denumerator in acceptance probability
    log_den = log_z[0] + log_invgamma(beta_prev**2, 0.01, 0.01) + log_invgamma(sigma_prev**2, 0.01, 0.01)

    for m in range(M):
        if m % 100 == 0:
            print(m, '/', M)
        
        # do the Gaussian random walk in beta and sigma
        beta_prop = np.random.normal(loc=beta_prev, scale=step_size_beta)
        sigma_prop = np.random.normal(loc=sigma_prev, scale=step_size_sigma)

        # a negative value is proposed for any of the two, neclegt and iterate m
        if sigma_prop >= 0 and beta_prop >= 0:
            u = np.random.uniform()
            
            # run BPF for the proposed parameters
            log_z_prop = log_lik_bpf(y, N, phi, sigma_prop, beta_prop)

            # compute the numerator of the acceptance probability
            log_num = log_z_prop + log_invgamma(beta_prop**2, 0.01, 0.01) + log_invgamma(sigma_prop**2, 0.01, 0.01)

            alpha = np.min((1, np.exp(log_num - log_den)))
            if u <= alpha: # accept
                beta[m] = beta_prop
                sigma[m] = sigma_prop
                log_z[m] = log_z_prop

                beta_prev = beta_prop
                sigma_prev = sigma_prop
                log_den = log_num
            else: # reject
                beta[m] = beta_prev
                sigma[m] = sigma_prev
                log_z[m] = log_z[m-1]
        else:
            beta[m] = beta_prev
            sigma[m] = sigma_prev
            log_z[m] = log_z[m-1]

    return (beta, sigma, log_z)


y = np.genfromtxt('Hand-in/OMXLogReturns.csv', delimiter=',')
phi = 0.985
step_size_beta = 1
step_size_sigma = 0.1
M = 10000
N = 100

beta, sigma, log_z = pmh(y, M, N, phi, step_size_beta, step_size_sigma)

plt.figure(1)
plt.subplot(121)
plt.hist(beta**2, bins=50)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma**2, bins=50)
plt.xlabel(r'$\sigma^2$')

plt.figure(2)
plt.subplot(121)
plt.hist(beta[0:1000]**2, bins=50)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma[0:1000]**2, bins=50)
plt.xlabel(r'$\sigma^2$')

plt.figure(3)
plt.subplot(121)
plt.hist(beta[1000::]**2, bins=50)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma[1000::]**2, bins=50)
plt.xlabel(r'$\sigma^2$')

plt.figure(4)
plt.subplot(121)
plt.hist(beta[1000::]**2, bins=50)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma[1000::]**2, bins=50)
plt.xlabel(r'$\sigma^2$')

plt.figure(5)
plt.subplot(121)
plt.hist(beta[2000::]**2, bins=50)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma[2000::]**2, bins=50)
plt.xlabel(r'$\sigma^2$')

plt.figure(6)
plt.subplot(211)
plt.plot(beta**2)
plt.xlabel('PMH iteration')
plt.ylabel(r'$\beta^2$')

plt.subplot(212)
plt.plot(sigma**2)
plt.ylabel(r'$\sigma^2$')
plt.xlabel('PMH iteration')
plt.show()