import numpy as np
from scipy.stats import norm
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def log_invgamma(x, a, b):
    return np.log(b**a/gamma(a) * x**(-a-1)) - b/x

def bpf_propagate(x_prev, phi, sigma):
    return np.random.normal(loc=phi*x_prev, scale=sigma)

def log_weights(y, x, beta):
    variance = beta**2*np.exp(x)
    return np.log(1/np.sqrt(2*np.pi * variance)) - y**2/(2*variance)

def log_lik_bpf(y, N, phi, sigma, beta):   
    '''
        Bootstrap Particle Filter
        returns the logartihm of the likelihood of the parameters sigma and beta
    '''
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
    '''
        Particle Metropolis Hastings
    '''
    beta = np.zeros(M)
    sigma = np.zeros(M)
    log_z = np.zeros(M)
    accepts = 0
    # initialize parameters beta and sigma
    beta[0] = .5
    sigma[0] = .5

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

        # if a negative value is proposed for any of the two, neclegt and iterate m
        if sigma_prop >= 0 and beta_prop >= 0:
            u = np.random.uniform()
            
            # run BPF for the proposed parameters
            log_z_prop = log_lik_bpf(y, N, phi, sigma_prop, beta_prop)

            # compute the numerator of the acceptance probability
            # the denominator will only be updated on accept.
            log_num = log_z_prop + log_invgamma(beta_prop**2, 0.01, 0.01) + log_invgamma(sigma_prop**2, 0.01, 0.01)

            alpha = np.min((1, np.exp(log_num - log_den)))
            if u <= alpha: # accept
                beta[m] = beta_prop
                sigma[m] = sigma_prop
                log_z[m] = log_z_prop

                beta_prev = beta_prop
                sigma_prev = sigma_prop
                log_den = log_num # update the denominator of the acceptance probability, which is the same as the nominator we just computed.
                accepts += 1
            else: # reject
                beta[m] = beta_prev
                sigma[m] = sigma_prev
                log_z[m] = log_z[m-1]
        else:
            beta[m] = beta_prev
            sigma[m] = sigma_prev
            log_z[m] = log_z[m-1]

    print('accept rate: ', accepts/M)
    return (beta, sigma, log_z)


# y = np.genfromtxt('Hand-in/OMXLogReturns.csv', delimiter=',')
y = np.genfromtxt('OMXLogReturns.csv', delimiter=',')
phi = 0.985
step_size_beta = 0.01
step_size_sigma = 0.01
M = 10000
N = 1000

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

plt.figure(7)
plt.plot(log_z, label='log_z')
plt.legend()


fig_final = plt.figure()
# gs = fig_final.add_gridspec(3, 2)
# ax1 = fig_final.add_subplot(gs[0,:])
# ax2 = fig_final.add_subplot(gs[1,:])
# ax3 = fig_final.add_subplot(gs[2,0])
# ax4 = fig_final.add_subplot(gs[2,1])
ax1 = plt.subplot2grid((3,2),(0,0), colspan=2)
ax2 = plt.subplot2grid((3,2),(1,0), colspan=2)
ax3 = plt.subplot2grid((3,2),(2,0))
ax4 = plt.subplot2grid((3,2),(2,1))


ax1.plot(beta**2)
ax1.set_ylabel(r'$\beta^2$')
ax1.set_xlabel('PMH iteration')

ax2.plot(sigma**2)
ax2.set_ylabel(r'$\sigma^2$')
ax2.set_xlabel('PMH iteration')

ax3.hist(beta[2000::]**2, bins=50, density=1)
ax3.set_xlabel(r'$\beta^2$')

ax4.hist(sigma[2000::]**2, bins=50, density=1)
ax4.set_xlabel(r'$\sigma^2$')

fig_final.tight_layout()

fig_final2 = plt.figure()
# gs = fig_final.add_gridspec(3, 2)
# ax1 = fig_final.add_subplot(gs[0,:])
# ax2 = fig_final.add_subplot(gs[1,:])
# ax3 = fig_final.add_subplot(gs[2,0])
# ax4 = fig_final.add_subplot(gs[2,1])
ax1 = plt.subplot2grid((3,2),(0,0), colspan=2)
ax2 = plt.subplot2grid((3,2),(1,0), colspan=2)
ax3 = plt.subplot2grid((3,2),(2,0))
ax4 = plt.subplot2grid((3,2),(2,1))


ax1.plot(beta**2)
ax1.set_ylabel(r'$\beta^2$')
ax1.set_xlabel('PMH iteration')

ax2.plot(sigma**2)
ax2.set_ylabel(r'$\sigma^2$')
ax2.set_xlabel('PMH iteration')

ax3.hist(beta[5000::]**2, bins=50, density=1)
ax3.set_xlabel(r'$\beta^2$')

ax4.hist(sigma[5000::]**2, bins=50, density=1)
ax4.set_xlabel(r'$\sigma^2$')

fig_final.tight_layout()


plt.show()