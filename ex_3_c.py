import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def invgamma_rvs(a, b):
    # b = np.min((100000, b))
    y = np.random.gamma(a, 1/b)
    return 1/y

def bpf_propagate(x_prev, phi, sigma):
    return np.random.normal(loc=phi*x_prev, scale=sigma)

def log_weights(y, x, beta):
    std = beta*np.exp(x/2)
    return -0.5*np.log(2*np.pi) - np.log(std) - np.square(y)/(2 * std**2)

def bpf(y, x_in, N, phi, sigma, beta):

    T = len(y)

    log_N = np.log(N)

    # initialize particles
    x = np.zeros((N, T))
    x[0:N-1, 0] = np.random.normal(loc=0, scale=sigma, size=N-1)
    x[N-1, 0] = x_in[0]
    x_prop = x[:, 0]
    # ancestor idx matrix
    a_mat = np.zeros((N, T))
    # set initial weights 
    w = 1/N * np.ones(N)
    w_mat = np.zeros((N,T))
    w_mat[:, 0] = w
    for t in range(T):
        # resample with linear weights
        idx = np.random.choice(N, size=N, replace=True, p=w)

        # propagate particles
        x_prop = bpf_propagate(x_prop[idx], phi, sigma)
        x_prop[N-1] = x_in[t]

        # compute logweights
        log_w = log_weights(y[t], x_prop, beta)
        c = np.max(log_w)
        v = log_w - c

        # compute normalized linear weights
        exp_v = np.exp(v)
        w = exp_v/exp_v.sum()

        x[:, t] = x_prop
        a_mat[:, t] = idx
        w_mat[:, t]  = w
    a_mat[N-1,:] = N-1

    # draw one of the final particles to choose its ancestor path as output
    b = np.random.choice(N, size=1, replace=1, p=w)
    # find the path
    a_mat = a_mat.astype(int)
    a_vec = np.zeros(T).astype(int)
    path = np.zeros(T)
    a_vec[-1] = a_mat[b, -1]
    for t in range(T-2, -1, -1):
        a_vec[t] = a_mat[a_vec[t+1], t]

    for t in range(T):
        path[t] = w_mat[a_vec[t], t]*x[a_vec[t], t]

    return x, path

def gibbs(M, N, y, x_in, beta_0, sigma_0, phi):
    T = len(y)
    beta2 = np.zeros(M)
    sigma2 = np.zeros(M)
    
    beta2[0] = beta_0**2
    sigma2[0] = sigma_0**2
    x = x_in
    for m in range(1, M):
        if m % 100 == 0:
            print(m, '/', M)
        # draw sigma and beta from their inverse gamma distributions
        a = 0.01 + T/2
        b_s = 0.01 + 0.5 * np.sum((x[1:T] - phi*x[0:T-1])**2)
        sigma2_new = invgamma_rvs(a, b_s)

        b_b = 0.01 + 0.5*np.sum(np.exp(-x[1:T]) * y[1:T]**2)
        beta2_new = invgamma_rvs(a, b_b)

        _, x = bpf(y, x, N, phi, sigma2_new**.5, beta2_new**.5)

        sigma2[m] = sigma2_new
        beta2[m] = beta2_new
    return sigma2, beta2

phi = 0.985
sigma_0 = 0.16
beta_0 = 0.7
M = 1000
N = 100

# y = np.genfromtxt('Hand-in/OMXLogReturns.csv', delimiter=',')
y = np.genfromtxt('OMXLogReturns.csv', delimiter=',')
T = len(y)
x_in = np.random.normal(0, 1, size=T)

sigma2, beta2 = gibbs(M, N, y, x_in, beta_0, sigma_0, phi)

plt.figure(1)
plt.subplot(121)
plt.hist(beta2, bins=50)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma2, bins=50)
plt.xlabel(r'$\sigma^2$')

plt.figure(2)
plt.subplot(121)
plt.hist(beta2[0:1000], bins=50)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma2[0:1000], bins=50)
plt.xlabel(r'$\sigma^2$')

plt.figure(3)
plt.subplot(121)
plt.hist(beta2[100::], bins=50)
plt.xlabel(r'$\beta^2$')
plt.subplot(122)
plt.hist(sigma2[100::], bins=50)
plt.xlabel(r'$\sigma^2$')

plt.figure(6)
plt.subplot(211)
plt.plot(beta2)
plt.xlabel('PMH iteration')
plt.ylabel(r'$\beta^2$')

plt.subplot(212)
plt.plot(sigma2)
plt.ylabel(r'$\sigma^2$')
plt.xlabel('PMH iteration')
plt.show()

# x, path = bpf(y, x_in, N, phi, sigma, beta)

# plt.plot(x.T, 'k.', c='lightgray')
# plt.plot(path, label='path', lw=2)
# plt.plot(x_in, label='x_in', lw=1)
# plt.legend()
# plt.show()