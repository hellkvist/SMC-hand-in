import numpy as np
from numpy import pi as PI
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mulnorm
from scipy.stats import norm
I2_CONST = np.identity(2)
COV_0 = 1/60 * I2_CONST
SIGMA_0 = 1/60**0.5
MEAN_0 = np.zeros(2)

def identifier(x1, x2, N):
    """
        returns 1 if both x1 and x2 are between 0 and 1
    """
    if N > 1:
        out = np.zeros(N)
        for n in range(N):
            if 0 <= x1[n] <= 1 and 0 <= x2[n] <= 1:
                out[n] = 1
            else:
                out[n] = 0
        return out
    else:
        if 0 <= x1 <= 1 and 0 <= x2 <= 1:
            return 1
        else:
            return 0


def pi_pdf(x1, x2, k, K, N):
    """ 
        Function for the annealing sequence pi_k. The 'pdf' is unnormalized
        returns the unnormalized pdf evaluated at the point (x1, x2) for annealing coefficient k
    """

    # return (identifier(x1, x2, N)*np.cos(PI*x1)**2*np.sin(3*PI*x2)**6*norm.pdf(x1, 0, SIGMA_0)*norm.pdf(x2, 0, SIGMA_0))**(k/K)*(norm.pdf(x1, 0, 1)*norm.pdf(x2, 0, 1))**(1-k/K)
    
    # this is good
    # return (identifier(x1, x2, N)*np.cos(PI*x1)**2*np.sin(3*PI*x2)**6)**(k/K)*norm.pdf(x1, 0, SIGMA_0)*norm.pdf(x2, 0, SIGMA_0)

    return identifier(x1, x2, N)*(np.cos(PI*x1)**2*np.sin(3*PI*x2)**6*norm.pdf(x1, 0, SIGMA_0)*norm.pdf(x2, 0, SIGMA_0))**(k/K)

    # this is not so good
    # return (identifier(x1, x2, N)*np.cos(PI*x1)**2*np.sin(3*PI*x2)**6*norm.pdf(x1, 0, SIGMA_0)*norm.pdf(x2, 0, SIGMA_0))**(k/K)


def sample_x0(N):
    # return np.random.normal(loc=0, scale=SIGMA_0, size=N)
    return np.random.uniform(size=N)


def metropolis_hastings(x1, x2, N, step_size_GRW, k, K):
    x1_prop = np.random.normal(loc=x1, scale=step_size_GRW, size=N)
    x2_prop = np.random.normal(loc=x2, scale=step_size_GRW, size=N)

    x1_out = np.copy(x1)
    x2_out = np.copy(x2)
    for n in range(N):
        u = np.random.uniform()
        x1_prop_n = x1_prop[n]
        x2_prop_n = x2_prop[n]
        x1_n = x1[n]
        x2_n = x2[n]

        alpha_num = pi_pdf(x1_prop_n, x2_prop_n, k, K, 1)*norm.pdf(x1_n, loc=x1_prop_n, scale=step_size_GRW)*norm.pdf(x2_n, loc=x2_prop_n, scale=step_size_GRW)
        alpha_den = pi_pdf(x1_n, x2_n, k, K, 1)*norm.pdf(x1_prop_n, loc=x1_n, scale=step_size_GRW)*norm.pdf(x2_prop_n, loc=x2_n, scale=step_size_GRW)

        alpha = np.min((1, alpha_num/alpha_den))
        if u <= alpha:
            x1_out[n] = x1_prop_n
            x2_out[n] = x2_prop_n
    return x1_out, x2_out


def smc_sampler(K, N, step_size_GRW):
    x1 = np.zeros((N, K))
    x2 = np.zeros((N, K))
    w = np.zeros((N, K))
    w_tilde_mat = np.zeros((N, K))
    N_eff = np.zeros(K)

    resamples = 0
    # sample N particles and set initial weights to 1/N
    x1[:, 0] = sample_x0(N)
    x2[:, 0] = sample_x0(N)
    w[:, 0] = 1/N*np.ones(N)
    w_tilde_mat[:, 0] = np.ones(N)

    # compute initial ESS
    N_eff[0] = 1/((w[:, 0]**2).sum())


    x1_prev = x1[:, 0]
    x2_prev = x2[:, 0]

    for k in range(1, K):
        if k % 10 == 0:
            print(k, '/', K)
        w_prev = w[:, k-1]
        # compute pi_k(x_{k-1})
        pi_k = pi_pdf(x1_prev, x2_prev, k, K, N)
        
        # compute pi_{k-1}(x_{k-1})
        pi_k_prev  = pi_pdf(x1_prev, x2_prev, k-1, K, N)

        # compute unnormalized weights and normalize
        w_tilde = w_prev * pi_k / pi_k_prev
        w_normed = w_tilde/w_tilde.sum()
        w[:, k] = w_normed
        w_tilde_mat[:, k] = w_tilde

        # compute ESS
        N_eff[k] = 1/((w_normed**2).sum())

        if N_eff[k] < 70:
            # resample
            idx = np.random.choice(N, size=N, replace=True, p=w_normed)
            resamples += 1
            resample_idx = k
            w[:, k] = 1/N*np.ones(N)
        else:
            # no resample
            idx = np.arange(N)
        # propagation
        x1_prev, x2_prev = metropolis_hastings(x1_prev[idx], x2_prev[idx], N, step_size_GRW, k, K)
        x1[:, k] = x1_prev
        x2[:, k] = x2_prev

    return x1, x2, w, resamples, N_eff, w_tilde_mat, resample_idx

K = 80
N = 100
step_size_GRW = 0.02

x1, x2, w, resamples, N_eff, w_tilde_mat, resample_idx = smc_sampler(K, N, step_size_GRW)

Z_0 = 1
Z_K_hat = Z_0 * np.prod(np.sum(w_tilde_mat, axis=0))

print('resampling rate:', resamples/K)
print('avg steps between resamples', resample_idx/resamples)
print('Z estimate:', Z_K_hat)


plt.figure(1)
plt.subplot(121)
plt.hist(x1[:, -1], bins=20, weights=w[:, -1], density=1)
plt.xlabel(r'$x_1$')
plt.subplot(122)
plt.hist(x2[:, -1], bins=20, weights=w[:, -1], density=1)
plt.xlabel(r'$x_2$')

plt.figure(2)
plt.plot(N_eff)
plt.ylabel(r'$N_{eff}$')
plt.xlabel('Iterations')
plt.hlines(70, 0, K-1)

plt.figure(3)
plt.subplot(221)
plt.plot(x1[:, 0], x2[:, 0], '.', label='k=1')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.grid()
plt.legend()
plt.subplot(222)
plt.plot(x1[:, 9], x2[:, 9], '.', label='k=10')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.grid()
plt.legend()
plt.subplot(223)
plt.plot(x1[:, 29], x2[:, 29], '.', label='k=30')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.grid()
plt.legend()
plt.subplot(224)
plt.plot(x1[:, 79], x2[:, 79], '.', label='k=80')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.grid()
plt.legend()
plt.show()