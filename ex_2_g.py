import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
T = 2000

x = np.zeros(T)
y = np.zeros(T)

x[0] = np.random.normal(0, 2**.5)
y[0] = np.random.normal(2*x[0], 0.1**.5)
for t in range(1, T):
    x[t] = np.random.normal(0.8*x[t-1], 0.5**.5)
    y[t] = np.random.normal(2*x[t], 0.1**.5)


## Kalman block ##
print('Kalman block')
A = 0.8
C = 2
Q = 0.5
R = 0.1
P = 2

x_kalman = np.zeros(T)
var_kalman = np.zeros(T)
x_kalman[0] = 0
var_kalman[0] = P
for t in range(1,T):
    P_pred = A*P*A + Q    
    K = P_pred*C*(C*P_pred*C + R)**-1
    
    P = P_pred - K*C*P_pred
    x_kalman[t] = A*x_kalman[t-1] + K*(y[t] - C*A*x_kalman[t-1])
    var_kalman[t] = P
mse_kalman = np.mean(np.abs(x_kalman - x))
## end of Kalman block ##

### FAPF block
print('\nFAPF block')
def fapf_q(y, x_, N):
    return norm.rvs(loc=1/42 * (20*y + 1.6*x_), scale=42**-0.5, size=N)

def fapf_q_pdf(x, y, x_):
    return norm.pdf(x, loc=1/42 * (20*y + 1.6*x_), scale=42**-0.5)

def fapf_v_pdf(y, x_):
    return norm.pdf(y, loc=1.6*x_, scale=2.1**0.5)

def fapf_y_from_x(y, x):
    return norm.pdf(y, loc=2*x, scale=0.1**0.5)

def systematic_resampling(U, v):
    N = len(v)
    v_cumsum = np.cumsum(v)
    a = np.zeros(N)
    i = 0
    for n in range(N):
        while U[i] < v_cumsum[n]:
            a[i] = n
            i += 1
            if i == N:
                return a
    return -1

N = 100 # particles in the FAPF

resamples = 0

mad_fapf_2_kalman = []
print(N)

x_fapf_hat = np.zeros(T)
x_fapf = np.zeros((N, T))
a = np.zeros((N, T)).astype(int)

# initialize x from q
x_fapf[:, 0] = fapf_q(y[0], 0, N)
w = 1/N * np.ones(N)

N_eff = np.zeros(T)
v = fapf_v_pdf(y[0], np.zeros(N))
v /= v.sum()
w = 1/N * np.ones(N)
N_eff[0] = 1/np.sum(w**2)
for t in range(1, T):
    # compute misadjustment multipliers v
    v = fapf_v_pdf(y[t], x_fapf[:, t-1])
    v /= v.sum()
    N_eff[t] = 1/np.sum(w**2)
    if N_eff[t] < 50: # resample
         # ancenstor indexes
        U = np.zeros(N)
        U0 = np.random.uniform(0, 1/N)
        for i in range(N):
            U[i] = i/N + U0
        # a[:, t] = np.random.choice(N, size=N, p=v)
        a[:, t] = systematic_resampling(U, v)
        resamples += 1
        w = 1/N*np.ones(N)
    else: # do not resample
        a[:, t] = np.arange(N)

    # propagation
    x_fapf[:, t] = fapf_q(y[t], x_fapf[a[:, t].astype(int), t-1], N) 
    x_fapf_hat[t] = np.mean(x_fapf[:, t])
    w_tilde = fapf_v_pdf(y[t], x_fapf[:, t-1])*w
    # w_tilde = fapf_y_from_x(y[t], x_fapf[:, t])*w
    w = w_tilde/w_tilde.sum()

mad_fapf_2_kalman.append(np.mean(np.abs(x_fapf_hat-x_kalman)))

print('MAD between FAPF and Kalman state: ', mad_fapf_2_kalman)
print('N.o. resamples: ', resamples)
## end of FAPF block

a = a.astype(int)
idx = np.arange(N)
plot_from = 100
plot_to = 0
plt.subplot(211)
for t in range(T-1, 0, -1):
    n = len(idx)
    t_mat = np.vstack((t*np.ones(n), (t-1)*np.ones(n)))
    ancest = a[a[idx, t], t-1]
    val_mat =  np.vstack((x_fapf[idx, t], x_fapf[ancest, t-1]))
    idx_mat = np.vstack((idx, ancest))
    if t <= plot_from and t >= plot_to:
        plt.plot(t_mat, val_mat, 'ko-', linewidth=.5, markersize=.5)
    idx = np.unique(ancest)
plt.xlabel('Iteration')
plt.ylabel('State')
plt.xlim(plot_to, plot_from)
plt.grid()
plt.tight_layout()

plt.subplot(212)
plt.grid()
plt.plot(N_eff)
plt.xlim(plot_to, plot_from)
plt.tight_layout()
plt.hlines(50, plot_to, plot_from)

print(np.mean(N_eff))
plt.show()