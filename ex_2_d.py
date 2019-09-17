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

## Bootstrap PF block
print('\nBPF block')
N_vec = np.array([10, 50, 100, 2000, 5000])

mad_bpf_2_kalman = []
var_bpf = np.zeros((len(N_vec), T))
for N in N_vec:
    print(N)
    x_bpf_hat = np.zeros(T)

    x_bpf = np.random.normal(0, 2**.5, size=N)
    w_tilde = norm.pdf(y[0], loc=2*x_bpf, scale=0.1**.5)
    w = w_tilde/w_tilde.sum()
    x_bpf_hat[0] = (w*x_bpf).sum()
    for t in range(1, T):
        # resample  
        res_idx = np.random.choice(N, size=N, p=w)
        # propagate 
        x_bpf = np.random.normal(0.8*x_bpf[res_idx], 0.5**0.5)
        # weights
        w_tilde = norm.pdf(y[t], loc=2*x_bpf, scale=0.1**0.5)
        w = w_tilde/w_tilde.sum()
        # prediction of propagation
        prod_w_x = w*x_bpf
        x_bpf_hat[t] = prod_w_x.sum()
        var_bpf[N_vec==N, t] =  np.sum(w*x_bpf**2) - (prod_w_x.sum())**2
    mad_bpf_2_kalman.append( np.mean( np.abs(x_bpf_hat-x_kalman)) )

mean_var_bpf = np.mean(var_bpf, axis=1)
print('avg variance BPF:', mean_var_bpf)
print('avg kalman variance', np.mean(var_kalman))
print('MAD between BPF and Kalman State', mad_bpf_2_kalman)
## end of BPF block

### FAPF block
print('\nFAPF block')
def fapf_q(y, x_, N):
    return norm.rvs(loc=1/42 * (20*y + 1.6*x_), scale=42**-0.5, size=N)

def fapf_q_pdf(x, y, x_):
    return norm.pdf(x, loc=1/42 * (20*y + 1.6*x_), scale=42**-0.5)

def fapf_v(y, x_):
    return norm.pdf(y, loc=1.6*x_, scale=2.1**0.5)

N_vec = np.array([10, 50, 100, 2000, 5000])

mad_fapf_2_kalman = []
var_fapf = np.zeros((len(N_vec), T))
for N in N_vec:
    print(N)

    x_fapf_hat = np.zeros(T)
    x_fapf = np.zeros((N, T))
    a = np.zeros((N, T))

    # initialize x from q
    x_fapf[:, 0] = fapf_q(y[0], 0, N)

    for t in range(1, T):
        # compute misadjustment multipliers v
        v = fapf_v(y[t-1], x_fapf[:, t-1])
        v /= v.sum()

        # ancenstor indexes
        a[:, t] = np.random.choice(N, size=N, p=v)

        # propagation
        x_fapf[:, t] = fapf_q(y[t], x_fapf[a[:, t].astype(int), t-1], N) 
        x_fapf_hat[t] = np.mean(x_fapf[:, t])

    mad_fapf_2_kalman.append(np.mean(np.abs(x_fapf_hat-x_kalman)))

print('MAD between FAPF and Kalman state: ', mad_fapf_2_kalman)
## end of FAPF block


# plt.plot(var_bpf[0,:], label='N=10', linestyle='-')
# plt.plot(var_bpf[1,:], label='N=50', linestyle='-')
# plt.plot(var_bpf[2,:], label='N=100', linestyle='-')
# plt.plot(var_bpf[3,:], label='N=2000', linestyle='-')
# plt.plot(var_bpf[4,:], label='N=5000', linestyle='-')
# plt.plot(var_kalman, label='kalman', color='k', linestyle=':')
# plt.ylim(1e-3, 1)
# plt.xlim(1950, 2000)
# plt.xlabel('Iteration')
# plt.ylabel(r'Var($w_ix_i$)')
# plt.legend()
# plt.yscale('log')

plt.figure()
plt.plot(x_bpf_hat,'o', label='BPF trajectory')
plt.plot(x_fapf_hat,'o', label='FAPF trajectory')
plt.plot(x_kalman, 'o',label='kalman trajectory')
plt.plot(x, 'o',label='true')
plt.legend()

# print(mad_bpf_2_kalman)
# print(mse_kalman)

plt.show()
