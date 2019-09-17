import numpy as np
import matplotlib.pyplot as plt

T = 2000

x = np.zeros(T)
y = np.zeros(T)

x[0] = np.random.normal(0, 2**.5)
y[0] = np.random.normal(2*x[0], 0.1**.5)

for t in range(1, T):
    x[t] = np.random.normal(0.8*x[t-1], 0.5**.5)
    y[t] = np.random.normal(2*x[t], 0.1**.5)


A = 0.8
C = 2
Q = 0.5
R = 0.1
P = 2

x_kalman = np.zeros(T)

x_kalman[0] = 0
for t in range(1,T):
    P_pred = A*P*A + Q    
    K = P_pred*C*(C*P_pred*C + R)**-1
    
    P = P_pred - K*C*P_pred
    x_kalman[t] = A*x_kalman[t-1] + K*(y[t] - C*x_kalman[t-1])

plt.plot(range(T-150,T), x_kalman[-150::], '-', label='Kalman')
plt.plot(range(T-150,T), x[-150::], '-',label='True states')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()
    