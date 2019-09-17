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

plt.subplot(121)
plt.plot(y)
plt.subplot(122)
plt.hist(y, bins=100, density=1)
plt.show()