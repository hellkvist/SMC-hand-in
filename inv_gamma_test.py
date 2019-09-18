import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def log_invgamma(x, a, b):
    return np.log(b**a/gamma(a) * x**(-a-1)) - b/x

x = np.linspace(0, .1, 1000000)

pdf = np.exp(log_invgamma(x, 0.01, 0.01))

plt.plot(x, pdf)
plt.show()
