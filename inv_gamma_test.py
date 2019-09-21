import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def log_invgamma(x, a, b):
    return np.log(b**a/gamma(a) * x**(-a-1)) - b/x

def log_invgamma_rvs(a, b, size):
    # b = np.min((100000, b))
    y = np.log(np.random.gamma(a, 1/b, size))
    return -y

def invgamma(a,b):
    return 1/np.random.gamma(a, 1/b)

y = invgamma(300, 400)
print(y)