import numpy as np
from scipy.stats import norm

def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S/K)) + (r + 0.5*sigma**2) / sigma*np.sqrt(T)
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, r, sigma, T):
    d1 = (np.log(S/K)) + (r + 0.5*sigma**2) / sigma*np.sqrt(T)
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
