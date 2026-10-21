import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S/K)) + (r + 0.5*sigma**2) / sigma*np.sqrt(T)
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_put(S, K, r, sigma, T):
    d1 = (np.log(S/K)) + (r + 0.5*sigma**2) / sigma*np.sqrt(T)
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def geo_asian_call(S0, K, r, sigma, T, m):
    mu_G = np.log(S0) + (r - 0.5 * sigma**2) * T * (m + 1) / (2 * m)
    var_G = sigma**2 * T * (m + 1) * (2 * m + 1) / (6 * m**2)
    sd_G = np.sqrt(var_G)
    d1 = (mu_G + var_G - np.log(K)) / sd_G
    d2 = d1 - sd_G
    return np.exp(-r * T) * (np.exp(mu_G + 0.5 * var_G) * norm.cdf(d1) - K * norm.cdf(d2))

def implied_vol(price, S, K, r, T):
    f = lambda sigma: bs_call(S, K, r, sigma, T) - price
    return brentq(f, 1e-4, 3.0)
