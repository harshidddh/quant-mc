import numpy as np
from scipy.stats import chi2

def acf(x, nlags=30):
    x = np.asarray(x) - np.mean(x)
    denom = np.sum(x**2)
    return np.array([np.sum(x[:-k] * x[k:]) / denom for k in range(1, nlags + 1)])

def ljung_box(x, nlags=20):
    n = len(x)
    rho = acf(x, nlags)
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, nlags + 1)))
    return q, chi2.sf(q, nlags)