import numpy as np
from quantlab.bs import bs_call, bs_put

def test_put_call_parity():
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    c = bs_call(S, K, r, sigma, T)
    p = bs_put(S, K, r, sigma, T)
    assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-10