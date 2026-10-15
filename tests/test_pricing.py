import numpy as np
from quantlab.bs import bs_call, bs_put
from quantlab.mc import mc_european_call

def test_put_call_parity():
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    c = bs_call(S, K, r, sigma, T)
    p = bs_put(S, K, r, sigma, T)
    assert abs((c - p) - (S - K * np.exp(-r * T))) < 1e-10

def test_mc_call_agrees_with_black_scholes():
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    exact = bs_call(S, K, r, sigma, T)
    price, se, _ = mc_european_call(S, K, r, sigma, T, n=200_000, seed=1)
    assert abs(price - exact) < 4 * se