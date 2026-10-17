import numpy as np
from quantlab.bs import bs_call, bs_put
from quantlab.mc import mc_european_call
from quantlab.mc import simulate_gbm_paths
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

def test_paths_start_at_S0_and_final_mean_matches_theory():
    S0, r, sigma, T = 100, 0.05, 0.2, 1.0
    rng = np.random.default_rng(3)
    paths = simulate_gbm_paths(S0, r, sigma, T, 50, 20_000, rng)
    assert np.all(paths[:, 0] == S0)
    ST = paths[:, -1]
    se = ST.std(ddof=1) / np.sqrt(len(ST))
    assert abs(ST.mean() - S0 * np.exp(r * T)) < 4 * se