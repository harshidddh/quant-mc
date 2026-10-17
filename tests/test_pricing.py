import numpy as np
from quantlab.bs import bs_call, bs_put, geo_asian_call
from quantlab.mc import mc_european_call, mc_european_call_antithetic, simulate_gbm_paths, mc_european_call_cv, mc_asian_call, mc_asian_call_cv

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

def test_antithetic_agrees_with_bs_and_beats_plain():
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    exact = bs_call(S, K, r, sigma, T)
    p, se, _ = mc_european_call_antithetic(S, K, r, sigma, T, n=100_000, seed=2)
    _, se_plain, _ = mc_european_call(S, K, r, sigma, T, n=200_000, seed=2)
    assert abs(p - exact) < 4 * se
    assert se < se_plain

def test_control_variate_agrees_with_bs_and_beats_plain():
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    exact = bs_call(S, K, r, sigma, T)
    p, se, _ = mc_european_call_cv(S, K, r, sigma, T, n=200_000, seed=4)
    _, se_plain, _ = mc_european_call(S, K, r, sigma, T, n=200_000, seed=4)
    assert abs(p - exact) < 4 * se
    assert se < se_plain

def test_asian_one_step_equals_european_and_more_steps_is_cheaper():
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    exact = bs_call(S, K, r, sigma, T)
    p1, se1, _ = mc_asian_call(S, K, r, sigma, T, 1, 200_000, seed=5)
    assert abs(p1 - exact) < 4 * se1
    p, se, _ = mc_asian_call(S, K, r, sigma, T, 52, 50_000, seed=5)
    assert p + 4 * se < exact

def test_geo_asian_one_date_equals_black_scholes():
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    assert abs(geo_asian_call(S, K, r, sigma, T, 1) - bs_call(S, K, r, sigma, T)) < 1e-9

def test_geo_asian_mc_matches_closed_form():
    S, K, r, sigma, T, m = 100, 100, 0.05, 0.2, 1.0, 52
    rng = np.random.default_rng(6)
    prices = simulate_gbm_paths(S, r, sigma, T, m, 50_000, rng)[:, 1:]
    G = np.exp(np.log(prices).mean(axis=1))
    X = np.exp(-r * T) * np.maximum(G - K, 0)
    se = X.std(ddof=1) / np.sqrt(len(X))
    assert abs(X.mean() - geo_asian_call(S, K, r, sigma, T, m)) < 4 * se

def test_asian_cv_is_much_tighter_and_consistent_with_plain():
    S, K, r, sigma, T, m = 100, 100, 0.05, 0.2, 1.0, 52
    p, se, _ = mc_asian_call(S, K, r, sigma, T, m, 20_000, seed=7)
    p_cv, se_cv, _ = mc_asian_call_cv(S, K, r, sigma, T, m, 20_000, seed=7)
    assert se_cv < se / 10
    assert abs(p_cv - p) < 4 * se