import numpy as np
from scipy import stats
from quantlab.garch import garch_neg_loglik, fit_garch, simulate_garch, simulate_garch_rn_terminal

def test_garch_reduces_to_normal_when_alpha_beta_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 1.2, 2000)
    mu, omega = x.mean(), x.var()
    mine = garch_neg_loglik([mu, omega, 0.0, 0.0], x)
    ref = -np.sum(stats.norm.logpdf(x, mu, np.sqrt(omega)))
    assert abs(mine - ref) < 1e-8

def _simulate_garch(omega, alpha, beta, n, rng):
    x = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        x[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = omega + alpha * x[t] ** 2 + beta * s2
    return x

def test_fit_garch_recovers_known_parameters():
    rng = np.random.default_rng(0)
    x = _simulate_garch(0.05, 0.10, 0.85, 5000, rng)
    res = fit_garch(x)
    _, omega, alpha, beta = res.x
    assert res.success
    assert abs(alpha - 0.10) < 0.05
    assert abs(beta - 0.85) < 0.08
    assert abs(alpha + beta - 0.95) < 0.03

def test_simulate_garch_matches_long_run_variance():
    rng = np.random.default_rng(1)
    omega, alpha, beta = 0.05, 0.10, 0.85
    V = omega / (1 - alpha - beta)
    sim = simulate_garch(0.0, omega, alpha, beta, V, 500, 2000, rng)
    assert abs(sim.var() - V) / V < 0.05

def test_rn_terminal_is_martingale():
    rng = np.random.default_rng(2)
    omega, alpha, beta = 0.05, 0.10, 0.85
    V = omega / (1 - alpha - beta)
    ST = simulate_garch_rn_terminal(100, 0.05, omega, alpha, beta, V, 252, 50_000, rng)
    disc = np.exp(-0.05) * ST
    se = disc.std(ddof=1) / np.sqrt(len(disc))
    assert abs(disc.mean() - 100) < 4 * se
