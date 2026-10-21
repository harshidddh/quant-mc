import numpy as np
from scipy.optimize import minimize

def garch_variance(omega, alpha, beta, eps):
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = eps.var()
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2

def garch_neg_loglik(params, x):
    mu, omega, alpha, beta = params
    eps = x - mu
    sigma2 = garch_variance(omega, alpha, beta, eps)
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)

def fit_garch(x):
    start = [x.mean(), x.var() * 0.02, 0.08, 0.90]
    bounds = [(-1, 1), (1e-6, None), (0, 1), (0, 1)]
    cons = {"type": "ineq", "fun": lambda p: 0.9999 - p[2] - p[3]}
    return minimize(garch_neg_loglik, start, args=(x,),
                    method="SLSQP", bounds=bounds, constraints=cons)

def simulate_garch(mu, omega, alpha, beta, sigma2_0, n_steps, n_paths, rng):
    returns = np.empty((n_paths, n_steps))
    sigma2 = np.full(n_paths, sigma2_0)
    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        eps = np.sqrt(sigma2) * z
        returns[:, t] = mu + eps
        sigma2 = omega + alpha * eps**2 + beta * sigma2
    return returns

def simulate_garch_rn_terminal(S0, r, omega, alpha, beta, sigma2_0, n_steps, n_paths, rng):
    r_d = 100 * r / 252
    total = np.zeros(n_paths)
    sigma2 = np.full(n_paths, sigma2_0)
    for t in range(n_steps):
        z = rng.standard_normal(n_paths)
        eps = np.sqrt(sigma2) * z
        total += r_d - 0.5 * sigma2 / 100 + eps
        sigma2 = omega + alpha * eps**2 + beta * sigma2
    return S0 * np.exp(total / 100)