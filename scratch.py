import numpy as np
from src.quantlab.bs import geo_asian_call
from src.quantlab.mc import simulate_gbm_paths, mc_asian_call, mc_asian_call_cv

S, K, r, sigma, T, m = 100, 100, 0.05, 0.2, 1.0, 252

# 1) Simulation vs closed form for the geometric option
rng = np.random.default_rng(0)
prices = simulate_gbm_paths(S, r, sigma, T, m, 100_000, rng)[:, 1:]
G = np.exp(np.log(prices).mean(axis=1))
X = np.exp(-r * T) * np.maximum(G - K, 0)
se_g = X.std(ddof=1) / np.sqrt(len(X))
exact_g = geo_asian_call(S, K, r, sigma, T, m)
print("geometric MC :", X.mean(), "se", se_g)
print("geometric exact:", exact_g)
print("within 4 se:", abs(X.mean() - exact_g) < 4 * se_g)

# 2) Control variate vs plain on the arithmetic Asian
p1, se1, _ = mc_asian_call(S, K, r, sigma, T, m, 100_000, seed=1)
p2, se2, (lo, hi) = mc_asian_call_cv(S, K, r, sigma, T, m, 100_000, seed=1)
print("plain Asian :", p1, "se", se1)
print("cv Asian    :", p2, "se", se2)
print("variance reduction factor:", (se1 / se2) ** 2)