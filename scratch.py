import numpy as np
from src.quantlab.mc import simulate_gbm_paths
from src.quantlab.mc import mc_european_call
from src.quantlab.bs import bs_call

price, se, (lo, hi) = mc_european_call(100, 100, 0.05, 0.2, 1.0, 100_000)
exact = bs_call(100, 100, 0.05, 0.2, 1.0)

print("MC price:", price)
print("std error:", se)
print("95% CI:", lo, hi)
print("Black-Scholes:", exact)
print("inside CI:", lo <= exact <= hi)



rng = np.random.default_rng(42)
paths = simulate_gbm_paths(100, 0.05, 0.2, 1.0, 252, 100_000, rng)

print("shape:", paths.shape)
print("first column all S0:", np.all(paths[:, 0] == 100))
print("mean of final prices:", paths[:, -1].mean())
print("theory:", 100 * np.exp(0.05))