import numpy as np
from scipy import stats
from src.quantlab.data import load_log_returns
from src.quantlab.garch import fit_garch, simulate_garch
from src.quantlab.diagnostics import acf

x = 100 * load_log_returns().values
mu, omega, alpha, beta = fit_garch(x).x
V = omega / (1 - alpha - beta)

rng = np.random.default_rng(0)
sim = simulate_garch(mu, omega, alpha, beta, V, 1000, 500, rng)
eps = sim - mu

print("variance: simulated", eps.var(), "  long-run V", V)
print("excess kurtosis: simulated", stats.kurtosis(eps.ravel()), "  real data", stats.kurtosis(x))
print("lag-1 acf of squared returns: simulated", np.mean([acf(p**2, 1)[0] for p in eps[:100]]), "  real data 0.183")