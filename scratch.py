import numpy as np
from scipy import stats
from src.quantlab.data import load_log_returns
from src.quantlab.garch import garch_variance, garch_neg_loglik

R = load_log_returns()
x = 100 * R.values

# Test 1: with alpha = beta = 0 the variance is constant, so we must match a plain normal
mu, omega = x.mean(), x.var()
mine = garch_neg_loglik([mu, omega, 0.0, 0.0], x)
ref = -np.sum(stats.norm.logpdf(x, mu, np.sqrt(omega)))
print("mine:", mine)
print("scipy:", ref)

# Test 2: with typical GARCH parameters the variance path must stay positive
eps = x - x.mean()
s2 = garch_variance(0.02, 0.1, 0.85, eps)
print("all positive:", np.all(s2 > 0))
print("highest-variance day:", R.index[np.argmax(s2)].date())