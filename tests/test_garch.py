import numpy as np
from scipy import stats
from quantlab.garch import garch_neg_loglik

def test_garch_reduces_to_normal_when_alpha_beta_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0.05, 1.2, 2000)
    mu, omega = x.mean(), x.var()
    mine = garch_neg_loglik([mu, omega, 0.0, 0.0], x)
    ref = -np.sum(stats.norm.logpdf(x, mu, np.sqrt(omega)))
    assert abs(mine - ref) < 1e-8