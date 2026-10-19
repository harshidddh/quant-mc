import numpy as np
from arch import arch_model
from src.quantlab.data import load_log_returns
from src.quantlab.garch import fit_garch

x = 100 * load_log_returns().values

res = fit_garch(x)
mu, omega, alpha, beta = res.x
print("converged:", res.success)
print("mine : mu=%.4f omega=%.4f alpha=%.4f beta=%.4f  logL=%.2f" % (mu, omega, alpha, beta, -res.fun))
print("persistence alpha+beta:", alpha + beta)
print("long-run annualised vol (%):", np.sqrt(omega / (1 - alpha - beta) * 252))

ref = arch_model(x, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
p = ref.params
print("arch : mu=%.4f omega=%.4f alpha=%.4f beta=%.4f  logL=%.2f" % (p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"], ref.loglikelihood))