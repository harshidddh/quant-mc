from src.quantlab.mc import mc_european_call
from src.quantlab.bs import bs_call

price, se, (lo, hi) = mc_european_call(100, 100, 0.05, 0.2, 1.0, 100_000)
exact = bs_call(100, 100, 0.05, 0.2, 1.0)

print("MC price:", price)
print("std error:", se)
print("95% CI:", lo, hi)
print("Black-Scholes:", exact)
print("inside CI:", lo <= exact <= hi)