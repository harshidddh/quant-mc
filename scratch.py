from src.quantlab.mc import mc_asian_call
from src.quantlab.bs import bs_call

exact = bs_call(100, 100, 0.05, 0.2, 1.0)

p, se, (lo, hi) = mc_asian_call(100, 100, 0.05, 0.2, 1.0, 252, 100_000)
print("Asian (252 steps):", p, "se", se, "CI", lo, hi)
print("European (BS)    :", exact)

p1, se1, _ = mc_asian_call(100, 100, 0.05, 0.2, 1.0, 1, 200_000)
print("Asian (1 step)   :", p1, "se", se1)
print("1-step within 4 se of BS:", abs(p1 - exact) < 4 * se1)