from src.quantlab.mc import mc_european_call, mc_european_call_cv
from src.quantlab.bs import bs_call

exact = bs_call(100, 100, 0.05, 0.2, 1.0)
p1, se1, _ = mc_european_call(100, 100, 0.05, 0.2, 1.0, 200_000)
p2, se2, (lo, hi) = mc_european_call_cv(100, 100, 0.05, 0.2, 1.0, 200_000)

print("plain :", p1, "se", se1)
print("cv    :", p2, "se", se2)
print("variance reduction factor:", (se1 / se2) ** 2)
print("BS inside CV CI:", lo <= exact <= hi)