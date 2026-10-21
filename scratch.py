from src.quantlab.bs import bs_call, implied_vol

for sigma in (0.10, 0.25, 0.60):
    p = bs_call(100, 110, 0.05, sigma, 0.5)
    print(sigma, "->", p, "->", implied_vol(p, 100, 110, 0.05, 0.5))