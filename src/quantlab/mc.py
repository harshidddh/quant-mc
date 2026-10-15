import numpy as np

def simulate_gbm_terminal(S0, r, sigma, T, n, rng):
    z = rng.standard_normal(n)
    return S0*np.exp((r-0.5*sigma**2)*T + sigma*z*np.sqrt(T))

def mc_european_call(S0, K, r, sigma, T, n, seed=0):
    rng = np.random.default_rng(seed)
    ST = simulate_gbm_terminal(S0, r, sigma, T, n, rng)
    payoff = np.maximum(ST - K, 0)
    disc = np.exp(-r * T) * payoff
    price = disc.mean()
    se = disc.std(ddof=1) / np.sqrt(n)
    return price, se, (price - 1.96 * se, price + 1.96 * se)


