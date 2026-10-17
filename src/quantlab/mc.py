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

def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, rng):
    dt = T/n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_incr = (r-0.5*sigma**2)*dt + sigma*z*np.sqrt(dt)
    paths = S0*np.exp(np.cumsum(log_incr, axis=1))
    return np.hstack([np.full((n_paths, 1), S0), paths])

def mc_european_call_antithetic(S0, K, r, sigma, T, n, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    ST_up = S0 * np.exp(drift + vol * z)
    ST_down = S0 * np.exp(drift - vol * z)
    pair = 0.5 * (np.maximum(ST_up - K, 0) + np.maximum(ST_down - K, 0))
    disc = np.exp(-r * T) * pair
    price = disc.mean()
    se = disc.std(ddof=1) / np.sqrt(n)
    return price, se, (price - 1.96 * se, price + 1.96 * se)


