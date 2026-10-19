import numpy as np
from quantlab.diagnostics import ljung_box

def test_ljung_box_accepts_white_noise_and_rejects_autocorrelated_series():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(3000)
    _, p_noise = ljung_box(noise, 20)
    assert p_noise > 0.001

    ar = np.zeros(3000)
    for t in range(1, 3000):
        ar[t] = 0.5 * ar[t - 1] + noise[t]
    _, p_ar = ljung_box(ar, 20)
    assert p_ar < 1e-6