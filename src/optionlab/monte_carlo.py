"""Monte Carlo pricing of European options under geometric Brownian motion (GBM).

Idea: the option price is the average payoff at expiry, discounted back to
today, when the stock grows at the risk-free rate. We cannot compute that
average exactly here (pretend we have no formula), so we simulate many
possible final stock prices and average the payoffs.
"""
import numpy as np


def terminal_prices(S0, T, r, sigma, Z):
    """Stock price at expiry for each standard normal draw in Z.

    Under GBM the log of the stock price is normally distributed, so one
    random number per path takes us straight from today to expiry.
    """
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def payoff(ST, K, kind="call"):
    """What the option pays at expiry when the stock ends at ST."""
    if kind == "call":
        return np.maximum(ST - K, 0.0)
    if kind == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError("kind must be 'call' or 'put'")


def mc_price(S0, K, T, r, sigma, n_paths, kind="call", antithetic=False, seed=None):
    """Monte Carlo price with its uncertainty.

    Returns (price, standard_error, ci_low, ci_high), where the last two
    numbers are a 95% confidence interval: price -/+ 1.96 standard errors.

    antithetic=True uses each random draw Z twice, once as +Z and once as -Z.
    A path where the stock ends high is paired with one where it ends low,
    so the two payoffs partly cancel each other's noise. n_paths is the total
    number of simulated prices, so it is n_paths / 2 pairs.
    """
    rng = np.random.default_rng(seed)
    discount = np.exp(-r * T)

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")
        Z = rng.standard_normal(n_paths // 2)
        payoff_plus = payoff(terminal_prices(S0, T, r, sigma, Z), K, kind)
        payoff_minus = payoff(terminal_prices(S0, T, r, sigma, -Z), K, kind)
        # The two payoffs in a pair are NOT independent of each other, but
        # different pairs are. So one pair-average counts as one sample.
        samples = discount * 0.5 * (payoff_plus + payoff_minus)
    else:
        Z = rng.standard_normal(n_paths)
        samples = discount * payoff(terminal_prices(S0, T, r, sigma, Z), K, kind)

    estimate = samples.mean()
    standard_error = samples.std(ddof=1) / np.sqrt(len(samples))
    ci_low = estimate - 1.96 * standard_error
    ci_high = estimate + 1.96 * standard_error
    return estimate, standard_error, ci_low, ci_high
