"""Black-Scholes prices and Greeks for European options (no dividends).

Inputs used everywhere in this module:
    S      stock price today
    K      strike price
    T      time to expiry in years (must be > 0)
    r      risk-free interest rate, continuously compounded (0.05 = 5%)
    sigma  volatility per year (0.20 = 20%)
    kind   "call" or "put"

Units of the Greeks:
    vega   per 1.00 change in sigma (divide by 100 for "per vol point")
    theta  per year                 (divide by 365 for "per calendar day")
    rho    per 1.00 change in r     (divide by 100 for "per 1% rate move")

Every function also works when S is a numpy array, which lets the hedging
simulation compute the delta of thousands of paths in one call.
"""
import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """The two numbers that appear in every Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def price(S, K, T, r, sigma, kind="call"):
    """Fair price of the option today."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if kind == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("kind must be 'call' or 'put'")


def delta(S, K, T, r, sigma, kind="call"):
    """Change in option price per 1.00 change in the stock price."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if kind == "call":
        return norm.cdf(d1)
    if kind == "put":
        return norm.cdf(d1) - 1.0
    raise ValueError("kind must be 'call' or 'put'")


def gamma(S, K, T, r, sigma):
    """Change in delta per 1.00 change in the stock price (same for call and put)."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """Change in option price per 1.00 change in volatility (same for call and put)."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, kind="call"):
    """Change in option price per year of time passing (usually negative)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    decay = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    if kind == "call":
        return decay - r * K * np.exp(-r * T) * norm.cdf(d2)
    if kind == "put":
        return decay + r * K * np.exp(-r * T) * norm.cdf(-d2)
    raise ValueError("kind must be 'call' or 'put'")


def rho(S, K, T, r, sigma, kind="call"):
    """Change in option price per 1.00 change in the interest rate."""
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if kind == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    if kind == "put":
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    raise ValueError("kind must be 'call' or 'put'")
