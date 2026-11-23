"""Delta-hedging simulation for a short European option.

We are the SELLER of one option. We receive the premium today and owe the
payoff at expiry. To avoid betting on the stock, we hold `delta` shares and
adjust that holding at fixed time steps. The function below returns our
profit or loss (P&L) at expiry on every simulated path.

Two volatilities appear on purpose:
    sigma_true   how the stock really moves in the simulation
    sigma_hedge  the volatility in OUR model: we use it for the premium we
                 charge and for every delta we compute
When they are equal the model is right. When they differ we are hedging
with a wrong model, which is the model-risk experiment.
"""
import numpy as np

from . import black_scholes as bs
from .monte_carlo import payoff


def simulate_paths(S0, T, mu, sigma, n_steps, n_paths, rng):
    """GBM stock paths. Returns an array of shape (n_paths, n_steps + 1).

    Column 0 is today's price S0, column n_steps is the price at expiry.
    mu is the stock's drift (expected return per year).
    """
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_steps = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_steps, axis=1))
    return paths


def delta_hedge_pnl(S0, K, T, r, sigma_true, n_steps, n_paths,
                    sigma_hedge=None, kind="call", mu=None, seed=None):
    """P&L at expiry of selling one option and delta-hedging it.

    Returns an array with one P&L number per path, in money at time T.
    n_steps is the number of rebalancing intervals (52 = weekly for T = 1).
    sigma_hedge defaults to sigma_true (correct model).
    mu defaults to r.
    """
    if sigma_hedge is None:
        sigma_hedge = sigma_true
    if mu is None:
        mu = r

    rng = np.random.default_rng(seed)
    paths = simulate_paths(S0, T, mu, sigma_true, n_steps, n_paths, rng)
    dt = T / n_steps
    growth = np.exp(r * dt)  # interest earned (or paid) on cash over one step

    # Today: receive the premium, buy `delta` shares, the rest sits in cash.
    # Cash can be negative, which means we borrowed to buy the shares.
    premium = bs.price(S0, K, T, r, sigma_hedge, kind)
    shares = np.full(n_paths, bs.delta(S0, K, T, r, sigma_hedge, kind))
    cash = premium - shares * S0

    # Each rebalancing date before expiry: cash earns interest, then we
    # recompute delta and buy or sell shares to match it.
    for i in range(1, n_steps):
        S = paths[:, i]
        time_left = T - i * dt
        cash = cash * growth
        new_shares = bs.delta(S, K, time_left, r, sigma_hedge, kind)
        cash = cash - (new_shares - shares) * S
        shares = new_shares

    # Expiry: last interest, sell the shares, pay the option holder.
    ST = paths[:, -1]
    cash = cash * growth
    return cash + shares * ST - payoff(ST, K, kind)
