"""Tests for the delta-hedging simulation."""
import numpy as np
import pytest

from optionlab import black_scholes as bs
from optionlab.hedging import delta_hedge_pnl, simulate_paths

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
N = 20_000


def standard_error(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def test_paths_start_at_S0_and_have_right_shape():
    rng = np.random.default_rng(0)
    paths = simulate_paths(S0, T, r, sigma, n_steps=12, n_paths=50, rng=rng)
    assert paths.shape == (50, 13)
    assert np.all(paths[:, 0] == S0)
    assert np.all(paths > 0)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_correct_model_has_zero_mean_pnl(kind):
    # With the right volatility the premium exactly pays for the hedge on
    # average, so the mean P&L should be zero up to simulation noise.
    pnl = delta_hedge_pnl(S0, K, T, r, sigma, n_steps=52, n_paths=N, kind=kind, seed=1)
    assert abs(pnl.mean()) < 4 * standard_error(pnl)


def test_hedging_is_much_safer_than_not_hedging():
    pnl = delta_hedge_pnl(S0, K, T, r, sigma, n_steps=252, n_paths=N, seed=2)
    rng = np.random.default_rng(2)
    ST = simulate_paths(S0, T, r, sigma, 1, N, rng)[:, -1]
    unhedged = bs.price(S0, K, T, r, sigma) * np.exp(r * T) - np.maximum(ST - K, 0.0)
    assert pnl.std() < 0.05 * unhedged.std()


def test_error_shrinks_like_one_over_root_n():
    # 4 times as many rebalances should roughly halve the hedging error.
    weekly = delta_hedge_pnl(S0, K, T, r, sigma, n_steps=52, n_paths=N, seed=3)
    four_per_week = delta_hedge_pnl(S0, K, T, r, sigma, n_steps=208, n_paths=N, seed=3)
    assert weekly.std() / four_per_week.std() == pytest.approx(2.0, rel=0.10)


@pytest.mark.parametrize("sigma_hedge", [0.15, 0.25])
def test_wrong_volatility_shifts_mean_by_the_mispricing(sigma_hedge):
    # If we charge and hedge with sigma_hedge but the stock moves with sigma,
    # the average P&L equals the premium we charged minus the true price,
    # grown to expiry at the risk-free rate (exact when mu = r).
    pnl = delta_hedge_pnl(S0, K, T, r, sigma, n_steps=52, n_paths=N,
                          sigma_hedge=sigma_hedge, seed=4)
    mispricing = bs.price(S0, K, T, r, sigma_hedge) - bs.price(S0, K, T, r, sigma)
    expected = mispricing * np.exp(r * T)
    assert abs(pnl.mean() - expected) < 4 * standard_error(pnl)
    assert np.sign(pnl.mean()) == np.sign(sigma_hedge - sigma)
