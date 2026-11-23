"""Tests for the closed-form prices and Greeks.

Each Greek is a derivative of the price, so we can check it without trusting
the formula: bump one input up and down by a small amount h, reprice, and
measure the slope. That is a (central) finite difference.
"""
import numpy as np
import pytest

from optionlab import black_scholes as bs

# (S, K, T, r, sigma): at-the-money, in-the-money call, out-of-the-money call
CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.20),
    (120.0, 100.0, 0.5, 0.03, 0.30),
    (80.0, 100.0, 2.0, 0.01, 0.15),
]
KINDS = ["call", "put"]


def test_known_value():
    # Standard textbook case: S=K=100, T=1, r=5%, sigma=20%.
    assert bs.price(100, 100, 1, 0.05, 0.20, "call") == pytest.approx(10.4506, abs=1e-4)
    assert bs.price(100, 100, 1, 0.05, 0.20, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
def test_put_call_parity(S, K, T, r, sigma):
    # call - put = S - K * exp(-rT), whatever the volatility is.
    call = bs.price(S, K, T, r, sigma, "call")
    put = bs.price(S, K, T, r, sigma, "put")
    assert call - put == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_delta(S, K, T, r, sigma, kind):
    h = 1e-4
    slope = (bs.price(S + h, K, T, r, sigma, kind) - bs.price(S - h, K, T, r, sigma, kind)) / (2 * h)
    assert bs.delta(S, K, T, r, sigma, kind) == pytest.approx(slope, abs=1e-6)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_gamma(S, K, T, r, sigma, kind):
    # Gamma is the slope of delta, so bump S inside delta.
    h = 1e-4
    slope = (bs.delta(S + h, K, T, r, sigma, kind) - bs.delta(S - h, K, T, r, sigma, kind)) / (2 * h)
    assert bs.gamma(S, K, T, r, sigma) == pytest.approx(slope, abs=1e-6)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_vega(S, K, T, r, sigma, kind):
    h = 1e-5
    slope = (bs.price(S, K, T, r, sigma + h, kind) - bs.price(S, K, T, r, sigma - h, kind)) / (2 * h)
    assert bs.vega(S, K, T, r, sigma) == pytest.approx(slope, abs=1e-5)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_theta(S, K, T, r, sigma, kind):
    # Theta is the change as time PASSES, i.e. as T gets smaller: minus sign.
    h = 1e-5
    slope = -(bs.price(S, K, T + h, r, sigma, kind) - bs.price(S, K, T - h, r, sigma, kind)) / (2 * h)
    assert bs.theta(S, K, T, r, sigma, kind) == pytest.approx(slope, abs=1e-5)


@pytest.mark.parametrize("S, K, T, r, sigma", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_rho(S, K, T, r, sigma, kind):
    h = 1e-5
    slope = (bs.price(S, K, T, r + h, sigma, kind) - bs.price(S, K, T, r - h, sigma, kind)) / (2 * h)
    assert bs.rho(S, K, T, r, sigma, kind) == pytest.approx(slope, abs=1e-5)


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        bs.price(100, 100, 1, 0.05, 0.2, "straddle")
