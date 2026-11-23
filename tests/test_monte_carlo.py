"""Tests for the Monte Carlo pricer.

Monte Carlo answers are random, so "equal to Black-Scholes" is the wrong
test. The right test is: the Black-Scholes price lies inside the Monte Carlo
confidence interval. Seeds are fixed so the tests give the same result on
every run.
"""
import pytest

from optionlab import black_scholes as bs
from optionlab.monte_carlo import mc_price

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
N = 200_000


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_black_scholes_inside_confidence_interval(kind, antithetic):
    exact = bs.price(S0, K, T, r, sigma, kind)
    estimate, se, ci_low, ci_high = mc_price(S0, K, T, r, sigma, N, kind, antithetic, seed=1)
    assert ci_low < exact < ci_high


@pytest.mark.parametrize("seed", range(20))
def test_error_is_never_far_outside_the_noise(seed):
    # A 95% interval misses about 1 time in 20 by design, so across many
    # seeds we use a wider band: 4 standard errors should essentially never fail.
    exact = bs.price(S0, K, T, r, sigma, "call")
    estimate, se, _, _ = mc_price(S0, K, T, r, sigma, 50_000, "call", seed=seed)
    assert abs(estimate - exact) < 4 * se


def test_antithetic_reduces_standard_error():
    _, se_plain, _, _ = mc_price(S0, K, T, r, sigma, N, "call", antithetic=False, seed=2)
    _, se_anti, _, _ = mc_price(S0, K, T, r, sigma, N, "call", antithetic=True, seed=2)
    assert se_anti < se_plain


def test_standard_error_shrinks_like_one_over_root_n():
    # 4 times the paths should roughly halve the standard error.
    _, se_small, _, _ = mc_price(S0, K, T, r, sigma, 50_000, "call", seed=3)
    _, se_large, _, _ = mc_price(S0, K, T, r, sigma, 200_000, "call", seed=3)
    assert se_small / se_large == pytest.approx(2.0, rel=0.05)


def test_antithetic_needs_even_paths():
    with pytest.raises(ValueError):
        mc_price(S0, K, T, r, sigma, 1001, antithetic=True)
