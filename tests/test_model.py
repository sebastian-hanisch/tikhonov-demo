"""Kern: jede Einheit an der 3 x 3-Mini-Instanz von Hand (tests/conftest.py) und an Eigenschaften der geschlossenen Formeln."""

import math

import numpy as np
import pytest

import tik_model as model
from inverse_generator import SplitMix64, grid, operator, phantom


def _parts(a, x, b):
    u, sv, vt = model.decompose(a)
    return u, sv, vt, vt @ x


def test_mini_filter_factors_by_hand(mini):
    a, x, b, lam = mini
    _, sv, _ = model.decompose(a)
    assert model.filter_factors(sv, lam) == pytest.approx([1.0 / 1.01, 0.25 / 0.26, 0.5], abs=1e-12)


def test_mini_solution_without_noise_by_hand(mini):
    a, x, b, lam = mini
    u, sv, vt, c = _parts(a, x, b)
    sol = model.solve(u, sv, vt, b[None, :], lam)[0]
    assert sol == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-6)
    assert float(np.linalg.norm(sol - x)) == pytest.approx(0.408888, abs=1e-6)
    assert float(model.rel_error(sol[None, :], x)[0]) == pytest.approx(0.408888 / math.sqrt(6.0), abs=1e-6)


def test_mini_bias_and_variance_by_hand(mini):
    a, x, b, lam = mini
    u, sv, vt, c = _parts(a, x, b)
    bias2, var = model.bias_variance(sv, c, np.zeros(3), 1.0, lam)
    assert math.sqrt(bias2) == pytest.approx(0.408888, abs=1e-6)
    assert var == pytest.approx(1.0 / 1.01 ** 2 + 0.25 / 0.26 ** 2 + 0.01 / 0.02 ** 2, abs=1e-12)
    assert var == pytest.approx(29.6785, abs=1e-4)
    # ohne Strafe: kein Bias, Varianz 1/1 + 1/0,25 + 1/0,01 = 105
    b0, v0 = model.bias_variance(sv, c, np.zeros(3), 1.0, 0.0)
    assert b0 == pytest.approx(0.0, abs=1e-20) and v0 == pytest.approx(105.0, abs=1e-9)


def test_mini_degrees_of_freedom_by_hand(mini):
    a, *_ = mini
    _, sv, _ = model.decompose(a)
    assert model.degrees_of_freedom(sv, 0.1) == pytest.approx(1.0 / 1.01 + 0.25 / 0.26 + 0.5, abs=1e-12)
    assert model.degrees_of_freedom(sv, 0.0) == pytest.approx(3.0)
    assert model.degrees_of_freedom(sv, 1e6) == pytest.approx(0.0, abs=1e-9)


def test_lambda_zero_is_the_direct_inversion(mini):
    a, x, b, _ = mini
    u, sv, vt, _ = _parts(a, x, b)
    assert model.solve(u, sv, vt, b[None, :], 0.0)[0] == pytest.approx(np.linalg.solve(a, b), abs=1e-12)


def test_huge_lambda_gives_the_zero_solution_and_error_one(mini):
    a, x, b, _ = mini
    u, sv, vt, c = _parts(a, x, b)
    assert np.allclose(model.solve(u, sv, vt, b[None, :], 1e8), 0.0, atol=1e-12)
    assert model.expected_error(sv, c, np.zeros(3), 0.01, 1e8, float(np.linalg.norm(x))) == pytest.approx(1.0, abs=1e-6)


def test_the_error_curve_has_one_valley_with_bias_rising_and_std_falling():
    n, w = 48, 0.03
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c = vt @ x
    curve = model.error_curve(sv, c, np.zeros(n), 3.4e-4, float(np.linalg.norm(x)), np.logspace(-8, 1, 91))
    assert np.all(np.diff(curve["bias"]) >= -1e-15) and np.all(np.diff(curve["std"]) <= 1e-15)
    i = int(np.argmin(curve["total"]))
    assert 0 < i < 90 and curve["total"][0] > 10 * curve["total"][i] and curve["total"][-1] > 5 * curve["total"][i]


def test_optimal_lambda_beats_every_grid_point():
    n, w = 48, 0.03
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c, xn = vt @ x, float(np.linalg.norm(x))
    lam, err = model.optimal_lambda(sv, c, np.zeros(n), 3.4e-4, xn)
    grid_err = [model.expected_error(sv, c, np.zeros(n), 3.4e-4, l, xn) for l in np.logspace(-9, 1, 400)]
    assert err <= min(grid_err) + 1e-12 and 1e-4 < lam < 1e-1
    assert err == pytest.approx(model.expected_error(sv, c, np.zeros(n), 3.4e-4, lam, xn), rel=1e-12)


def test_optimal_lambda_sits_at_the_grid_edge_when_no_penalty_is_needed():
    n, w = 24, 0.03
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    lam, _ = model.optimal_lambda(sv, vt @ x, np.zeros(n), 1e-9, float(np.linalg.norm(x)))
    assert lam < 1e-6


def test_valley_flatness_flag():
    n, w = 48, 0.03
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c, xn = vt @ x, float(np.linalg.norm(x))
    r = u.T @ (np.zeros(n))
    lam, _ = model.optimal_lambda(sv, c, r, 3.4e-4, xn)
    assert model.valley_is_flat(sv, c, r, 3.4e-4, xn, lam) is False
    lam2, _ = model.optimal_lambda(sv, c, r, 1e-12, xn)
    assert model.valley_is_flat(sv, c, r, 1e-12, xn, lam2, tol=0.5) is True


def test_the_closed_form_matches_a_small_monte_carlo():
    n, w, sigma, lam = 32, 0.04, 2e-4, 3e-3
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    e = SplitMix64(1).normal(n * 3000).reshape(3000, n) * sigma
    sol = model.solve(u, sv, vt, (a @ x)[None, :] + e, lam)
    sim = math.sqrt(float(np.mean(model.rel_error(sol, x) ** 2)))
    assert sim == pytest.approx(model.expected_error(sv, vt @ x, np.zeros(n), sigma, lam, float(np.linalg.norm(x))), rel=0.03)


def test_error_curve_total_is_the_euclidean_combination_of_bias_and_std():
    n = 24
    a = operator(n, 0.03)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    curve = model.error_curve(sv, vt @ x, np.zeros(n), 1e-3, float(np.linalg.norm(x)), np.logspace(-6, 0, 7))
    assert curve["total"] == pytest.approx(np.hypot(curve["bias"], curve["std"]), rel=1e-12)


def test_valley_is_flat_needs_both_sides_below_the_tolerance():
    n = 48
    a = operator(n, 0.03)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c, xn = vt @ x, float(np.linalg.norm(x))
    lam, _ = model.optimal_lambda(sv, c, np.zeros(n), 3.4e-4, xn)
    up = model.expected_error(sv, c, np.zeros(n), 3.4e-4, lam * 10, xn) - model.expected_error(sv, c, np.zeros(n), 3.4e-4, lam, xn)
    down = model.expected_error(sv, c, np.zeros(n), 3.4e-4, lam / 10, xn) - model.expected_error(sv, c, np.zeros(n), 3.4e-4, lam, xn)
    assert up < down                                       # das Tal ist links steiler
    tol = (up + down) / 2.0
    assert model.valley_is_flat(sv, c, np.zeros(n), 3.4e-4, xn, lam, tol=tol) is False
    assert model.valley_is_flat(sv, c, np.zeros(n), 3.4e-4, xn, lam, tol=down * 1.01) is True
