"""Orakel (DEMO-PLAYBOOK §12): Kernroutinen gegen einen anderen Rechenweg (Normalgleichungen, erweiterte kleinste Quadrate, Spur der Hat-Matrix, scipy, Monte Carlo).

Jeder Test läuft in unter 10 s. Fehlt scipy im Test-venv, überspringen sich die Tests (pytest.importorskip), sie raten nie."""

import math

import numpy as np
import pytest

import tik_model as model
from inverse_generator import SplitMix64, blurred_exact, draw_data, grid, noise_sigma, operator, phantom

scipy = pytest.importorskip("scipy")
from scipy import optimize  # noqa: E402


@pytest.mark.parametrize("n,w,lam", [(24, 0.02, 1e-3), (48, 0.03, 1e-2), (48, 0.03, 1e-5), (64, 0.03, 1e-1), (96, 0.04, 1.0)])
def test_filter_form_matches_the_normal_equations_and_the_augmented_least_squares(n, w, lam):
    a = operator(n, w)
    b = draw_data(n, w, 1e-3, SplitMix64(n), crime=False)[0]
    u, sv, vt = model.decompose(a)
    ours = model.solve(u, sv, vt, b[None, :], lam)[0]
    normal = np.linalg.solve(a.T @ a + lam ** 2 * np.eye(n), a.T @ b)
    augmented = np.linalg.lstsq(np.vstack([a, lam * np.eye(n)]), np.concatenate([b, np.zeros(n)]), rcond=None)[0]
    assert np.linalg.norm(ours - augmented) / np.linalg.norm(ours) < 1e-8
    assert np.linalg.norm(ours - normal) / np.linalg.norm(ours) < 1e-5


@pytest.mark.parametrize("n,w,lam", [(32, 0.03, 1e-3), (48, 0.03, 1e-2), (48, 0.04, 1e-1)])
def test_degrees_of_freedom_match_the_trace_of_the_hat_matrix(n, w, lam):
    a = operator(n, w)
    hat = a @ np.linalg.solve(a.T @ a + lam ** 2 * np.eye(n), a.T)
    _, sv, _ = model.decompose(a)
    assert model.degrees_of_freedom(sv, lam) == pytest.approx(float(np.trace(hat)), rel=1e-6)


@pytest.mark.parametrize("honest", [False, True])
def test_expected_error_matches_a_monte_carlo_with_and_without_model_error(honest):
    n, w, noise, lam = 48, 0.03, 1e-3, 5e-3
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    sigma = noise_sigma(n, w, noise)
    r = u.T @ (blurred_exact(grid(n), w) - a @ x) if honest else np.zeros(n)
    b = draw_data(n, w, noise, SplitMix64(11), crime=not honest, reps=3000)
    sim = math.sqrt(float(np.mean(model.rel_error(model.solve(u, sv, vt, b, lam), x) ** 2)))
    assert sim == pytest.approx(model.expected_error(sv, vt @ x, r, sigma, lam, float(np.linalg.norm(x))), rel=0.02)


@pytest.mark.parametrize("noise", [1e-1, 1e-2, 1e-3, 1e-4])
def test_optimal_lambda_matches_scipy_bounded_minimization(noise):
    n, w = 48, 0.03
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c, xn = vt @ x, float(np.linalg.norm(x))
    r = u.T @ (blurred_exact(grid(n), w) - a @ x)
    sigma = noise_sigma(n, w, noise)
    ours_l, ours_e = model.optimal_lambda(sv, c, r, sigma, xn)
    res = optimize.minimize_scalar(lambda t: model.expected_error(sv, c, r, sigma, 10.0 ** t, xn), bounds=(-9.0, 1.0), method="bounded", options={"xatol": 1e-10})
    assert ours_e == pytest.approx(float(res.fun), rel=1e-4)
    assert ours_e <= float(res.fun) + 1e-9


def test_both_data_kinds_share_the_bias_variance_split_at_zero_model_error():
    n, w, lam, sigma = 32, 0.04, 1e-3, 1e-4
    a = operator(n, w)
    x = phantom(grid(n))
    u, sv, vt = model.decompose(a)
    c = vt @ x
    bias2, var = model.bias_variance(sv, c, np.zeros(n), sigma, lam)
    # Bias als ||E x_lam - x||: Lösung ohne Rauschen gegen die Formel
    sol = model.solve(u, sv, vt, (a @ x)[None, :], lam)[0]
    assert float(np.sum((sol - x) ** 2)) == pytest.approx(bias2, rel=1e-8)
    # Varianz: Spur der Kovarianzmatrix des Schätzers sigma^2 * G G^T mit G = (A^T A + lam^2 I)^-1 A^T
    g = np.linalg.solve(a.T @ a + lam ** 2 * np.eye(n), a.T)
    assert var == pytest.approx(sigma ** 2 * float(np.trace(g @ g.T)), rel=1e-6)
