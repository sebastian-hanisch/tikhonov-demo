"""Kern: Tikhonov-Lösung über Filterfaktoren, geschlossene Bias-Varianz-Zerlegung, Freiheitsgrade, bester Parameter. Nur numpy, kein Streamlit.

Jede Funktion ist eine kleine, einzeln prüfbare Einheit (DEMO-PLAYBOOK §1b): `tests/test_model.py` rechnet sie an der 3 x 3-Mini-Instanz von Hand und
`tests/test_oracle_tikhonov.py` gegen Normalgleichungen, erweiterte kleinste Quadrate, die Spur der Hat-Matrix, scipy und Monte Carlo.

Bezeichnungen: A = U diag(sv) V^T, x die Wahrheit, c = V^T x ihre Koeffizienten, b = A x + r + e die Daten mit Modellfehler r (bei Daten aus demselben Gitter
r = 0) und Rauschen e ~ N(0, sigma^2 I); r_coef = U^T r. Die Lösung zum Strafparameter lam ist x_lam = V diag(sv / (sv^2 + lam^2)) U^T b.
"""

from __future__ import annotations

import math

import numpy as np

_TINY = 1e-300
_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def decompose(a: np.ndarray):
    """Volle SVD A = U diag(sv) V^T (sv absteigend)."""
    u, sv, vt = np.linalg.svd(a)
    return u, sv, vt


def filter_factors(sv: np.ndarray, lam: float) -> np.ndarray:
    """f_i = sv_i^2 / (sv_i^2 + lam^2): wie viel der i-ten Komponente die Strafe durchlässt (1 ohne Strafe, 0 bei unendlicher)."""
    s2 = np.asarray(sv, dtype=float) ** 2
    return s2 / np.maximum(s2 + lam * lam, _TINY)


def solve(u: np.ndarray, sv: np.ndarray, vt: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    """Tikhonov-Lösung für b der Form (reps, n); lam = 0 ist die direkte Inversion von Stück 1."""
    gain = sv / np.maximum(sv ** 2 + lam * lam, _TINY)
    return ((np.atleast_2d(b) @ u) * gain) @ vt


def rel_error(xhat: np.ndarray, x: np.ndarray) -> np.ndarray:
    """||xhat - x|| / ||x|| je Zeile von xhat."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.sqrt(np.sum((np.atleast_2d(xhat) - x) ** 2, axis=-1) / np.sum(x ** 2))


def bias_variance(sv: np.ndarray, c: np.ndarray, r_coef: np.ndarray, sigma: float, lam: float) -> tuple[float, float]:
    """(Bias^2, Varianz) von x_lam als Summen der Quadrate, absolut: bias_i = -lam^2 c_i / (sv_i^2 + lam^2) + sv_i r_i / (sv_i^2 + lam^2),
    var_i = sigma^2 sv_i^2 / (sv_i^2 + lam^2)^2."""
    d = np.maximum(sv ** 2 + lam * lam, _TINY)
    bias = -(lam * lam) * c / d + sv * r_coef / d
    var = sigma ** 2 * sv ** 2 / d ** 2
    return float(np.sum(bias ** 2)), float(np.sum(var))


def expected_error(sv: np.ndarray, c: np.ndarray, r_coef: np.ndarray, sigma: float, lam: float, x_norm: float) -> float:
    """Wurzel von E||x_lam - x||^2, geteilt durch ||x||."""
    bias2, var = bias_variance(sv, c, r_coef, sigma, lam)
    return math.sqrt(bias2 + var) / x_norm


def error_curve(sv, c, r_coef, sigma, x_norm, lams) -> dict:
    """Bias, Streuung (Wurzel der Varianz) und Gesamtfehler, alle relativ zur Signalnorm, über die Parameter `lams`."""
    bias = np.empty(len(lams))
    std = np.empty(len(lams))
    for i, lam in enumerate(lams):
        b2, v = bias_variance(sv, c, r_coef, sigma, float(lam))
        bias[i], std[i] = math.sqrt(b2) / x_norm, math.sqrt(v) / x_norm
    return {"lam": np.asarray(lams, dtype=float), "bias": bias, "std": std, "total": np.sqrt(bias ** 2 + std ** 2)}


def degrees_of_freedom(sv: np.ndarray, lam: float) -> float:
    """Wirksame Zahl der Parameter: Summe der Filterfaktoren (Spur der Hat-Matrix)."""
    return float(np.sum(filter_factors(sv, lam)))


def optimal_lambda(sv, c, r_coef, sigma, x_norm, lo: float = 1e-9, hi: float = 10.0, points: int = 241) -> tuple[float, float]:
    """Bester Parameter und sein erwarteter Fehler (Wahrheit bekannt, also ein Orakel): Gitter über [lo, hi] und goldener Schnitt in der Umgebung des Minimums."""
    lams = np.logspace(math.log10(lo), math.log10(hi), points)
    errs = np.array([expected_error(sv, c, r_coef, sigma, float(l), x_norm) for l in lams])
    i = int(np.argmin(errs))
    a = math.log10(lams[max(i - 1, 0)])
    b = math.log10(lams[min(i + 1, points - 1)])
    f = lambda t: expected_error(sv, c, r_coef, sigma, 10.0 ** t, x_norm)  # noqa: E731
    x1, x2 = b - _PHI * (b - a), a + _PHI * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(60):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - _PHI * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + _PHI * (b - a)
            f2 = f(x2)
    t = 0.5 * (a + b)
    best = f(t)
    if errs[i] < best:               # Rand des Gitters (flaches Tal an der Grenze)
        return float(lams[i]), float(errs[i])
    return 10.0 ** t, float(best)


def valley_is_flat(sv, c, r_coef, sigma, x_norm, lam_star: float, tol: float = 0.005) -> bool:
    """True, wenn der Fehler bei dem 10-fachen und dem 0,1-fachen des besten Parameters weniger als `tol` (Anteil der Signalnorm) über dem Optimum liegt:
    dann ist der beste Parameter nicht mehr bestimmt."""
    e0 = expected_error(sv, c, r_coef, sigma, lam_star, x_norm)
    up = expected_error(sv, c, r_coef, sigma, lam_star * 10.0, x_norm) - e0
    down = expected_error(sv, c, r_coef, sigma, lam_star / 10.0, x_norm) - e0
    return bool(up < tol and down < tol)
