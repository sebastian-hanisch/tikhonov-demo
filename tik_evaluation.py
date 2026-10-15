"""Auswertung: Kennzahlen der Instanz, die vier Experimente über feste Ströme und die Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tik_constants as C
import tik_model as model
from inverse_generator import SplitMix64, blurred_exact, draw_data, grid, noise_sigma, operator, phantom

_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,5·10⁹."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def lam_text(lam: float) -> str:
    """Strafparameter: ab 0,01 mit Dezimalstellen, darunter als Zehnerpotenz."""
    if lam == 0.0:
        return "0"
    return de(lam, 3 if lam >= 0.1 else 4) if lam >= 0.01 else sci(lam)


def err_pct(err: float) -> str:
    """Fehler als Prozent der Signalnorm: ab 10 Prozent ohne, ab 1 Prozent mit einer, ab 0,01 Prozent mit zwei Stellen, darunter als Zehnerpotenz;
    ab dem Zehnfachen der Signalnorm als Vielfaches."""
    if not math.isfinite(err):
        return "unendlich"
    if err >= 10.0:
        return f"{sci(err)}-fach"
    x = 100.0 * err
    if x >= 99.5:
        return f"{de(x, 0)} %"
    if x >= 0.95:
        return f"{de(x, 1)} %"
    if x >= 0.01:
        return f"{de(x, 2)} %"
    return f"{sci(x)} %"


def rel_text(err: float) -> str:
    """Fehler als Anteil der Signalnorm in Worten."""
    if not math.isfinite(err):
        return "unendlich viel"
    if err >= 10.0:
        return f"das {sci(err)}-Fache der Signalnorm"
    return f"{err_pct(err)} der Signalnorm"


def noise_label(v: float) -> str:
    """Rauschstufe als Prozent des Effektivwerts, z. B. 0,1 %."""
    return de(100.0 * v, 4).rstrip("0").rstrip(",") + " %" if v > 0.0 else "0 %"


# ----------------------------------------------------------------------------- Instanz

@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    w: float = C.DEFAULT_W
    noise: float = C.DEFAULT_NOISE
    data: str = "ehrlich"
    loglam: float = C.DEFAULT_LOGLAM
    seed: int = C.DEFAULT_SEED


@dataclass
class Analysis:
    settings: Settings
    s: np.ndarray            # Gitterpunkte
    x: np.ndarray            # Wahrheit
    g: np.ndarray            # exakte unscharfe Daten
    sv: np.ndarray
    cond: float
    sigma: float
    lam: float
    b: np.ndarray            # Messwerte der gewählten Art
    x_naive: np.ndarray      # direkte Inversion (lam = 0)
    x_tik: np.ndarray        # Tikhonov-Lösung
    err_naive: float
    err_tik: float
    expected: float          # erwarteter Fehler zum gewählten lam (geschlossen, gewählte Datenart)
    bias: float              # Bias relativ zur Signalnorm
    std: float               # Streuung relativ zur Signalnorm
    df: float                # Freiheitsgrade
    f: np.ndarray            # Filterfaktoren
    kstar: int
    lam_star: float          # bester Parameter der gewählten Datenart (Orakel)
    err_star: float
    flat: bool               # das Tal ist flach: lam_star nicht bestimmt
    lam_star_crime: float
    err_star_crime: float    # der im Test mit demselben Gitter gemeldete beste Fehler
    err_honest_at_lam: float # erwarteter Fehler zum gewählten lam mit ehrlichen Daten
    err_crime_at_lam: float  # und mit Daten aus demselben Gitter
    valley: dict             # Fehlertal über lam (gewählte Datenart)
    x_norm: float
    singular: bool


def _parts(n: int, w: float):
    a = operator(n, w)
    u, sv, vt = model.decompose(a)
    x = phantom(grid(n))
    g = blurred_exact(grid(n), w)
    c = vt @ x
    r_coef = u.T @ (g - a @ x)
    return a, u, sv, vt, x, g, c, r_coef


def analyse(settings: Settings) -> Analysis:
    st = settings
    a, u, sv, vt, x, g, c, r_coef = _parts(st.n, st.w)
    sigma = noise_sigma(st.n, st.w, st.noise)
    x_norm = float(np.linalg.norm(x))
    use_crime = st.data == "verbrechen"
    b = draw_data(st.n, st.w, st.noise, SplitMix64(st.seed), crime=use_crime)
    r_use = np.zeros(st.n) if use_crime else r_coef
    lam = 10.0 ** st.loglam
    xt = model.solve(u, sv, vt, b, lam)[0]
    xn = model.solve(u, sv, vt, b, 0.0)[0]
    bias2, var = model.bias_variance(sv, c, r_use, sigma, lam)
    lam_star, err_star = model.optimal_lambda(sv, c, r_use, sigma, x_norm)
    lam_c, err_c = model.optimal_lambda(sv, c, np.zeros(st.n), sigma, x_norm)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    lo, hi, pts = C.LAM_GRID
    lams = np.logspace(math.log10(lo), math.log10(hi), pts)
    return Analysis(
        settings=st, s=grid(st.n), x=x, g=g, sv=sv, cond=cond, sigma=sigma, lam=lam, b=b[0], x_naive=xn, x_tik=xt,
        err_naive=float(model.rel_error(xn, x)[0]), err_tik=float(model.rel_error(xt, x)[0]),
        expected=math.sqrt(bias2 + var) / x_norm, bias=math.sqrt(bias2) / x_norm, std=math.sqrt(var) / x_norm,
        df=model.degrees_of_freedom(sv, lam), f=model.filter_factors(sv, lam), kstar=int(np.sum(np.abs(u.T @ g) > sigma)),
        lam_star=lam_star, err_star=err_star, flat=model.valley_is_flat(sv, c, r_use, sigma, x_norm, lam_star, C.FLAT_TOL),
        lam_star_crime=lam_c, err_star_crime=err_c,
        err_honest_at_lam=model.expected_error(sv, c, r_coef, sigma, lam, x_norm),
        err_crime_at_lam=model.expected_error(sv, c, np.zeros(st.n), sigma, lam, x_norm),
        valley=model.error_curve(sv, c, r_use, sigma, x_norm, lams), x_norm=x_norm, singular=bool(cond >= C.SINGULAR_COND),
    )


def standard_sigma() -> float:
    """Absolutes Rauschen der Standardinstanz: bleibt in Experiment 2 fest, wenn das Gitter wechselt."""
    return noise_sigma(C.DEFAULT_N, C.DEFAULT_W, C.DEFAULT_NOISE)


# ----------------------------------------------------------------------------- Experiment 1: das Tal und seine Nachsicht

def valley_experiment(factors=C.EXP_FACTORS, reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    """Fehler bei Vielfachen des besten Parameters (ehrliche Daten): geschlossen und Simulation über feste Ströme, dazu der Bias-Anteil."""
    a, u, sv, vt, x, g, c, r_coef = _parts(n, w)
    sigma = noise_sigma(n, w, noise)
    x_norm = float(np.linalg.norm(x))
    lam_star, err_star = model.optimal_lambda(sv, c, r_coef, sigma, x_norm)
    b = draw_data(n, w, noise, SplitMix64(C.SEED_BASE + 1), crime=False, reps=reps)
    rows = []
    for fct in factors:
        lam = lam_star * fct
        b2, v = model.bias_variance(sv, c, r_coef, sigma, lam)
        sim = float(np.sqrt(np.mean(model.rel_error(model.solve(u, sv, vt, b, lam), x) ** 2)))
        rows.append({"factor": fct, "lam": lam, "closed": math.sqrt(b2 + v) / x_norm, "sim": sim, "ratio": math.sqrt(b2 + v) / x_norm / err_star,
                     "bias_share": b2 / (b2 + v), "df": model.degrees_of_freedom(sv, lam)})
    naive = model.expected_error(sv, c, r_coef, sigma, 0.0, x_norm)
    nothing = model.expected_error(sv, c, r_coef, sigma, 1e3, x_norm)
    return {"rows": rows, "lam_star": lam_star, "err_star": err_star, "naive": naive, "zero_solution": nothing, "n": n, "reps": reps}


# ----------------------------------------------------------------------------- Experiment 2: feineres Gitter schadet nicht mehr

def grid_experiment(ns=C.EXP_N_SWEEP, w: float = C.DEFAULT_W, sigma: float | None = None) -> list[dict]:
    """Erwarteter Fehler der direkten Inversion und der Tikhonov-Lösung beim besten Parameter, mit ehrlichen Daten und mit Daten aus demselben Gitter."""
    sigma = standard_sigma() if sigma is None else sigma
    rows = []
    for n in ns:
        a, u, sv, vt, x, g, c, r_coef = _parts(n, w)
        x_norm = float(np.linalg.norm(x))
        lam_h, err_h = model.optimal_lambda(sv, c, r_coef, sigma, x_norm)
        lam_c, err_c = model.optimal_lambda(sv, c, np.zeros(n), sigma, x_norm)
        rows.append({
            "n": n, "cond": float(sv[0] / sv[-1]), "direct": model.expected_error(sv, c, r_coef, sigma, 0.0, x_norm),
            "lam_honest": lam_h, "tik_honest": err_h, "lam_crime": lam_c, "tik_crime": err_c,
        })
    return rows


# ----------------------------------------------------------------------------- Experiment 3: weniger Rauschen hilft bis zum Boden

def noise_experiment(levels=C.EXP_NOISE_LEVELS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W) -> list[dict]:
    """Bester Fehler nach Rauschstufe: ehrliche Daten gegen Daten aus demselben Gitter, dazu der Parameter aus dem Test mit demselben Gitter auf ehrliche Daten."""
    a, u, sv, vt, x, g, c, r_coef = _parts(n, w)
    x_norm = float(np.linalg.norm(x))
    rows = []
    for lv in levels:
        sg = noise_sigma(n, w, lv)
        lam_h, err_h = model.optimal_lambda(sv, c, r_coef, sg, x_norm)
        lam_c, err_c = model.optimal_lambda(sv, c, np.zeros(n), sg, x_norm)
        rows.append({
            "noise": lv, "sigma": sg, "lam_honest": lam_h, "err_honest": err_h, "flat": model.valley_is_flat(sv, c, r_coef, sg, x_norm, lam_h, C.FLAT_TOL),
            "lam_crime": lam_c, "err_crime": err_c, "err_crime_lam_on_honest": model.expected_error(sv, c, r_coef, sg, lam_c, x_norm),
        })
    return rows


# ----------------------------------------------------------------------------- Experiment 4: was bleibt an den Kanten

def edge_experiment(reps: int = C.EXP_REPS, n: int = C.DEFAULT_N, w: float = C.DEFAULT_W, noise: float = C.DEFAULT_NOISE) -> dict:
    """Mittlere Tikhonov-Lösung und Streuung über feste Ströme beim besten Parameter (ehrliche Daten): Spitzen, Kastenmitte, Kante, Punktfehler je Bereich."""
    a, u, sv, vt, x, g, c, r_coef = _parts(n, w)
    sigma = noise_sigma(n, w, noise)
    x_norm = float(np.linalg.norm(x))
    lam, err = model.optimal_lambda(sv, c, r_coef, sigma, x_norm)
    b = draw_data(n, w, noise, SplitMix64(C.SEED_BASE + 4), crime=False, reps=reps)
    sol = model.solve(u, sv, vt, b, lam)
    mean, sd = sol.mean(axis=0), sol.std(axis=0)
    s = grid(n)
    box = (s >= 0.72) & (s < 0.84)
    i_box_mid = int(np.argmin(np.abs(s - 0.78)))
    i_left = int(np.argmax(s >= 0.72))
    rms = np.sqrt(np.mean((sol - x) ** 2, axis=0))
    i1, i2 = int(np.argmin(np.abs(s - 0.30))), int(np.argmin(np.abs(s - 0.58)))
    return {
        "s": s, "truth": x, "mean": mean, "sd": sd, "lam": lam, "err": err, "reps": reps, "n": n,
        "peak1_truth": float(x[i1]), "peak1_mean": float(mean[i1]), "peak2_truth": float(x[i2]), "peak2_mean": float(mean[i2]),
        "box_truth": float(x[i_box_mid]), "box_mean": float(mean[i_box_mid]),
        "edge_before": float(mean[i_left - 1]), "edge_at": float(mean[i_left]), "undershoot": float(mean.min()),
        "rms_box": float(np.mean(rms[box])), "rms_hill1": float(np.mean(rms[np.abs(s - 0.30) < 0.1])),
        "rms_hill2": float(np.mean(rms[np.abs(s - 0.58) < 0.06])), "rms_rest": float(np.mean(rms[(s < 0.15) | (s > 0.9)])),
    }


# ----------------------------------------------------------------------------- Verdikte: (Art, Text)

def verdict_instance(a: Analysis) -> tuple[str, str]:
    s = a.settings
    where = (f"Beim besten Parameter λ* = {lam_text(a.lam_star)} läge er bei {err_pct(a.err_star)}"
             if not a.flat else "Das Tal ist flach: der beste Parameter ist bei diesem Rauschen nicht bestimmt")
    if a.lam < a.lam_star / 30.0 and a.std > a.bias:
        return "warning", (f"Zu wenig Strafe: Die Streuung ({err_pct(a.std)}) dominiert den Bias ({err_pct(a.bias)}); der erwartete Fehler beträgt {err_pct(a.expected)}. {where}.")
    if a.lam > a.lam_star * 30.0 and a.bias > a.std:
        return "warning", (f"Zu viel Strafe: Der Bias ({err_pct(a.bias)}) dominiert die Streuung ({err_pct(a.std)}); die Lösung wird glatt, der erwartete Fehler beträgt "
                           f"{err_pct(a.expected)}. {where}.")
    if s.data == "verbrechen" and a.err_honest_at_lam > 1.3 * a.expected:
        return "info", (f"Mit Daten aus demselben Gitter beträgt der erwartete Fehler {err_pct(a.expected)}; mit ehrlichen Daten wären es {err_pct(a.err_honest_at_lam)}. "
                        "Der Test mit demselben Gitter meldet eine Genauigkeit, die es nicht gibt.")
    return "success", (f"Der erwartete Fehler beträgt {err_pct(a.expected)} (Bias {err_pct(a.bias)}, Streuung {err_pct(a.std)}), die direkte Inversion hätte "
                       f"{err_pct(a.err_naive)}. {where}.")


def verdict_valley(res: dict) -> tuple[str, str]:
    by = {r["factor"]: r for r in res["rows"]}
    lo, hi = by[0.3], by[10.0]
    return "info", (f"Der beste Parameter λ* = {lam_text(res['lam_star'])} gibt {err_pct(res['err_star'])}; die direkte Inversion hat {err_pct(res['naive'])}. "
                    f"Wer ihn um den Faktor 0,3 bis 10 verfehlt, zahlt höchstens {de(100.0 * (max(lo['ratio'], hi['ratio']) - 1.0), 0)} % mehr Fehler "
                    f"({err_pct(lo['closed'])} bei 0,3, {err_pct(hi['closed'])} bei 10); beim Hundertfachen sind es {err_pct(by[100.0]['closed'])}.")


def verdict_grid(rows: list[dict]) -> tuple[str, str]:
    fine = [r for r in rows if r["n"] >= C.DEFAULT_N]
    lo, hi = min(r["tik_honest"] for r in fine), max(r["tik_honest"] for r in fine)
    worst = max(rows, key=lambda r: r["direct"])
    return "success", (f"Ab n = {fine[0]['n']} bleibt der erwartete Fehler mit Tikhonov zwischen {err_pct(lo)} und {err_pct(hi)}, ohne Strafe wächst er bis auf "
                       f"{rel_text(worst['direct'])} (n = {worst['n']}). Besser als die grobe Rechnung wird es nicht: bei n = {rows[1]['n']} genügt "
                       f"{err_pct(rows[1]['tik_honest'])}, weil die Kastenkanten dort fast auf Zellgrenzen liegen.")


def verdict_noise(rows: list[dict]) -> tuple[str, str]:
    first, last = rows[0], rows[-1]
    return "warning", (f"Ehrliche Daten: von {err_pct(first['err_honest'])} bei {noise_label(first['noise'])} Rauschen auf {err_pct(last['err_honest'])} bei "
                       f"{noise_label(last['noise'])}: der Boden ist der Modellfehler des Gitters. Mit Daten aus demselben Gitter sinkt der Fehler "
                       f"auf {err_pct(last['err_crime'])}: ein Test mit demselben Gitter zeigt eine Genauigkeit, die es nicht gibt.")


def verdict_edge(res: dict) -> tuple[str, str]:
    return "info", (f"Die Hügel bleiben erhalten (Mitte {de(res['peak1_mean'], 3)} statt {de(res['peak1_truth'], 3)}, schmaler Hügel {de(res['peak2_mean'], 3)} statt "
                    f"{de(res['peak2_truth'], 3)}); der Kasten verliert in der Mitte {de(100.0 * (1.0 - res['box_mean'] / res['box_truth']), 0)} % "
                    f"({de(res['box_mean'], 2)} statt {de(res['box_truth'], 2)}), und seine Kante läuft über mehrere Punkte. Der Punktfehler ist am Kasten "
                    f"{de(res['rms_box'] / res['rms_hill1'], 1)}-mal so groß wie an den Hügeln.")
