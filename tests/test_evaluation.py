"""Auswertung: Formatierung, Analyse der Instanz, Verdikte (Art und Text), Form der Experimente und ihre Reproduzierbarkeit."""

import numpy as np
import pytest

import tik_constants as C
import tik_evaluation as ev


# ----------------------------------------------------------------------------- Formatierung

def test_de_and_sci():
    assert ev.de(-1.234, 2) == "−1,23"
    assert ev.sci(2.5e9) == "2,5·10⁹" and ev.sci(3.0e-12) == "3,0·10⁻¹²" and ev.sci(9.96e3) == "1,0·10⁴"


def test_err_pct_digits_by_magnitude():
    assert [ev.err_pct(v) for v in (2.58, 0.1168, 0.0399, 0.0087, 8.7e-4, 4471.0, 0.57)] == ["258 %", "11,7 %", "4,0 %", "0,87 %", "0,09 %", "4,5·10³-fach", "57,0 %"]


def test_lam_text_switches_to_powers_of_ten_below_one_hundredth():
    assert ev.lam_text(0.0095) == "9,5·10⁻³" and ev.lam_text(0.01) == "0,0100" and ev.lam_text(0.5) == "0,500" and ev.lam_text(0.0) == "0"


def test_noise_label():
    assert [ev.noise_label(v) for v in C.NOISE_CHOICES] == ["0 %", "0,0001 %", "0,001 %", "0,01 %", "0,1 %", "1 %", "10 %"]


# ----------------------------------------------------------------------------- Analyse der Instanz

def test_the_default_settings_are_the_standard_preset():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_lambda_zero_corner_equals_the_naive_error():
    a = ev.analyse(ev.Settings(loglam=-8.0))
    assert a.err_tik == pytest.approx(a.err_naive, rel=1e-3)
    assert a.expected == pytest.approx(ev.analyse(ev.Settings(loglam=-8.0)).expected)


def test_the_expected_error_is_bias_and_std_combined_and_df_is_the_filter_sum():
    a = ev.analyse(ev.Settings())
    assert a.expected == pytest.approx(float(np.hypot(a.bias, a.std)), rel=1e-12)
    assert a.df == pytest.approx(float(np.sum(a.f)), rel=1e-12)


def test_analysis_is_deterministic_and_the_seed_matters():
    a, b, c = ev.analyse(ev.Settings()), ev.analyse(ev.Settings()), ev.analyse(ev.Settings(seed=36))
    assert np.array_equal(a.x_tik, b.x_tik) and not np.array_equal(a.x_tik, c.x_tik)


def test_the_valley_curve_contains_the_chosen_lambda_error():
    a = ev.analyse(ev.Settings())
    lo, hi, pts = C.LAM_GRID
    assert len(a.valley["lam"]) == pts and a.valley["lam"][0] == pytest.approx(lo) and a.valley["lam"][-1] == pytest.approx(hi)
    i = int(np.argmin(np.abs(np.log10(a.valley["lam"]) - np.log10(a.lam))))
    assert a.valley["total"][i] == pytest.approx(a.expected, rel=0.05)


def test_crime_data_report_a_smaller_expected_error_than_honest_data():
    a = ev.analyse(ev.Settings(data="verbrechen"))
    assert a.err_crime_at_lam < a.err_honest_at_lam and a.expected == pytest.approx(a.err_crime_at_lam, rel=1e-12)
    assert a.lam_star_crime == pytest.approx(a.lam_star, rel=1e-9)


# ----------------------------------------------------------------------------- Verdikte

def test_verdict_kinds_of_the_presets():
    kinds = {name: ev.verdict_instance(ev.analyse(ev.Settings(**p)))[0] for name, p in C.PRESETS.items()}
    assert kinds == {"Standard": "success", "Zu wenig Strafe": "warning", "Zu viel Strafe": "warning", "Feineres Gitter": "success"}


def test_verdict_texts_name_the_dominating_part():
    low = ev.verdict_instance(ev.analyse(ev.Settings(**C.PRESETS["Zu wenig Strafe"])))[1]
    high = ev.verdict_instance(ev.analyse(ev.Settings(**C.PRESETS["Zu viel Strafe"])))[1]
    assert "Zu wenig Strafe" in low and "Streuung" in low and "Zu viel Strafe" in high and "Bias" in high


def test_crime_verdict_mentions_the_honest_error():
    a = ev.analyse(ev.Settings(data="verbrechen"))
    kind, text = ev.verdict_instance(a)
    assert kind == "info" and ev.err_pct(a.err_honest_at_lam) in text


# ----------------------------------------------------------------------------- Experimente: Form und Reproduzierbarkeit

def test_valley_experiment_has_the_factors_and_is_reproducible():
    r1, r2 = ev.valley_experiment(reps=100), ev.valley_experiment(reps=100)
    assert [r["factor"] for r in r1["rows"]] == list(C.EXP_FACTORS) and r1 == r2
    by = {r["factor"]: r for r in r1["rows"]}
    assert by[1.0]["ratio"] == pytest.approx(1.0, abs=1e-9) and all(r["ratio"] >= 1.0 - 1e-9 for r in r1["rows"])
    assert r1["zero_solution"] == pytest.approx(1.0, abs=1e-6) and r1["naive"] > 2.0


def test_valley_closed_form_and_simulation_agree():
    res = ev.valley_experiment(reps=400)
    for r in res["rows"]:
        assert r["sim"] == pytest.approx(r["closed"], rel=0.06)


def test_grid_experiment_tikhonov_stays_bounded_while_direct_explodes():
    rows = {r["n"]: r for r in ev.grid_experiment()}
    assert rows[96]["direct"] > 1e6 and all(rows[n]["tik_honest"] < 0.2 for n in (48, 64, 80, 96))
    assert all(r["tik_crime"] <= r["tik_honest"] + 1e-9 for r in rows.values())


def test_noise_experiment_rows_and_flags():
    rows = ev.noise_experiment()
    assert [r["noise"] for r in rows] == list(C.EXP_NOISE_LEVELS)
    assert rows[0]["err_honest"] > rows[2]["err_honest"] > rows[-1]["err_honest"] > 0.1
    assert rows[-1]["err_crime"] < 0.01 and rows[0]["flat"] is False and rows[-1]["flat"] is True


def test_edge_experiment_numbers_are_internally_consistent():
    res = ev.edge_experiment(reps=100)
    assert res["rms_box"] > res["rms_hill1"] and res["box_mean"] < res["box_truth"] and res["undershoot"] < 0.0
    assert res["edge_before"] < res["edge_at"] and len(res["s"]) == res["n"]
