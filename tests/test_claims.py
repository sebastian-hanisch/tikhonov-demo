"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Experimente laufen über feste Zufallsströme (tik_constants.SEED_BASE), nie über einen einzelnen Seed als Befund; die Standard-Instanz der Primäransicht ist
ausdrücklich ein einzelner Strom (Seed 35). Erwartete Fehler sind geschlossene Formeln (Erwartungswerte), Simulationswerte tragen die Toleranz ihrer Wiederholungen.
"""

import numpy as np
import pytest

import tik_constants as C
import tik_evaluation as ev
import tik_model as model


@pytest.fixture(scope="module")
def valley():
    return ev.valley_experiment()


@pytest.fixture(scope="module")
def grid_rows():
    return {r["n"]: r for r in ev.grid_experiment()}


@pytest.fixture(scope="module")
def noise_rows():
    return {r["noise"]: r for r in ev.noise_experiment()}


@pytest.fixture(scope="module")
def edge():
    return ev.edge_experiment()


# ----------------------------------------------------------------------------- Standardinstanz und Presets

def test_standard_instance_expected_error_bias_variance_and_dof():
    a = ev.analyse(ev.Settings())
    assert a.expected == pytest.approx(0.1168, abs=0.0005) and a.bias == pytest.approx(0.1154, abs=0.0005) and a.std == pytest.approx(0.0177, abs=0.0005)
    assert a.df == pytest.approx(32.7, abs=0.1) and a.kstar == 31 and a.cond == pytest.approx(1.26e4, rel=0.01)


def test_standard_instance_sample_error_against_the_direct_inversion():
    a = ev.analyse(ev.Settings())
    assert a.err_tik == pytest.approx(0.1178, abs=0.002) and a.err_naive == pytest.approx(2.578, abs=0.02)


def test_best_parameter_and_the_crime_comparison():
    a = ev.analyse(ev.Settings())
    assert a.lam_star == pytest.approx(0.0095, rel=0.01) and a.err_star == pytest.approx(0.1168, abs=0.0005) and a.flat is False
    assert a.lam_star_crime == pytest.approx(0.00707, rel=0.01) and a.err_star_crime == pytest.approx(0.0710, abs=0.0005)
    assert a.err_crime_at_lam == pytest.approx(0.0724, abs=0.0005) and a.err_honest_at_lam == pytest.approx(0.1168, abs=0.0005)


def test_preset_numbers_in_the_help_texts():
    low = ev.analyse(ev.Settings(**C.PRESETS["Zu wenig Strafe"]))
    high = ev.analyse(ev.Settings(**C.PRESETS["Zu viel Strafe"]))
    fine = ev.analyse(ev.Settings(**C.PRESETS["Feineres Gitter"]))
    assert low.err_tik == pytest.approx(1.146, abs=0.01) and low.expected == pytest.approx(1.349, abs=0.005) and low.std == pytest.approx(1.34, abs=0.01) and low.bias == pytest.approx(0.11, abs=0.01)
    assert high.expected == pytest.approx(0.5735, abs=0.002) and high.bias == pytest.approx(0.5735, abs=0.002) and high.std < 0.001
    assert fine.expected == pytest.approx(0.102, abs=0.002) and fine.err_naive > 1e11 and fine.cond > 1e15


# ----------------------------------------------------------------------------- Experiment 1

def test_experiment_1_best_parameter_naive_and_zero_solution(valley):
    assert valley["lam_star"] == pytest.approx(0.0095, rel=0.01) and valley["err_star"] == pytest.approx(0.1168, abs=0.0005)
    assert valley["naive"] == pytest.approx(2.469, abs=0.005) and valley["zero_solution"] == pytest.approx(1.0, abs=1e-5)


def test_experiment_1_error_by_factor(valley):
    by = {r["factor"]: r for r in valley["rows"]}
    assert by[0.01]["closed"] == pytest.approx(1.399, abs=0.01) and by[0.1]["closed"] == pytest.approx(0.2024, abs=0.002)
    assert by[0.3]["closed"] == pytest.approx(0.1289, abs=0.001) and by[0.5]["closed"] == pytest.approx(0.1199, abs=0.001)
    assert by[2.0]["closed"] == pytest.approx(0.1187, abs=0.001) and by[3.0]["closed"] == pytest.approx(0.1213, abs=0.001)
    assert by[10.0]["closed"] == pytest.approx(0.1353, abs=0.001) and by[100.0]["closed"] == pytest.approx(0.5507, abs=0.003)


def test_experiment_1_ratios_and_bias_shares(valley):
    by = {r["factor"]: r for r in valley["rows"]}
    assert by[0.1]["ratio"] == pytest.approx(1.73, abs=0.02) and by[0.3]["ratio"] == pytest.approx(1.10, abs=0.01) and by[10.0]["ratio"] == pytest.approx(1.16, abs=0.01)
    assert by[0.01]["bias_share"] < 0.01 and by[0.1]["bias_share"] == pytest.approx(0.33, abs=0.02) and by[10.0]["bias_share"] > 0.99
    assert by[1.0]["df"] == pytest.approx(32.9, abs=0.1)


def test_experiment_1_simulation_confirms_the_closed_form(valley):
    for r in valley["rows"]:
        assert r["sim"] == pytest.approx(r["closed"], rel=0.02), r["factor"]


# ----------------------------------------------------------------------------- Experiment 2

def test_experiment_2_tikhonov_error_by_grid(grid_rows):
    assert grid_rows[48]["tik_honest"] == pytest.approx(0.1168, abs=0.001) and grid_rows[64]["tik_honest"] == pytest.approx(0.0961, abs=0.001)
    assert grid_rows[80]["tik_honest"] == pytest.approx(0.1031, abs=0.001) and grid_rows[96]["tik_honest"] == pytest.approx(0.1022, abs=0.001)
    assert grid_rows[32]["tik_honest"] == pytest.approx(0.0423, abs=0.001) and grid_rows[24]["tik_honest"] == pytest.approx(0.0944, abs=0.001)


def test_experiment_2_direct_inversion_by_grid(grid_rows):
    assert grid_rows[48]["direct"] == pytest.approx(2.47, abs=0.02) and grid_rows[64]["direct"] == pytest.approx(4583, rel=0.01)
    assert grid_rows[80]["direct"] == pytest.approx(8.7e7, rel=0.02) and grid_rows[96]["direct"] > 1e12


def test_experiment_2_coarse_grids_need_no_penalty(grid_rows):
    assert grid_rows[24]["lam_honest"] < 1e-6 and grid_rows[32]["lam_honest"] < 1e-6
    assert grid_rows[48]["lam_honest"] == pytest.approx(0.0095, rel=0.01)


# ----------------------------------------------------------------------------- Experiment 3

def test_experiment_3_honest_error_has_a_floor(noise_rows):
    honest = {k: v["err_honest"] for k, v in noise_rows.items()}
    assert honest[1e-1] == pytest.approx(0.192, abs=0.002) and honest[1e-2] == pytest.approx(0.133, abs=0.002) and honest[1e-3] == pytest.approx(0.1168, abs=0.001)
    assert honest[1e-4] == pytest.approx(0.1145, abs=0.001) and honest[1e-5] == pytest.approx(0.1122, abs=0.001) and honest[1e-6] == pytest.approx(0.1095, abs=0.001)


def test_experiment_3_crime_error_keeps_falling(noise_rows):
    crime = {k: v["err_crime"] for k, v in noise_rows.items()}
    assert crime[1e-1] == pytest.approx(0.1835, abs=0.002) and crime[1e-2] == pytest.approx(0.1065, abs=0.002) and crime[1e-3] == pytest.approx(0.0710, abs=0.001)
    assert crime[1e-4] == pytest.approx(0.0607, abs=0.001) and crime[1e-5] == pytest.approx(0.0223, abs=0.001) and crime[1e-6] == pytest.approx(0.00246, abs=0.0002)


def test_experiment_3_crime_parameter_costs_little_on_honest_data(noise_rows):
    for k, v in noise_rows.items():
        assert v["err_crime_lam_on_honest"] - v["err_honest"] < 0.0035, k
    assert noise_rows[1e-4]["err_crime_lam_on_honest"] == pytest.approx(0.1176, abs=0.001)


def test_experiment_3_valley_is_flat_from_one_in_a_hundred_thousand(noise_rows):
    assert [noise_rows[k]["flat"] for k in sorted(noise_rows, reverse=True)] == [False, False, False, False, True, True]


# ----------------------------------------------------------------------------- Experiment 4

def test_experiment_4_hills_box_and_edge(edge):
    assert edge["peak1_mean"] == pytest.approx(0.999, abs=0.002) and edge["peak2_mean"] == pytest.approx(0.676, abs=0.003) and edge["peak2_truth"] == pytest.approx(0.681, abs=0.002)
    assert edge["box_mean"] == pytest.approx(0.461, abs=0.003) and edge["box_truth"] == pytest.approx(0.5, abs=0.001)
    assert edge["edge_before"] == pytest.approx(0.224, abs=0.005) and edge["edge_at"] == pytest.approx(0.511, abs=0.005) and edge["undershoot"] == pytest.approx(-0.028, abs=0.003)


def test_experiment_4_pointwise_errors(edge):
    assert edge["rms_box"] == pytest.approx(0.0260, abs=0.001) and edge["rms_hill1"] == pytest.approx(0.0069, abs=0.0005)
    assert edge["rms_hill2"] == pytest.approx(0.0079, abs=0.0005) and edge["rms_rest"] == pytest.approx(0.0084, abs=0.0005)
    assert edge["rms_box"] / edge["rms_hill1"] == pytest.approx(3.8, abs=0.2)


def test_mini_instance_by_hand_numbers(mini):
    a, x, b, lam = mini
    u, sv, vt = model.decompose(a)
    assert model.solve(u, sv, vt, b[None, :], lam)[0] == pytest.approx([1.153465, 1.653465, 1.153465], abs=1e-6)
    bias2, var = model.bias_variance(sv, vt @ x, np.zeros(3), 1.0, lam)
    assert bias2 ** 0.5 / float(np.linalg.norm(x)) == pytest.approx(0.1669, abs=1e-4) and var == pytest.approx(29.68, abs=0.01)
