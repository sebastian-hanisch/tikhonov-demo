"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Reglergrenzen."""

import pytest

import tik_constants as C
import tik_evaluation as ev
import tik_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_step_grid():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0, name
        assert C.W_MIN <= p["w"] <= C.W_MAX and (p["w"] - C.W_MIN) / C.W_STEP == pytest.approx(round((p["w"] - C.W_MIN) / C.W_STEP)), name
        assert C.LOGLAM_MIN <= p["loglam"] <= C.LOGLAM_MAX and (p["loglam"] - C.LOGLAM_MIN) / C.LOGLAM_STEP == pytest.approx(round((p["loglam"] - C.LOGLAM_MIN) / C.LOGLAM_STEP)), name
        assert p["noise"] in C.NOISE_CHOICES and p["data"] in C.DATA_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standard"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_default_is_a_valid_widget_value():
    for key, spec in P.SETTING_SPECS.items():
        if spec.choices is not None:
            assert spec.default in spec.choices, key
        else:
            assert spec.lo <= spec.default <= spec.hi, key


def test_the_presets_show_different_outcomes():
    res = {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}
    std, low, high, fine = (res[k] for k in C.PRESETS)
    assert std.expected < 0.15 and low.expected > 1.0 and 0.4 < high.expected < 0.7
    assert low.std > low.bias and high.bias > high.std
    assert fine.expected < 0.15 and fine.err_naive > 1e3


def test_experiment_constants_are_consistent():
    assert 1.0 in C.EXP_FACTORS and 0.3 in C.EXP_FACTORS and 10.0 in C.EXP_FACTORS and 100.0 in C.EXP_FACTORS
    assert C.DEFAULT_N in C.EXP_N_SWEEP and 32 in C.EXP_N_SWEEP and C.EXP_REPS >= 200
    assert C.DEFAULT_NOISE in C.EXP_NOISE_LEVELS and 1e-4 in C.EXP_NOISE_LEVELS
