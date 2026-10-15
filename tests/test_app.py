"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden
Regler an seine Grenzen und prüft, dass kein Python-Fehler auftritt (insbesondere StreamlitDuplicateElementId bei st.plotly_chart ohne eindeutiges key=)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 180
AUTOPLAY_MARKER = "▶"


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_noise_level_and_data_kind_does_not_raise():
    import tik_constants as C

    for level in C.NOISE_CHOICES:
        for kind in C.DATA_CHOICES:
            at = _fresh_app()
            at.select_slider(key="noise_select").set_value(level)
            at.selectbox(key="data_select").set_value(kind)
            at.run(timeout=TIMEOUT)
            assert not at.exception, (level, kind, [str(e) for e in at.exception])


def test_all_controls_at_min_and_at_max_together_do_not_raise():
    import tik_constants as C

    for edge in ("min", "max"):
        at = _fresh_app()
        for key in ("n_slider", "w_slider", "loglam_slider"):
            s = at.slider(key=key)
            s.set_value(s.min if edge == "min" else s.max)
        at.select_slider(key="noise_select").set_value(C.NOISE_CHOICES[0] if edge == "min" else C.NOISE_CHOICES[-1])
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


def test_the_most_singular_corner_with_no_penalty_does_not_raise():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(96)
    at.slider(key="w_slider").set_value(0.05)
    at.slider(key="loglam_slider").set_value(-8.0)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_every_preset_is_applied_to_the_widgets():
    import tik_constants as C

    labels = {b.label: i for i, b in enumerate(_fresh_app().button)}
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.button[labels[name]].click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert at.slider(key="n_slider").value == preset["n"]
        assert abs(at.slider(key="w_slider").value - preset["w"]) < 1e-9
        assert abs(at.slider(key="loglam_slider").value - preset["loglam"]) < 1e-9
        assert at.select_slider(key="noise_select").value == preset["noise"]
        assert at.selectbox(key="data_select").value == preset["data"]


def test_revealing_the_oracle_adds_the_metrics_and_does_not_raise():
    at = _fresh_app()
    before = len(at.metric)
    at.toggle(key="reveal_oracle").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.metric) == before + 3
    assert any(m.label == "Bester Parameter λ*" for m in at.metric)


def test_the_standard_instance_shows_the_success_verdict_and_four_experiments():
    at = _fresh_app()
    assert any("Der erwartete Fehler beträgt" in s.value for s in at.success)
    headers = [s.value for s in at.subheader]
    assert sum("🔬" in h for h in headers) == 4


def test_too_little_penalty_shows_a_warning():
    at = _fresh_app()
    at.slider(key="loglam_slider").set_value(-5.0).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Zu wenig Strafe" in w.value for w in at.warning)


def test_pdf_is_created_only_after_the_click():
    at = _fresh_app()
    assert "pdf_made" not in at.session_state
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    settings, data = at.session_state["pdf_made"]
    assert data[:5] == b"%PDF-" and len(data) > 1000


def test_permalink_values_are_clipped_and_unknown_choices_ignored():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "999999"
    at.query_params["w"] = "-3"
    at.query_params["r"] = "0.5"
    at.query_params["d"] = "wuerfel"
    at.query_params["l"] = "99"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.slider(key="n_slider").value == 96
    assert abs(at.slider(key="w_slider").value - 0.01) < 1e-9
    assert at.select_slider(key="noise_select").value == 1e-3
    assert at.selectbox(key="data_select").value == "ehrlich"
    assert at.slider(key="loglam_slider").value == 1.0


def test_the_address_bar_mirrors_the_settings():
    at = _fresh_app()
    at.slider(key="loglam_slider").set_value(-3.0).run(timeout=TIMEOUT)
    assert at.query_params["l"] == ["-3.0"] or at.query_params["l"] == "-3.0"
