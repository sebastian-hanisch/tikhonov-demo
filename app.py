"""Tikhonov-Regularisierung: ein Strafterm gegen das Rauschen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Linie Inverse Probleme und Regularisierung, Wurzel des Verfahrens. Auf der Welt aus Stück 1 (Phantom, Gauß-Unschärfe, Rauschen) wird die
direkte Inversion durch einen Strafterm gedämpft. Kern: tik_model.py, Generator: inverse_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import tik_constants as C
from tik_evaluation import (
    Settings,
    analyse,
    de,
    edge_experiment,
    err_pct,
    grid_experiment,
    lam_text,
    noise_experiment,
    noise_label,
    rel_text,
    sci,
    valley_experiment,
    verdict_edge,
    verdict_grid,
    verdict_instance,
    verdict_noise,
    verdict_valley,
)
from tik_pdf_export import build_pdf
from tik_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from tik_visualization import (
    build_edge_chart,
    build_factor_chart,
    build_filter_chart,
    build_grid_chart,
    build_noise_chart,
    build_recon_chart,
    build_table,
    build_valley_chart,
    style_table,
)

st.set_page_config(page_title="Tikhonov-Regularisierung – Sebastian Hanisch", layout="wide")

SPINNER = "Experiment rechnet über feste Zufallsströme …"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=SPINNER)
def _valley():
    return valley_experiment()


@st.cache_data(show_spinner=SPINNER)
def _grid():
    return grid_experiment()


@st.cache_data(show_spinner=SPINNER)
def _noise():
    return noise_experiment()


@st.cache_data(show_spinner=SPINNER)
def _edge():
    return edge_experiment()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("🧯 Tikhonov-Regularisierung: ein Strafterm gegen das Rauschen")
st.markdown(
    """
**Ein einziger Strafterm macht aus einer wertlosen Lösung eine brauchbare, aber nicht aus jeder Lösung eine richtige.** Dies ist das zweite Stück der Linie
**Inverse Probleme und Regularisierung** und baut auf **Das schlecht gestellte Problem** auf: dasselbe Phantom, dieselbe Gauß-Unschärfe, dasselbe Rauschen. Die
direkte Inversion verstärkt das Rauschen um die Kondition. **Tikhonov** bestraft große Lösungen und nimmt dafür etwas Bias in Kauf. Die Welt ist simuliert; das
Phantom ist die bekannte Wahrheit.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Tikhonov", expanded=True):
    st.markdown(
        r"""
1. **Strafterm.** Statt $\|A x - b\|^2$ allein minimiert man $\|A x - b\|^2 + \lambda^2 \|x\|^2$. Der Parameter $\lambda$ legt fest, wie teuer große Lösungen sind.
2. **Filter.** Die Lösung ist die direkte Inversion, bei der jede Komponente mit $f_i = \sigma_i^2/(\sigma_i^2 + \lambda^2)$ gedämpft wird: Komponenten mit
   $\sigma_i \gg \lambda$ bleiben, solche mit $\sigma_i \ll \lambda$ verschwinden.
3. **Bias und Streuung.** Die Dämpfung verzerrt die Lösung (Bias) und senkt dafür die Streuung durch das Rauschen. Der Fehler hat ein Tal: zu wenig Strafe lässt das
   Rauschen durch, zu viel macht die Lösung flach.
4. **Der Parameter ist nicht gegeben.** Das beste $\lambda^*$ kennt man nur, wenn man die Wahrheit kennt; wie man es ohne sie findet, zeigt das Folgestück
   **Parameterwahl**.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Messung**")
    n = st.slider("Gitterpunkte n", *bounds("n_slider"), step=C.N_STEP, key="n_slider",
                  help="Wie viele Messpunkte und Unbekannte. Mit Strafe schadet ein feineres Gitter nicht mehr.")
    w = st.slider("Breite der Unschärfe w", *bounds("w_slider"), step=C.W_STEP, key="w_slider", format="%.3f",
                  help="Breite des Gauß-Kerns als Anteil der Länge 1. Je breiter, desto stärker fallen die Singulärwerte.")
    noise = st.select_slider("Rauschen (relativ)", options=C.NOISE_CHOICES, key="noise_select", format_func=noise_label,
                             help="Standardabweichung des weißen Rauschens, relativ zum Effektivwert der exakten Daten.")
    st.markdown("**Daten**")
    data = st.selectbox("Woher kommen die Daten?", C.DATA_CHOICES, key="data_select", format_func=lambda v: C.DATA_LABELS[v],
                        help="Ehrliche Daten kommen aus der analytischen Unschärfe und sind vom Gitter unabhängig. Daten aus demselben Gitter entstehen aus "
                             "der Matrix, die danach invertiert wird (inverse crime).")
    st.markdown("**Strafe**")
    loglam = st.slider("Strafparameter log₁₀ λ", *bounds("loglam_slider"), step=C.LOGLAM_STEP, key="loglam_slider", format="%.2f",
                       help="Zehnerlogarithmus von λ. Weit links: keine Strafe (direkte Inversion). Weit rechts: die Lösung wird null.")
    seed = st.number_input("Zufalls-Seed des Rauschens", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Rauschen würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "w_slider": round(float(w), 3), "noise_select": float(noise), "data_select": data,
                   "loglam_slider": round(float(loglam), 2), "seed_input": int(seed)})

settings = Settings(int(n), round(float(w), 3), float(noise), data, round(float(loglam), 2), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fehler der Tikhonov-Lösung", err_pct(a.err_tik), delta=f"direkte Inversion: {err_pct(a.err_naive)}", delta_color="off",
          help="‖x̂ − x‖ / ‖x‖ dieser Stichprobe, mit der Wahrheit x des Phantoms.")
c2.metric("Erwarteter Fehler", err_pct(a.expected), delta=f"Bias {err_pct(a.bias)}, Streuung {err_pct(a.std)}", delta_color="off",
          help="Geschlossene Formel: Wurzel aus Bias² plus Varianz, relativ zur Signalnorm, gemittelt über das Rauschen.")
c3.metric("Freiheitsgrade", f"{de(a.df, 1)} von {settings.n}", delta=f"k* = {a.kstar} Komponenten über dem Rauschen", delta_color="off",
          help="Summe der Filterfaktoren: wie viele Komponenten die Strafe wirksam durchlässt.")
c4.metric("Strafparameter λ", lam_text(a.lam), delta=f"Kondition {sci(a.cond) if a.cond >= 1000 else de(a.cond, 1)}", delta_color="off",
          help="10 hoch dem Reglerwert; λ = 0 wäre die direkte Inversion von Stück 1.")

tab_rec, tab_val, tab_fil, tab_tab = st.tabs(["Rekonstruktion", "Fehlertal", "Filter", "Tabelle"])
reveal = st.session_state.get("reveal_oracle", False)
with tab_rec:
    st.plotly_chart(build_recon_chart(a), width="stretch", key="recon_chart")
    st.caption("Grün die Wahrheit, blau die Tikhonov-Lösung, blass rot die direkte Inversion. Die Achse ist fest; was darüber hinausgeht, wird abgeschnitten.")
with tab_val:
    st.plotly_chart(build_valley_chart(a, reveal), width="stretch", key="valley_chart")
    st.caption("Erwarteter Fehler (schwarz), Bias (violett) und Streuung (rot) über λ, geschlossen gerechnet. Links dominiert die Streuung, rechts der Bias.")
with tab_fil:
    st.plotly_chart(build_filter_chart(a), width="stretch", key="filter_chart")
    st.caption("Filterfaktoren der Komponenten: vorn bleibt fast alles, hinten wird gedämpft. Die orange Marke ist k* aus Stück 1.")
with tab_tab:
    st.dataframe(style_table(build_table(a)), hide_index=True, width="stretch", height=min(40 + 35 * min(settings.n, 40), 560))
    st.caption("Die ersten Punkte der Instanz; „Direkte Inversion“ in Zehnerpotenzen, wenn der Betrag über 10 000 liegt.")

if st.button("📄 PDF der aktuellen Instanz erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="tikhonov.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ist der Strafparameter gut gewählt?")
st.toggle("🙈 Orakel zeigen (bester Parameter aus der Wahrheit)", key="reveal_oracle")
if st.session_state.get("reveal_oracle", False):
    r1, r2, r3 = st.columns(3)
    r1.metric("Bester Parameter λ*", "nicht bestimmt (flaches Tal)" if a.flat else lam_text(a.lam_star),
              help="Minimum des erwarteten Fehlers über λ; er braucht die Wahrheit und existiert in der Praxis nicht.")
    r2.metric("Erwarteter Fehler bei λ*", err_pct(a.err_star))
    r3.metric("λ im Vielfachen von λ*", "–" if a.flat else f"{de(a.lam / a.lam_star, 2)}×")
r4, r5 = st.columns(2)
r4.metric("Erwarteter Fehler mit ehrlichen Daten", err_pct(a.err_honest_at_lam), help="Zum gewählten λ, mit der analytischen Unschärfe als Daten.")
r5.metric("Erwarteter Fehler mit Daten aus demselben Gitter", err_pct(a.err_crime_at_lam), help="Zum gewählten λ, mit Daten aus der Matrix, die invertiert wird.")
verdict(verdict_instance(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Tal und seine Nachsicht")
res1 = _valley()
st.caption(f"Standardinstanz (n = {res1['n']}), ehrliche Daten, {res1['reps']} feste Wiederholungen je Punkt. Gezeigt ist der Fehler bei Vielfachen des besten Parameters "
           f"λ* = {lam_text(res1['lam_star'])}.")
st.plotly_chart(build_factor_chart(res1), width="stretch", key="factor_chart")
verdict(verdict_valley(res1))
st.dataframe(
    [{"Vielfaches von λ*": de(r["factor"], 2).rstrip("0").rstrip(","), "λ": lam_text(r["lam"]), "Erwartet": err_pct(r["closed"]), "Simuliert": err_pct(r["sim"]),
      "Verhältnis zum Optimum": de(r["ratio"], 2), "Anteil Bias²": f"{de(100.0 * r['bias_share'], 0)} %", "Freiheitsgrade": de(r["df"], 1)} for r in res1["rows"]],
    hide_index=True, width="stretch")
st.caption(f"Ohne Strafe (λ = 0) beträgt der erwartete Fehler {err_pct(res1['naive'])}, bei sehr großem λ geht die Lösung gegen null und der Fehler gegen "
           f"{err_pct(res1['zero_solution'])}. Das Tal ist links steiler als rechts: lieber etwas zu viel Strafe als zu wenig.")

st.subheader("🔬 Experiment 2: Feineres Gitter schadet nicht mehr")
rows2 = _grid()
st.caption("Standardunschärfe (w = 0,030), absolutes Rauschen der Standardinstanz, erwarteter Fehler (geschlossen). In Stück 1 wuchs der Fehler mit n ins Astronomische; "
           "hier steht daneben die Tikhonov-Lösung beim besten Parameter.")
st.plotly_chart(build_grid_chart(rows2), width="stretch", key="grid_chart")
verdict(verdict_grid(rows2))
st.dataframe(
    [{"n": r["n"], "Kondition": sci(r["cond"]) if r["cond"] >= 1000 else de(r["cond"], 1), "Direkte Inversion": rel_text(r["direct"]),
      "λ* (ehrlich)": lam_text(r["lam_honest"]), "Tikhonov (ehrlich)": err_pct(r["tik_honest"]), "Tikhonov (selbes Gitter)": err_pct(r["tik_crime"])} for r in rows2],
    hide_index=True, width="stretch")
st.caption("Bei kleinen n braucht es keine Strafe (λ* an der Untergrenze): das grobe Gitter regularisiert selbst. Dass n = 32 hier besser abschneidet als n = 48 bis 96, hängt "
           "an der Lage der Kastenkanten zum Gitter und ist kein allgemeiner Befund.")

st.subheader("🔬 Experiment 3: Weniger Rauschen hilft bis zum Boden")
rows3 = _noise()
st.caption(f"Standardinstanz (n = {C.DEFAULT_N}), bester Parameter je Rauschstufe, erwarteter Fehler (geschlossen): ehrliche Daten gegen Daten aus demselben Gitter.")
st.plotly_chart(build_noise_chart(rows3), width="stretch", key="noise_chart")
verdict(verdict_noise(rows3))
st.dataframe(
    [{"Rauschen": noise_label(r["noise"]), "λ* (ehrlich)": ("flach" if r["flat"] else lam_text(r["lam_honest"])), "Fehler (ehrlich)": err_pct(r["err_honest"]),
      "λ* (selbes Gitter)": lam_text(r["lam_crime"]), "Fehler (selbes Gitter)": err_pct(r["err_crime"]),
      "λ aus dem Test auf ehrlichen Daten": err_pct(r["err_crime_lam_on_honest"])} for r in rows3],
    hide_index=True, width="stretch")
first_flat = next((r["noise"] for r in rows3 if r["flat"]), None)
st.caption("Die letzte Spalte zeigt: Der aus dem Test mit demselben Gitter gewählte Parameter kostet auf ehrlichen Daten kaum etwas. Der Schaden ist die angezeigte Zahl, "
           "nicht die Wahl." + (f" Ab {noise_label(first_flat)} Rauschen und darunter ist das Tal flach und λ* nicht mehr bestimmt." if first_flat is not None else ""))

st.subheader("🔬 Experiment 4: Was bleibt an den Kanten")
res4 = _edge()
st.caption(f"Standardinstanz, λ* = {lam_text(res4['lam'])}, {res4['reps']} feste Wiederholungen mit ehrlichen Daten: Mittel der Lösung und ±1 Streuung gegen die Wahrheit.")
st.plotly_chart(build_edge_chart(res4), width="stretch", key="edge_chart")
verdict(verdict_edge(res4))
st.dataframe(
    [{"Bereich": name, "Punktfehler (RMS)": de(val, 4)} for name, val in (("Kasten", res4["rms_box"]), ("Hügel 1", res4["rms_hill1"]), ("Hügel 2 (schmal)", res4["rms_hill2"]),
                                                                            ("Rand", res4["rms_rest"]))],
    hide_index=True, width="stretch")
st.caption(f"Der Strafterm kennt keine Kanten: Er dämpft die schnellen Komponenten, und der Kasten braucht sie. Das Mittel der Lösung unterschreitet die Null bis auf "
           f"{de(res4['undershoot'], 3)}. Eine Strafe, die Sprünge schont, ist der Inhalt des Folgestücks **Sparsity und Total Variation**.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Parameter ist bekannt** | Das beste λ\\* braucht die Wahrheit. Ohne sie muss man es aus den Daten schätzen; Fehlgriffe um den Faktor 100 kosten {err_pct(res1['rows'][-1]['closed'])} bis {err_pct(res1['rows'][0]['closed'])}. | Parameterwahl |
| **Nur ‖x‖² wird bestraft** | Die Lösung wird kleiner, nicht glatter; Kanten und schmale Spitzen leiden (Experiment 4). Eine Glättungs- oder Sprungstrafe ist ein anderes Verfahren. | Sparsity und Total Variation |
| **Das Rauschen ist weiß und bekannt** | Formel und Optimum kennen σ; ein unbekanntes σ muss geschätzt werden. | Parameterwahl |
| **Das Gitter bildet das Phantom ab** | Bei n = 48 liegt der Boden bei {err_pct(rows3[-1]['err_honest'])}: der Modellfehler des Gitters, nicht das Rauschen (Experiment 3). | Auflösung und Versuchsplanung |
| **Eine Dimension, ein Phantom** | Zwei Hügel und ein Kasten, weißes Rauschen; andere Formen verschieben die Zahlen. | – |
| **Doppelte Genauigkeit** | Ab Kondition rund 10¹⁵ ist die Matrix numerisch singulär; die Strafe löst das nicht, die Daten tragen die Information nicht. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gegeben $A = U\,\mathrm{diag}(\sigma_1 \ge \dots \ge \sigma_n)\,V^\top$ und Daten $b = A x + r + e$ mit Modellfehler $r$ (null bei Daten aus demselben Gitter)
und Rauschen $e \sim N(0, \sigma_e^2 I)$.

**Tikhonov.** $x_\lambda = \arg\min_x \|A x - b\|^2 + \lambda^2 \|x\|^2 = (A^\top A + \lambda^2 I)^{-1} A^\top b = V\,\mathrm{diag}\!\left(\frac{\sigma_i}{\sigma_i^2 + \lambda^2}\right) U^\top b$.
Mit den Filterfaktoren $f_i = \sigma_i^2/(\sigma_i^2 + \lambda^2)$ ist $x_\lambda = \sum_i f_i\,\frac{u_i^\top b}{\sigma_i}\,v_i$. Für $\lambda \to 0$ ist es die direkte Inversion, für
$\lambda \to \infty$ die Nulllösung.

**Bias und Varianz.** Mit $c_i = v_i^\top x$ und $r_i = u_i^\top r$ ist $E[x_\lambda] - x = \sum_i \left(-\frac{\lambda^2 c_i}{\sigma_i^2 + \lambda^2} + \frac{\sigma_i r_i}{\sigma_i^2 + \lambda^2}\right) v_i$ und
$\mathrm{Var} = \sigma_e^2 \sum_i \frac{\sigma_i^2}{(\sigma_i^2 + \lambda^2)^2}$. Der erwartete Fehler ist $\sqrt{\|\text{Bias}\|^2 + \mathrm{Var}}\,/\,\|x\|$.

**Freiheitsgrade.** $\mathrm{df}(\lambda) = \sum_i f_i = \mathrm{tr}\,[A (A^\top A + \lambda^2 I)^{-1} A^\top]$, die Spur der Hat-Matrix.

**Bester Parameter.** $\lambda^* = \arg\min_\lambda \sqrt{\|\text{Bias}\|^2 + \mathrm{Var}}$ braucht die Wahrheit $x$ (Orakel). Das Tal gilt als flach, wenn der Fehler beim 10-fachen und
beim 0,1-fachen von $\lambda^*$ weniger als 0,5 Prozentpunkte der Signalnorm höher liegt.

**Literatur.** Das Beispiel ist erfunden; das Verfahren geht zurück auf A. N. Tikhonov (1963) und D. L. Phillips (1962), in der Statistik als Ridge-Regression bei A. E. Hoerl
und R. W. Kennard (1970); Filterfaktoren und Bias-Varianz-Zerlegung stehen bei P. C. Hansen, *Discrete Inverse Problems: Insight and Algorithms*, SIAM 2010
(alles aus dem Gedächtnis, nicht geprüft).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
