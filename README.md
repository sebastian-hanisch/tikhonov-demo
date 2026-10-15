# Tikhonov-Regularisierung: ein Strafterm gegen das Rauschen – Streamlit-Demo

Zweites Stück der Linie **Inverse Probleme und Regularisierung** in der „Konzepte“-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning). Es baut auf [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo) auf und nutzt dieselbe Welt: ein Phantom aus zwei Hügeln und einem Kasten, gemessen mit einer Gauß-Unschärfe und weißem Rauschen. Die Wahrheit ist bekannt, weil das Phantom selbst die Wahrheit ist.

## Kernfrage

Die direkte Inversion verstärkt das Rauschen um die Kondition der Matrix. Ein Strafterm auf die Größe der Lösung dämpft das. Wie weit senkt er den Fehler, wie genau muss der Strafparameter stimmen, schadet ein feineres Gitter noch, und was bleibt, wenn das Rauschen verschwindet?

## Modell

Generator `inverse_generator.py`, wortgleich in allen Stücken der Linie (Phantom, analytische Unschärfe, Gitter, Rauschen, zwei Datenarten):

- **Wahrheit** x(s) auf [0, 1]: zwei Gauß-Hügel und ein Kasten; **Messung** g(t) = ∫ k(t − s) x(s) ds mit dem Gauß-Kern der Breite w, analytisch bekannt.
- **Gitter** mit n Punkten: A = h K. Standardinstanz n = 48, w = 0,03, Rauschen 0,1 % des Effektivwerts (absolut 3,4·10⁻⁴), Kondition 1,26·10⁴.
- **Daten:** *ehrlich* (analytisch, vom Gitter unabhängig) oder *aus demselben Gitter* (b = A x + e, „inverse crime“); das Rauschen ist in beiden gleich.
- **Tikhonov:** x_λ = argmin ‖A x − b‖² + λ²‖x‖² = V diag(σᵢ/(σᵢ² + λ²)) Uᵀ b.

Alles ist erfunden; die Wahrheit ist das simulierte Phantom.

## Methodik

- **Filterform** über die Singulärwertzerlegung; Filterfaktor fᵢ = σᵢ²/(σᵢ² + λ²), Freiheitsgrade Σ fᵢ.
- **Geschlossene Bias-Varianz-Zerlegung:** E‖x_λ − x‖² = Σ[(−λ²cᵢ/(σᵢ² + λ²) + σᵢrᵢ/(σᵢ² + λ²))² + σ²σᵢ²/(σᵢ² + λ²)²] mit c = Vᵀx und dem Modellfehler r der ehrlichen Daten. Die App zeigt ihre Erwartungswerte; die Stichprobe der Primäransicht ist ein einzelner Strom.
- **Bester Parameter λ\*:** Minimum des erwarteten Fehlers über λ (Gitter plus goldener Schnitt); er braucht die Wahrheit und ist ein Orakel. Das Tal gilt als *flach*, wenn der Fehler beim 10-fachen und beim 0,1-fachen von λ\* weniger als 0,5 Punkte höher liegt.
- **Experimente** laufen über feste Zufallsströme (SplitMix64): 400 Wiederholungen je Punkt, nie ein einzelner Seed.

## Befunde (gemessen, keine Behauptungen)

Standardinstanz (n = 48, w = 0,03, Rauschen 0,1 %, ehrliche Daten), wenn nicht anders genannt. Jede Zahl steht in `tests/test_claims.py`.

| Frage | Befund | Test |
|---|---|---|
| Was bringt der Strafterm? | Mit λ = 0,01: erwarteter Fehler **11,7 %** der Signalnorm (Bias 11,5 %, Streuung 1,8 %), die direkte Inversion hat 247 % (erwartet; Stichprobe mit Seed 35: 258 %, Tikhonov 11,8 %). Freiheitsgrade 32,7 von 48, k\* = 31. | `test_standard_instance_expected_error_bias_variance_and_dof`, `test_standard_instance_sample_error_against_the_direct_inversion` |
| Wie genau muss λ stimmen (Experiment 1)? | Bester Wert λ\* = 0,0095. Faktor 0,3: 12,9 % (×1,10); 0,5: 12,0 %; 2: 11,9 %; 3: 12,1 %; **10: 13,5 % (×1,16)**; 0,1: 20,2 % (×1,73); 0,01: 140 %; 100: 55 %. Das Tal ist links steiler als rechts. | `test_experiment_1_error_by_factor`, `test_experiment_1_ratios_and_bias_shares` |
| Stimmt die geschlossene Formel? | Simulation (400 Wdh.) und Formel liegen bei allen neun Vielfachen auf 2 % beieinander. Bei λ → 0 ergibt sie die direkte Inversion (2,47), bei λ → ∞ die Nulllösung (Fehler 1,000). | `test_experiment_1_simulation_confirms_the_closed_form`, `test_experiment_1_best_parameter_naive_and_zero_solution` |
| Wo kippt Bias gegen Streuung? | Bias² trägt 1 % des Fehlers bei 0,01 λ\*, 33 % bei 0,1 λ\*, 80 % bei 0,3 λ\*, über 99 % ab 10 λ\*. | `test_experiment_1_ratios_and_bias_shares` |
| Schadet ein feineres Gitter noch (Experiment 2)? | Direkte Inversion: 2,5 (n = 48), 4,6·10³ (64), 8,7·10⁷ (80), über 10¹² (96). Tikhonov beim besten Parameter: **11,7 % / 9,6 % / 10,3 % / 10,2 %**. Bei n = 24 und 32 braucht es keine Strafe (λ\* an der Untergrenze): 9,4 % und 4,2 %. | `test_experiment_2_tikhonov_error_by_grid`, `test_experiment_2_direct_inversion_by_grid`, `test_experiment_2_coarse_grids_need_no_penalty` |
| Was bringt weniger Rauschen (Experiment 3)? | Ehrlich: 19,2 / 13,3 / 11,7 / 11,5 / 11,2 / **10,9 %** für 10 % bis 0,0001 % Rauschen; der Boden ist der Modellfehler des Gitters bei n = 48. Daten aus demselben Gitter: 18,4 / 10,7 / 7,1 / 6,1 / 2,2 / **0,25 %**. | `test_experiment_3_honest_error_has_a_floor`, `test_experiment_3_crime_error_keeps_falling` |
| Was kostet der Test mit demselben Gitter beim Parameter? | Höchstens 0,35 Punkte (λ\* aus dem Test mit demselben Gitter auf ehrliche Daten: 11,73 % statt 11,68 % bei 0,1 %, 11,8 % statt 11,5 % bei 0,01 %). Der Schaden ist die angezeigte Zahl: 7,1 % statt 11,7 %. | `test_experiment_3_crime_parameter_costs_little_on_honest_data`, `test_best_parameter_and_the_crime_comparison` |
| Wann ist λ\* nicht mehr bestimmt? | Ab 0,001 % Rauschen (und darunter) ist das Tal flach; λ\* springt von 0,0055 auf 2,4·10⁻⁵ und 10⁻⁹. | `test_experiment_3_valley_is_flat_from_one_in_a_hundred_thousand` |
| Was bleibt an den Kanten (Experiment 4)? | Mittel über 400 Wdh. bei λ\*: Hügel 1 0,999 (Wahrheit 0,999), schmaler Hügel 0,676 (0,681), Kasten Mitte **0,461** (0,5); Kante links 0,224 → 0,511 über zwei Punkte, Unterschwinger bis −0,028. Punktfehler (RMS): Kasten 0,026, Hügel 0,007 bis 0,008, Rand 0,008. | `test_experiment_4_hills_box_and_edge`, `test_experiment_4_pointwise_errors` |
| Presets | Zu wenig Strafe (λ = 10⁻⁴): erwartet 135 %, davon Streuung 134 %. Zu viel (λ = 1): 57 %, ganz Bias. n = 96 mit λ = 0,01: 10,2 %, die direkte Inversion hätte 10¹². | `test_preset_numbers_in_the_help_texts` |
| Mini-Instanz von Hand | 3 × 3-Matrix (Singulärwerte 1 / 0,5 / 0,1), λ = 0,1: Filterfaktoren 0,990 / 0,962 / 0,5, Lösung (1,1535, 1,6535, 1,1535), Bias 16,7 % der Norm, Varianz 29,68 σ² (ohne Strafe 105 σ²). | `test_mini_instance_by_hand_numbers`, `tests/test_model.py` |

## Befunde und Korrekturen gegenüber dem Plan

- **Der Boden ist ein Befund über das Gitter, nicht über Tikhonov.** Bei n = 48 liegen die Kastenkanten auf halben Zellen; bei n = 32 fast auf Zellgrenzen, deshalb 4,2 % schon ohne Strafe. Der Vergleich n = 32 gegen n = 48 ist kein allgemeiner Befund.
- **Kein Fit-Exponent „Fehler ~ Rauschen^p“:** die lokalen Steigungen je Zehnerschritt laufen von 0,07 bis 0,96; deshalb steht in der App eine Tabelle, keine Zahl.
- **Das Optimum ist nicht überall bestimmt.** Unter 0,001 % Rauschen ist das Tal flach (Fehlerunterschiede unter 0,6 Punkten); die App zeigt dort „flach“ statt eines λ\*.
- **Größere Abweichung der Stichprobe vom Erwartungswert** bei zu wenig Strafe: Seed 35 liefert 115 %, erwartet sind 135 %; die Streuung der Streuung ist groß, wenn das Rauschen durchschlägt.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Parameter ist bekannt** | Das beste λ\* braucht die Wahrheit. | Parameterwahl |
| **Nur ‖x‖² wird bestraft** | Die Lösung wird kleiner, nicht glatter; Kanten und schmale Spitzen leiden. | Sparsity und Total Variation |
| **Das Rauschen ist weiß und seine Größe bekannt** | Formel und Optimum kennen σ. | Parameterwahl |
| **Das Gitter bildet das Phantom ab** | Der Boden bei 11 % ist der Modellfehler des Gitters. | Auflösung und Versuchsplanung |
| **Eine Dimension, ein Phantom** | Andere Formen verschieben die Zahlen. | – |
| **Doppelte Genauigkeit** | Ab Kondition rund 10¹⁵ ist die Matrix numerisch singulär; die Strafe löst das nicht. | – |

Die Zahlen gelten für diese erfundene Welt und zeigen nichts über echte Messungen.

## Bezug zu den Nachbarn im Portfolio

- [schlecht-gestellt-demo](https://github.com/sebastian-hanisch/schlecht-gestellt-demo): Stück 1 der Linie; liefert die Welt, die direkte Inversion und k\*. Hier ist λ = 0 nur der Grenzfall.
- [pca-demo](https://github.com/sebastian-hanisch/pca-demo): Singulärwertzerlegung als Datenwerkzeug; hier ist sie das Mittel, die Strafe komponentenweise zu lesen.
- [kalman-filter-demo](https://github.com/sebastian-hanisch/kalman-filter-demo): schätzt einen Zustand sequentiell; der Strafterm ist später in dieser Linie der Prior der Bayes-Sicht.

## Tests

Mini-Instanz von Hand (3 × 3), Filterform gegen Normalgleichungen und erweiterte kleinste Quadrate, Freiheitsgrade gegen die Spur der Hat-Matrix, Bias und Varianz gegen Monte Carlo und eine direkte Kovarianzrechnung, bester Parameter gegen scipy (`minimize_scalar`), Generator (SplitMix64 gegen eine unabhängige Fassung), Auswertung, jede README-Zahl in `test_claims.py`, AppTest (Presets, Slider-Grenzen, jede Rauschstufe und Datenart, Orakel, PDF, Permalink), Footer, Presets und Laufzeit-Importe.

```bash
python -m pytest tests -q
```

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `inverse_generator.py` | gemeinsamer Generator der Linie (SplitMix64, Phantom, analytische Unschärfe, Operator, Daten) |
| `tik_model.py` | Filter, Lösung, Bias-Varianz-Zerlegung, Freiheitsgrade, bester Parameter |
| `tik_evaluation.py` | Kennzahlen, Experimente, Verdikte |
| `tik_visualization.py`, `tik_pdf_export.py` | Figuren, PDF |
| `tik_constants.py`, `tik_presets.py` | Regler, Presets, Permalink |
| `tests/` | Mini-Instanz, Orakel, Claims, App |

## Bewusst nicht umgesetzt

- **Parameterwahl ohne Wahrheit** (L-Kurve, GCV, Diskrepanzprinzip): das Folgestück; hier ist λ\* ein Orakel.
- **Harter Schnitt statt glatter Filter** (abgeschnittene SVD): das Folgestück.
- **Glättungs- und Sprungstrafen, Sparsity:** spätere Stücke.
- **Bayes-Sicht und Unsicherheit der Rekonstruktion:** spätere Stücke.
- **Praxisdaten:** nicht gebaut; alle Zahlen sind erzeugt.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python 3.12.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
