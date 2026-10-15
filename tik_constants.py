"""Konstanten: Regler-Grenzen, feste Annahmen, Experimente, Farben und Presets."""

# Regler. Rechenaufwand am Maximum: SVD einer 96 x 96-Matrix (O(n^3), unter 20 ms); die Experimente nutzen feste Ströme mit 400 Wiederholungen.
N_MIN, N_MAX, N_STEP, DEFAULT_N = 12, 96, 4, 48
W_MIN, W_MAX, W_STEP, DEFAULT_W = 0.01, 0.05, 0.005, 0.03        # Breite der Unschärfe (Anteil der Länge 1)
NOISE_CHOICES = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)        # Rauschen relativ zum Effektivwert der exakten Daten
DEFAULT_NOISE = 1e-3
DATA_CHOICES = ("ehrlich", "verbrechen")
DATA_LABELS = {"ehrlich": "Ehrliche Daten (analytisch, vom Gitter unabhängig)", "verbrechen": "Daten aus demselben Gitter (inverse crime)"}
LOGLAM_MIN, LOGLAM_MAX, LOGLAM_STEP, DEFAULT_LOGLAM = -8.0, 1.0, 0.25, -2.0   # Zehnerlogarithmus des Strafparameters lambda
SEED_MAX, DEFAULT_SEED = 999999, 35

SINGULAR_COND = 1e15                                              # ab hier ist die Matrix in doppelter Genauigkeit numerisch singulär
SHOW_RANGE = (-1.5, 2.0)                                          # Wertebereich der Rekonstruktion im Diagramm; größere Werte werden abgeschnitten
FLAT_TOL = 0.005                                                  # Das Tal ist flach, wenn der Fehler 10-fach daneben weniger als 0,5 Punkte höher liegt
LAM_GRID = (1e-9, 10.0, 241)                                      # untere Grenze, obere Grenze, Zahl der Punkte (logarithmisch) für das Fehlertal

# Experimente: feste Ströme (Seeds aus SEED_BASE), nie ein einzelner Seed als Befund.
SEED_BASE = 900000
EXP_REPS = 400
EXP_FACTORS = (0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 10.0, 100.0)   # Experiment 1: Vielfache des besten Parameters
EXP_N_SWEEP = (24, 32, 48, 64, 80, 96)                            # Experiment 2
EXP_NOISE_LEVELS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)           # Experiment 3

COLOR_TRUTH = "#2F6B65"
COLOR_NAIVE = "#c0392b"
COLOR_TIK = "#4c78a8"
COLOR_CRIME = "#D68A2E"
COLOR_BIAS = "#8e44ad"
COLOR_GREY = "#8a8a8a"

PRESETS = {
    "Standard": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "loglam": DEFAULT_LOGLAM, "seed": DEFAULT_SEED},
    "Zu wenig Strafe": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "loglam": -4.0, "seed": DEFAULT_SEED},
    "Zu viel Strafe": {"n": DEFAULT_N, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "loglam": 0.0, "seed": DEFAULT_SEED},
    "Feineres Gitter": {"n": 96, "w": DEFAULT_W, "noise": DEFAULT_NOISE, "data": "ehrlich", "loglam": DEFAULT_LOGLAM, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standard": "Standardinstanz aus Stück 1 mit λ = 0,01: nahe dem besten Wert, der Fehler fällt von 258 % auf rund 12 %.",
    "Zu wenig Strafe": "λ = 10⁻⁴: das Rauschen wird kaum gedämpft, der Fehler liegt noch über 100 %. Die Varianz dominiert.",
    "Zu viel Strafe": "λ = 1: die Lösung wird glatt und flach, der Fehler liegt bei 57 %. Der Bias dominiert.",
    "Feineres Gitter": "n = 96 statt 48: ohne Strafe wäre der Fehler astronomisch (Stück 1), mit λ = 0,01 liegt er bei rund 10 %.",
}
