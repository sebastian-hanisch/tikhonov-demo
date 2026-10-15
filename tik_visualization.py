"""Plotly-Figuren und die Tabelle der Instanz. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import tik_constants as C
from tik_evaluation import Analysis, de, lam_text, noise_label, sci

LOG_FLOOR = 1e-17


def _fixed(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_table(a: Analysis, rows: int = 40) -> pd.DataFrame:
    k = min(rows, a.settings.n)
    return pd.DataFrame({
        "Punkt": [f"{i + 1}" for i in range(k)],
        "Ort s": a.s[:k],
        "Wahrheit x": a.x[:k],
        "Messwert b": a.b[:k],
        "Direkte Inversion": a.x_naive[:k],
        "Tikhonov": a.x_tik[:k],
    })


def style_table(df: pd.DataFrame):
    big = lambda v: sci(v, 1) if abs(v) >= 1e4 else de(v, 3)  # noqa: E731
    return df.style.format({"Ort s": lambda v: de(v, 3), "Wahrheit x": lambda v: de(v, 3), "Messwert b": lambda v: de(v, 4), "Direkte Inversion": big,
                            "Tikhonov": lambda v: de(v, 3)})


def build_recon_chart(a: Analysis) -> go.Figure:
    """Wahrheit, Tikhonov-Lösung und (blass) die direkte Inversion; die Achse ist fest, größere Werte werden abgeschnitten."""
    lo, hi = C.SHOW_RANGE
    peak = float(np.max(np.abs(a.x_naive))) if np.all(np.isfinite(a.x_naive)) else float("inf")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.s, y=a.x_naive, mode="lines", name="direkte Inversion (λ = 0)", line=dict(color=C.COLOR_NAIVE, width=1, dash="dot"), opacity=0.55,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x, mode="lines", name="Wahrheit x", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=a.s, y=a.x_tik, mode="lines+markers", name=f"Tikhonov (λ = {lam_text(a.lam)})", line=dict(color=C.COLOR_TIK, width=2.5),
                             marker=dict(size=5), hovertemplate="s %{x:.3f}<br>x̂ %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert", range=[lo, hi])
    peak_text = "unendlich" if not np.isfinite(peak) else (de(peak, 1) if peak < 1000.0 else sci(peak))
    note = f"Die direkte Inversion reicht bis ±{peak_text} und wird bei {de(lo, 1)} bis {de(hi, 1)} abgeschnitten." if peak > max(abs(lo), abs(hi)) else ""
    return _fixed(fig, 380, title=dict(text=note, font=dict(size=12, color="#9aa0a6"), x=0.0))


def build_filter_chart(a: Analysis) -> go.Figure:
    """Filterfaktoren fᵢ je Komponente: wie viel jede Komponente die Strafe durchlässt; die Marke k* aus Stück 1 steht zum Vergleich."""
    idx = np.arange(1, a.settings.n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=a.f, mode="lines+markers", name="Filterfaktor fᵢ", line=dict(color=C.COLOR_TIK, width=2.5)))
    fig.add_hline(y=0.5, line=dict(color="#888888", dash="dot"), annotation_text="fᵢ = 0,5", annotation_position="top right")
    fig.add_vline(x=a.kstar + 0.5, line=dict(color=C.COLOR_CRIME, dash="dot"), annotation_text=f"k* = {a.kstar}", annotation_position="top left")
    fig.update_xaxes(title="Index i der Komponente")
    fig.update_yaxes(title="Durchlass fᵢ = σᵢ² / (σᵢ² + λ²)", range=[-0.03, 1.05])
    return _fixed(fig, 340)


def build_valley_chart(a: Analysis, reveal: bool) -> go.Figure:
    """Erwarteter Fehler, Bias und Streuung über λ (logarithmisch); der gewählte Parameter, bei „Orakel zeigen“ auch das Optimum."""
    v = a.valley
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v["lam"], y=np.maximum(v["total"], LOG_FLOOR), mode="lines", name="erwarteter Fehler", line=dict(color="#222222", width=3)))
    fig.add_trace(go.Scatter(x=v["lam"], y=np.maximum(v["bias"], LOG_FLOOR), mode="lines", name="Bias", line=dict(color=C.COLOR_BIAS, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=v["lam"], y=np.maximum(v["std"], LOG_FLOOR), mode="lines", name="Streuung", line=dict(color=C.COLOR_NAIVE, width=2, dash="dot")))
    fig.add_vline(x=a.lam, line=dict(color=C.COLOR_TIK, width=2), annotation_text=f"λ = {lam_text(a.lam)}", annotation_position="top left")
    if reveal:
        fig.add_vline(x=a.lam_star, line=dict(color=C.COLOR_TRUTH, width=2, dash="dash"), annotation_text=f"λ* = {lam_text(a.lam_star)}", annotation_position="top right")
    fig.update_xaxes(title="Strafparameter λ", type="log", tickformat=".0e")
    fig.update_yaxes(title="Fehler relativ zur Signalnorm", type="log", range=[-3, 3], tickformat=".0e")
    return _fixed(fig, 380)


def build_factor_chart(res: dict) -> go.Figure:
    rows = res["rows"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["factor"] for r in rows], y=[100.0 * r["closed"] for r in rows], mode="lines+markers", name="geschlossen",
                             line=dict(color=C.COLOR_TIK, width=2.5)))
    fig.add_trace(go.Scatter(x=[r["factor"] for r in rows], y=[100.0 * r["sim"] for r in rows], mode="markers", name=f"simuliert ({res['reps']} Wdh.)",
                             marker=dict(size=9, color=C.COLOR_CRIME, symbol="diamond")))
    fig.add_hline(y=100.0 * res["naive"], line=dict(color=C.COLOR_NAIVE, dash="dash"), annotation_text="direkte Inversion", annotation_position="bottom right")
    fig.update_xaxes(title="Vielfaches des besten Parameters λ*", type="log", tickvals=[r["factor"] for r in rows], ticktext=[de(r["factor"], 2).rstrip("0").rstrip(",") for r in rows])
    ys = [100.0 * r["closed"] for r in rows] + [100.0 * r["sim"] for r in rows] + [100.0 * res["naive"]]
    ticks = [t for t in (10, 20, 50, 100, 200, 500) if min(ys) / 1.4 <= t <= max(ys) * 1.4]
    fig.update_yaxes(title="Erwarteter Fehler (% der Signalnorm)", type="log", tickvals=ticks, ticktext=[str(t) for t in ticks],
                     range=[float(np.log10(min(ys))) - 0.08, float(np.log10(max(ys))) + 0.12])
    return _fixed(fig, 380)


def build_grid_chart(rows: list[dict]) -> go.Figure:
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["direct"] for r in rows], mode="lines+markers", name="direkte Inversion (ehrliche Daten)", line=dict(color=C.COLOR_NAIVE, width=2.5)))
    fig.add_trace(go.Scatter(x=ns, y=[r["tik_honest"] for r in rows], mode="lines+markers", name="Tikhonov, bester Parameter (ehrliche Daten)", line=dict(color=C.COLOR_TIK, width=2.5)))
    fig.add_trace(go.Scatter(x=ns, y=[r["tik_crime"] for r in rows], mode="lines+markers", name="Tikhonov (Daten aus demselben Gitter)", line=dict(color=C.COLOR_CRIME, width=2.5)))
    fig.add_hline(y=1.0, line=dict(color="#888888", dash="dot"), annotation_text="100 % der Signalnorm", annotation_position="bottom right")
    fig.update_xaxes(title="Gitterpunkte n", tickvals=ns)
    fig.update_yaxes(title="Erwarteter relativer Fehler", type="log", tickformat=".0e")
    return _fixed(fig, 400)


def build_noise_chart(rows: list[dict]) -> go.Figure:
    xs = [100.0 * r["noise"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[100.0 * r["err_honest"] for r in rows], mode="lines+markers", name="ehrliche Daten", line=dict(color=C.COLOR_TIK, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[100.0 * r["err_crime"] for r in rows], mode="lines+markers", name="Daten aus demselben Gitter", line=dict(color=C.COLOR_CRIME, width=2.5)))
    fig.update_xaxes(title="Rauschen (% des Effektivwerts der Daten)", type="log", tickvals=xs, ticktext=[noise_label(r["noise"]).replace(" %", "") for r in rows])
    ys = [100.0 * r["err_honest"] for r in rows] + [100.0 * r["err_crime"] for r in rows]
    ticks = [t for t in (0.2, 0.5, 1, 2, 5, 10, 20) if min(ys) / 1.4 <= t <= max(ys) * 1.4]
    fig.update_yaxes(title="Bester erwarteter Fehler (% der Signalnorm)", type="log", tickvals=ticks, ticktext=[de(t, 1).rstrip("0").rstrip(",") for t in ticks],
                     range=[float(np.log10(min(ys))) - 0.1, float(np.log10(max(ys))) + 0.1])
    return _fixed(fig, 380)


def build_edge_chart(res: dict) -> go.Figure:
    s = res["s"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([s, s[::-1]]), y=np.concatenate([res["mean"] + res["sd"], (res["mean"] - res["sd"])[::-1]]), fill="toself",
                             fillcolor="rgba(76,120,168,0.18)", line=dict(width=0), hoverinfo="skip", name="±1 Streuung"))
    fig.add_trace(go.Scatter(x=s, y=res["truth"], mode="lines", name="Wahrheit", line=dict(color=C.COLOR_TRUTH, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=s, y=res["mean"], mode="lines+markers", name="Mittel der Tikhonov-Lösungen", line=dict(color=C.COLOR_TIK, width=2.5), marker=dict(size=5)))
    fig.update_xaxes(title="Ort s")
    fig.update_yaxes(title="Wert")
    return _fixed(fig, 380)
