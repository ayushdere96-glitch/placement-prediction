"""
Placement Predictor - a Streamlit app around a LogisticRegression model.

The model was trained on SCALED data, so every prediction goes through
scaler.transform() first (see predict()). Nothing is saved to disk:
everything lives in st.session_state and disappears with the session.
"""

import html
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent  # files are loaded relative to this script
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "C:\\Users\\ashish\\Downloads\\ml_student_placement_100_rows.csv"

FEATURES = ["IQ", "CGPA"]  # same order as used during training
# Batch uploads: values outside these limits are treated as data errors
# and skipped (they are not just "outside the training range").
HARD_LIMITS = {"IQ": (0.0, 250.0), "CGPA": (0.0, 10.0)}
MAX_TABLE_ROWS = 200  # rows shown on screen; downloads have everything

FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", '
    '"Inter", "Segoe UI", sans-serif'
)
# Inter is only a fallback for non-Apple devices. SF Pro is NOT loaded from
# a CDN (it is not freely licensable); Apple devices use it automatically.
FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&display=swap');"
)

PALETTES = {
    "dark": {
        "bg": "#000000", "surface": "#1C1C1E", "card": "#2C2C2E",
        "field": "#1C1C1E", "text": "#F5F5F7", "muted": "#98989D",
        "border": "rgba(255,255,255,0.10)", "grid": "rgba(255,255,255,0.08)",
        "track": "#3A3A3C", "accent": "#0A84FF", "green": "#30D158",
        "red": "#FF453A", "shadow": "0 1px 2px rgba(0,0,0,0.45)",
    },
    "light": {
        "bg": "#F2F2F7", "surface": "#FFFFFF", "card": "#FFFFFF",
        "field": "#FFFFFF", "text": "#1D1D1F", "muted": "#6E6E73",
        "border": "rgba(0,0,0,0.08)", "grid": "rgba(0,0,0,0.06)",
        "track": "#E5E5EA", "accent": "#007AFF", "green": "#34C759",
        "red": "#FF3B30", "shadow": "0 1px 3px rgba(0,0,0,0.06)",
    },
}

# All colours come from CSS variables (--bg, --card, ...), so switching the
# palette in apply_theme() restyles the whole page at once.
BASE_CSS = """
.stApp, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
.stDeployButton { display: none !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
  background: var(--surface) !important; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
.block-container { max-width: 880px; padding: 2.5rem 1.25rem 4rem; }

.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp label, .stApp button, .stApp input, .stApp table, .stApp summary {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
    "SF Pro Text", "Inter", "Segoe UI", sans-serif !important; }
.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp label, .stApp strong, .stApp summary,
[data-testid="stMetricValue"] { color: var(--text) !important; }
.stApp h3 { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stMetricLabel"] p, .ds-muted { color: var(--muted) !important; }

.ds-title { font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }
.ds-sub { color: var(--muted); font-size: 0.95rem; margin: 0.15rem 0 1.25rem; }

.ds-card { background: var(--card); border: 1px solid var(--border);
  border-radius: 18px; padding: 22px; margin: 0.75rem 0;
  box-shadow: var(--shadow); animation: ds-fade 0.25s ease-out; }
@keyframes ds-fade { from { opacity: 0; } to { opacity: 1; } }
.ds-label { color: var(--muted); font-size: 0.8rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.04em; }
.ds-result { font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em;
  margin: 4px 0 2px; }
.ds-pos { color: var(--green); }
.ds-neg { color: var(--red); }
.ds-prob { color: var(--muted); font-size: 1rem; }
.ds-gauge { position: relative; height: 10px; background: var(--track);
  border-radius: 999px; margin: 18px 0 6px; }
.ds-gauge-fill { height: 100%; border-radius: 999px; }
.ds-fill-pos { background: var(--green); }
.ds-fill-neg { background: var(--red); }
.ds-gauge-mid { position: absolute; left: 50%; top: -4px; width: 2px;
  height: 18px; background: var(--muted); opacity: 0.6; }
.ds-gauge-labels { display: flex; justify-content: space-between;
  font-size: 0.75rem; color: var(--muted); }
.ds-meta { margin-top: 14px; font-size: 0.95rem; }
.ds-note { margin-top: 6px; color: var(--muted); font-size: 0.82rem; }

.ds-notice { background: var(--card); border: 1px solid var(--border);
  border-left: 3px solid var(--accent); border-radius: 14px;
  padding: 12px 16px; font-size: 0.92rem; margin: 0.6rem 0; }

.ds-table-wrap { max-height: 420px; overflow: auto; background: var(--card);
  border: 1px solid var(--border); border-radius: 14px; }
.ds-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.ds-table th { position: sticky; top: 0; background: var(--card);
  color: var(--muted); font-weight: 600; text-align: left !important;
  padding: 10px 14px; border-bottom: 1px solid var(--border); }
.ds-table td { padding: 9px 14px; color: var(--text); white-space: nowrap;
  border-bottom: 1px solid var(--border); }
[data-testid="stSidebar"] .ds-table-wrap { max-height: 300px; }

[data-testid="stPlotlyChart"], [data-testid="stMetric"] {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 18px; box-shadow: var(--shadow); overflow: hidden; }
[data-testid="stPlotlyChart"] { padding: 8px; }
[data-testid="stMetric"] { padding: 14px 16px; }

.stButton > button, .stDownloadButton > button {
  background: var(--card); color: var(--text); font-weight: 500;
  border: 1px solid var(--border); border-radius: 12px; box-shadow: var(--shadow); }
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] { background: var(--accent);
  border-color: var(--accent); }
.stButton > button[kind="primary"] p { color: #FFFFFF !important; }
.stButton > button[kind="primary"]:hover { background: var(--accent);
  color: #FFFFFF; opacity: 0.9; }
[data-testid="stButtonGroup"] button { background: var(--card);
  border: 1px solid var(--border); border-radius: 12px; }
button[kind="segmented_controlActive"] { background: var(--accent) !important;
  border-color: var(--accent) !important; }
button[kind="segmented_controlActive"] p { color: #FFFFFF !important; }

[data-baseweb="input"], [data-baseweb="base-input"],
[data-testid="stFileUploaderDropzone"] { background: var(--field) !important;
  border-color: var(--border) !important; border-radius: 12px !important; }
input, textarea { color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important; background: transparent !important; }
[data-testid="stNumberInput"] button { background: transparent !important;
  color: var(--text) !important; border-color: var(--border) !important; }
[data-testid="stFileUploaderDropzone"] * { color: var(--text) !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--card) !important;
  border: 1px solid var(--border) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stThumbValue"] {
  color: var(--accent) !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
  color: var(--muted) !important; }
[data-testid="stExpander"] { background: var(--card);
  border: 1px solid var(--border) !important; border-radius: 16px;
  overflow: hidden; }
[data-testid="stExpander"] details { border: none !important; }

@media (max-width: 640px) {
  .block-container { padding: 2rem 1rem 3rem; }
  .ds-result { font-size: 1.9rem; }
}
"""

REPORT_CSS = (
    "body{font-family:" + FONT_STACK.replace('"', "'") + ";max-width:560px;"
    "margin:40px auto;padding:0 20px;color:#1D1D1F}"
    "h1{font-size:1.5rem}.big{font-size:2rem;font-weight:700}"
    "table{border-collapse:collapse;width:100%}"
    "td{padding:8px 0;border-bottom:1px solid #E5E5EA}"
    ".muted{color:#6E6E73;font-size:.85rem}"
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    """Load the model and the scaler once and keep them in memory."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def to_binary(series):
    """Turn a Placement-style column into 1 (placed) / 0 (not placed)."""
    if series.dtype == object:
        mapping = {"yes": 1, "no": 0, "1": 1, "0": 0, "placed": 1,
                   "not placed": 0, "true": 1, "false": 0}
        return series.astype(str).str.strip().str.lower().map(mapping)
    return series.astype(int)


@st.cache_data(show_spinner=False)
def load_data():
    """Load the dataset once; adds a numeric 'Placed' column for charts."""
    df = pd.read_csv(DATA_PATH)
    df["Placed"] = to_binary(df["Placement"])
    return df


# --------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------
def positive_index(model):
    """Position of the 'placed' class inside model.classes_."""
    for i, c in enumerate(model.classes_):
        if str(c).strip().lower() in {"1", "yes", "true", "placed"}:
            return i
    return len(model.classes_) - 1


def predict(model, scaler, data):
    """Return P(placed) for every row of `data` (needs IQ and CGPA columns)."""
    scaled = scaler.transform(data[FEATURES])  # ALWAYS scale before predicting
    return model.predict_proba(scaled)[:, positive_index(model)]


def raw_coefficients(model, scaler):
    """
    Convert the model to raw (unscaled) units so we can solve for a CGPA.
    The model works on z = (x - mean) / scale, so:
        logit = b + sum(w * (x - mean) / scale)
              = b_raw + sum(a * x)   with  a = w / scale
                                          b_raw = b - sum(w * mean / scale)
    Returns (w, a, b_raw): scaled weights, raw weights, raw intercept.
    """
    sign = 1 if positive_index(model) == 1 else -1  # coef_ belongs to classes_[1]
    w = model.coef_[0] * sign
    b = model.intercept_[0] * sign
    a = w / scaler.scale_
    b_raw = b - np.sum(w * scaler.mean_ / scaler.scale_)
    return w, a, b_raw


def cgpa_for_probability(a, b_raw, iq, target=0.5):
    """CGPA at which the model gives `target` probability for this IQ."""
    if a[1] <= 0:  # CGPA would not raise the probability: no sensible answer
        return None
    target_logit = float(np.log(target / (1 - target)))
    return (target_logit - b_raw - a[0] * iq) / a[1]


def contributions(w, scaler, iq, cgpa):
    """
    How much each feature pushes the model's score up or down compared with
    an AVERAGE student (an average student has z = 0, so contributes 0).
    """
    z = (np.array([iq, cgpa]) - scaler.mean_) / scaler.scale_
    return w * z


def confidence_label(p):
    """Confidence = how far the probability is from the 0.5 decision line."""
    d = abs(p - 0.5)
    if d < 0.15:
        return "Low"
    if d < 0.30:
        return "Medium"
    return "High"


def band(p):
    """Simple probability band used in batch results."""
    return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"


def data_ranges(df):
    return {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES}


def clamp(value, lo, hi):
    return float(min(max(value, lo), hi))


@st.cache_data(show_spinner=False)
def evaluate(_model, _scaler, df):
    """
    Held-out accuracy (same split as the training script), a confusion matrix,
    and a 5-fold cross-validated accuracy, which is steadier than a 20-row test.
    """
    _, x_test, _, y_test = train_test_split(
        df[FEATURES], df["Placement"], test_size=0.2, random_state=42)
    pred = _model.predict(_scaler.transform(x_test))
    result = {
        "acc": accuracy_score(y_test, pred),
        "n_test": len(x_test),
        "cm": confusion_matrix(to_binary(y_test), to_binary(pd.Series(pred)),
                               labels=[0, 1]),
        "cv_mean": None, "cv_std": None,
    }
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(
            make_pipeline(StandardScaler(), LogisticRegression()),
            df[FEATURES], df["Placed"], cv=cv)
        result["cv_mean"], result["cv_std"] = scores.mean(), scores.std()
    except Exception:
        pass  # cross-validation is a bonus; never break the page for it
    return result


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
def current_palette():
    return PALETTES["dark" if st.session_state.dark_mode else "light"]


def apply_theme(dark):
    p = PALETTES["dark" if dark else "light"]
    variables = ":root{" + "".join(f"--{k}:{v};" for k, v in p.items()) + "}"
    # @import must come first in a stylesheet, then the colour variables.
    st.markdown(f"<style>\n{FONT_IMPORT}\n{variables}\n{BASE_CSS}\n</style>",
                unsafe_allow_html=True)


def notice(text, label="Note"):
    """A calm, theme-aware message box (replaces st.warning / st.error)."""
    st.markdown(f'<div class="ds-notice"><strong>{label}.</strong> {text}</div>',
                unsafe_allow_html=True)


def show_table(df, max_rows=MAX_TABLE_ROWS):
    """Render a DataFrame as themed HTML so it follows the light/dark toggle."""
    table = df.head(max_rows).to_html(index=False, classes="ds-table",
                                      border=0, escape=True)
    st.markdown(f'<div class="ds-table-wrap">{table.replace(chr(10), "")}</div>',
                unsafe_allow_html=True)
    if len(df) > max_rows:
        st.caption(f"Showing the first {max_rows} of {len(df)} rows. "
                   "The download contains every row.")


def rgba(hex_color, alpha):
    """'#RRGGBB' -> 'rgba(r,g,b,alpha)' (used for soft chart backgrounds)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style_figure(fig, pal, height=360):
    """Make a Plotly chart follow the active theme."""
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=pal["card"], plot_bgcolor=pal["card"],
        font=dict(family=FONT_STACK, color=pal["text"], size=13),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=pal["surface"], font_color=pal["text"]),
    )
    fig.update_xaxes(gridcolor=pal["grid"], zeroline=False,
                     linecolor=pal["grid"], tickfont=dict(color=pal["muted"]))
    fig.update_yaxes(gridcolor=pal["grid"], zeroline=False,
                     linecolor=pal["grid"], tickfont=dict(color=pal["muted"]))
    return fig


def show_figure(fig):
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displayModeBar": False})


def contribution_chart(labels, values, pal, axis_title="Log-odds points"):
    """Horizontal bars: right = pushes towards Placed, left = towards Not placed."""
    colors = [pal["green"] if v >= 0 else pal["red"] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig = style_figure(fig, pal, height=210)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=axis_title, zeroline=True, zerolinecolor=pal["muted"])
    return fig


def build_map(df, model, scaler, pal, point=None, heatmap=False, height=420):
    """Scatter of all students + the model's 50% line (+ optional heatmap)."""
    (iq_lo, iq_hi), (cg_lo, cg_hi) = data_ranges(df)["IQ"], data_ranges(df)["CGPA"]
    x_range = [iq_lo - 3, iq_hi + 3]
    y_range = [cg_lo - 0.3, cg_hi + 0.3]
    _, a, b_raw = raw_coefficients(model, scaler)

    fig = go.Figure()
    if heatmap:
        # Colour the whole plane by the model's probability (red -> green).
        xs, ys = np.linspace(*x_range, 60), np.linspace(*y_range, 60)
        gx, gy = np.meshgrid(xs, ys)
        z = predict(model, scaler,
                    pd.DataFrame({"IQ": gx.ravel(), "CGPA": gy.ravel()}))
        fig.add_trace(go.Heatmap(
            x=xs, y=ys, z=z.reshape(gx.shape), zmin=0, zmax=1, zsmooth="best",
            showscale=False, hoverinfo="skip",
            colorscale=[[0, rgba(pal["red"], 0.45)],
                        [0.5, rgba(pal["card"], 1.0)],
                        [1, rgba(pal["green"], 0.45)]]))

    for label, flag, color in (("Not placed", 0, pal["red"]),
                               ("Placed", 1, pal["green"])):
        sub = df[df["Placed"] == flag]
        fig.add_trace(go.Scatter(
            x=sub["IQ"], y=sub["CGPA"], mode="markers", name=label,
            marker=dict(color=color, size=9, opacity=0.9,
                        line=dict(color=pal["card"], width=1)),
            hovertemplate="IQ %{x:.0f}<br>CGPA %{y:.2f}<extra></extra>"))

    # Decision boundary: logit = b_raw + a_iq*IQ + a_cgpa*CGPA = 0.
    if a[1] != 0:
        xs = np.array(x_range)
        fig.add_trace(go.Scatter(
            x=xs, y=-(b_raw + a[0] * xs) / a[1], mode="lines",
            name="50% boundary", hoverinfo="skip",
            line=dict(color=pal["accent"], width=2, dash="dash")))
    if point is not None:
        fig.add_trace(go.Scatter(
            x=[point[0]], y=[point[1]], mode="markers", name="You",
            marker=dict(symbol="diamond", size=16, color=pal["accent"],
                        line=dict(color=pal["text"], width=2))))
    fig = style_figure(fig, pal, height=height)
    fig.update_xaxes(title="IQ", range=x_range)
    fig.update_yaxes(title="CGPA", range=y_range)
    return fig


def build_report(iq, cgpa, p, needed, pct_cgpa, pct_iq):
    """A small standalone HTML summary the student can download or print."""
    outcome = "Placed" if p >= 0.5 else "Not placed"
    needed_txt = f"{needed:.2f}" if needed is not None else "not available"
    rows = [
        ("IQ", f"{iq:.0f}"), ("CGPA", f"{cgpa:.2f}"),
        ("Estimated probability", f"{p * 100:.1f}%"),
        ("Confidence", confidence_label(p)),
        ("CGPA at the 50% line for this IQ", needed_txt),
        ("CGPA higher than", f"{pct_cgpa:.0f}% of dataset"),
        ("IQ higher than", f"{pct_iq:.0f}% of dataset"),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Placement estimate</title><style>{REPORT_CSS}</style></head><body>"
        f"<h1>Placement estimate</h1><div class='big'>{outcome}</div>"
        f"<p class='muted'>Generated {datetime.now():%d %b %Y, %H:%M}</p>"
        f"<table>{body}</table>"
        "<p class='muted'>An estimate from a small educational model trained on "
        "100 student records. It is not a guarantee and should not be used for "
        "real decisions.</p></body></html>")
    return page.encode("utf-8")


# --------------------------------------------------------------------------
# Session state callbacks (they run BEFORE the script reruns, so the page is
# already up to date when it is drawn)
# --------------------------------------------------------------------------
def run_prediction():
    """Called when Predict is pressed: remember the values that were used."""
    st.session_state.prediction = {
        "iq": float(st.session_state.iq_num),
        "cgpa": float(st.session_state.cgpa_num),
    }


def apply_preset(iq, cgpa):
    """Fill both inputs with an example profile and predict straight away."""
    for key, value in (("iq", iq), ("cgpa", cgpa)):
        st.session_state[f"{key}_slider"] = value
        st.session_state[f"{key}_num"] = value
    run_prediction()


def save_to_history():
    model, scaler = load_assets()
    pred = st.session_state.prediction  # the values from the last Predict click
    iq, cgpa = pred["iq"], pred["cgpa"]
    p = float(predict(model, scaler, pd.DataFrame({"IQ": [iq], "CGPA": [cgpa]}))[0])
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "IQ": round(iq, 1),
        "CGPA": round(cgpa, 2),
        "Prediction": "Placed" if p >= 0.5 else "Not placed",
        "Probability (%)": round(p * 100, 1),
    })


def clear_history():
    st.session_state.history = []


def render_history():
    st.markdown("#### Session history")
    if not st.session_state.history:
        st.caption("Nothing saved yet. Use \"Save to session history\" in "
                   "Student mode.")
        return
    show_table(pd.DataFrame(st.session_state.history).iloc[::-1])
    st.button("Clear history", on_click=clear_history)


# Slider <-> number input sync (the number input holds the exact value;
# the slider is clamped to the dataset range).
def slider_to_number(name):
    st.session_state[f"{name}_num"] = st.session_state[f"{name}_slider"]


def number_to_slider(name, lo, hi):
    st.session_state[f"{name}_slider"] = clamp(st.session_state[f"{name}_num"], lo, hi)


# --------------------------------------------------------------------------
# Student mode
# --------------------------------------------------------------------------
def render_student_mode(model, scaler, df):
    pal = current_palette()
    rng = data_ranges(df)
    (iq_lo, iq_hi), (cg_lo, cg_hi) = rng["IQ"], rng["CGPA"]
    w, a, b_raw = raw_coefficients(model, scaler)

    # (state key, label, range, step, slider format, number format, default digits)
    inputs = [
        ("iq", "IQ", rng["IQ"], 1.0, "%.0f", "%.1f", 0),
        ("cgpa", "CGPA", rng["CGPA"], 0.05, "%.2f", "%.2f", 1),
    ]
    # Set starting values once (before the widgets are created).
    for key, _, (lo, hi), _, _, _, digits in inputs:
        start = float(round((lo + hi) / 2, digits))
        st.session_state.setdefault(f"{key}_slider", start)
        st.session_state.setdefault(f"{key}_num", start)

    st.subheader("Your details")

    # ---- Quick example profiles, built from the dataset and the model ----
    q = df[FEATURES].quantile([0.2, 0.5, 0.8])
    mid_iq = float(round(q.loc[0.5, "IQ"]))
    edge = cgpa_for_probability(a, b_raw, mid_iq, 0.5)  # a truly borderline CGPA
    border_cgpa = q.loc[0.5, "CGPA"] if edge is None else edge
    presets = [
        ("Strong profile", q.loc[0.8, "IQ"], q.loc[0.8, "CGPA"]),
        ("Borderline", mid_iq, border_cgpa),
        ("Needs work", q.loc[0.2, "IQ"], q.loc[0.2, "CGPA"]),
    ]
    st.caption("Try an example profile:")
    for col, (name, p_iq, p_cg) in zip(st.columns(3), presets):
        col.button(name, key=f"preset_{name}", use_container_width=True,
                   on_click=apply_preset,
                   args=(clamp(round(p_iq), iq_lo, iq_hi),
                         clamp(round(p_cg, 2), cg_lo, cg_hi)))

    for key, label, (lo, hi), step, slider_fmt, number_fmt, _ in inputs:
        left, right = st.columns([3, 1], vertical_alignment="bottom")
        left.slider(label, min_value=lo, max_value=hi, step=step,
                    format=slider_fmt, key=f"{key}_slider",
                    on_change=slider_to_number, args=(key,))
        right.number_input(f"{label} value", min_value=HARD_LIMITS[label][0],
                           max_value=HARD_LIMITS[label][1], step=step,
                           format=number_fmt, key=f"{key}_num",
                           label_visibility="collapsed",
                           on_change=number_to_slider, args=(key, lo, hi))

    st.button("Predict", type="primary", use_container_width=True,
              on_click=run_prediction)

    # ---- Before the first Predict: show the landscape, not an empty page ----
    pred = st.session_state.get("prediction")
    if pred is None:
        notice("Set your IQ and CGPA (or pick an example), then press Predict.")
        st.subheader("Where students land")
        show_figure(build_map(df, model, scaler, pal, height=380))
        st.caption("Each dot is a student in the dataset. The dashed line is "
                   "where the model's estimate is exactly 50%.")
        return

    # Everything below uses the values from the last Predict click,
    # not the current slider positions.
    iq, cgpa = pred["iq"], pred["cgpa"]
    if iq != float(st.session_state.iq_num) or cgpa != float(st.session_state.cgpa_num):
        notice("You changed the inputs. Press Predict to update the results below.")

    p = float(predict(model, scaler, pd.DataFrame({"IQ": [iq], "CGPA": [cgpa]}))[0])
    placed = p >= 0.5
    pct_cgpa = (df["CGPA"] < cgpa).mean() * 100
    pct_iq = (df["IQ"] < iq).mean() * 100
    needed = cgpa_for_probability(a, b_raw, iq, 0.5)

    # Gentle warning when the input is outside what the model has seen.
    if not (iq_lo <= iq <= iq_hi and cg_lo <= cgpa <= cg_hi):
        notice(f"These values are outside the training data (IQ {iq_lo:.0f} to "
               f"{iq_hi:.0f}, CGPA {cg_lo:.2f} to {cg_hi:.2f}), so the estimate "
               "may be unreliable.", "Heads up")

    # ---- Result card with a simple linear gauge ----
    tone = "pos" if placed else "neg"
    st.markdown(
        '<div class="ds-card">'
        '<div class="ds-label">Estimated outcome</div>'
        f'<div class="ds-result ds-{tone}">{"Placed" if placed else "Not placed"}</div>'
        f'<div class="ds-prob">{p * 100:.1f}% estimated probability of placement</div>'
        f'<div class="ds-gauge"><div class="ds-gauge-fill ds-fill-{tone}" '
        f'style="width:{p * 100:.1f}%"></div><div class="ds-gauge-mid"></div></div>'
        '<div class="ds-gauge-labels"><span>0%</span><span>50%</span><span>100%</span></div>'
        f'<div class="ds-meta">Confidence: <strong>{confidence_label(p)}</strong> '
        '<span class="ds-muted">(how far the estimate is from 50%)</span></div>'
        '<div class="ds-note">This is an estimate based on 100 student records, '
        'not a guarantee.</div>'
        '</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    b1.button("Save to session history", on_click=save_to_history,
              use_container_width=True)
    b2.download_button("Download summary", build_report(
        iq, cgpa, p, needed, pct_cgpa, pct_iq),
        file_name="placement_estimate.html", mime="text/html",
        use_container_width=True)

    # ---- Explainability: what pushes the score up or down ----
    st.subheader("What is driving this estimate")
    show_figure(contribution_chart(FEATURES, contributions(w, scaler, iq, cgpa), pal))
    st.caption("Compared with an average student in the dataset. Bars to the "
               "right push the estimate up, bars to the left push it down. "
               "Units are the model's log-odds points.")

    # ---- Map with the user's position ----
    st.subheader("Where you sit")
    show_figure(build_map(df, model, scaler, pal, point=(iq, cgpa), height=380))
    st.markdown(f"Your CGPA is higher than **{pct_cgpa:.0f}%** of students in the "
                f"dataset, and your IQ is higher than **{pct_iq:.0f}%**.")

    # ---- What-if: probability vs CGPA at the user's IQ ----
    st.subheader("What if your CGPA were higher?")
    bump = 0.5
    cgpa_up = cgpa + bump
    p_up = float(predict(model, scaler,
                         pd.DataFrame({"IQ": [iq], "CGPA": [cgpa_up]}))[0])
    grid = np.linspace(min(cg_lo, cgpa), max(cg_hi, cgpa_up), 80)
    probs = predict(model, scaler, pd.DataFrame({"IQ": iq, "CGPA": grid}))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=probs * 100, mode="lines", name="Estimated probability",
        line=dict(color=pal["accent"], width=3),
        hovertemplate="CGPA %{x:.2f}<br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=50, line=dict(color=pal["muted"], width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=[cgpa], y=[p * 100], mode="markers", name="Now",
        marker=dict(color=pal["accent"], size=12,
                    line=dict(color=pal["card"], width=2))))
    fig.add_trace(go.Scatter(
        x=[cgpa_up], y=[p_up * 100], mode="markers", name=f"CGPA +{bump}",
        marker=dict(color=pal["card"], size=12,
                    line=dict(color=pal["accent"], width=2.5))))
    fig = style_figure(fig, pal)
    fig.update_xaxes(title="CGPA")
    fig.update_yaxes(title="Probability", range=[0, 100], ticksuffix="%")
    show_figure(fig)
    st.markdown(
        f"At IQ {iq:.0f}, raising CGPA from {cgpa:.2f} to {cgpa_up:.2f} moves the "
        f"estimate from **{p * 100:.1f}%** to **{p_up * 100:.1f}%** "
        f"({(p_up - p) * 100:+.1f} points).")
    if cgpa_up > cg_hi:
        st.caption("That CGPA is higher than anyone in the dataset, so treat "
                   "the second number with extra caution.")

    # ---- Target planner (solved from the model, not guessed) ----
    st.subheader("Plan a target")
    target = st.slider("Target probability", 50, 90, 70, 5, format="%d%%",
                       key="target_prob") / 100
    need = cgpa_for_probability(a, b_raw, iq, target)
    if need is None:
        st.markdown("In this model a higher CGPA does not raise the estimate, "
                    "so a CGPA target cannot be worked out.")
    elif need > cg_hi:
        st.markdown(f"At IQ {iq:.0f} the model would need a CGPA above "
                    f"{cg_hi:.2f} (the highest in the dataset) to reach "
                    f"{target:.0%}, so it has little to say about this case.")
    else:
        st.metric(f"CGPA needed for {target:.0%} at IQ {iq:.0f}", f"{need:.2f}",
                  delta=f"{need - cgpa:+.2f} vs yours", delta_color="off")
        if need <= cgpa:
            st.caption("You are already at or above this target.")

    # ---- Honest notes ----
    st.subheader("What the model suggests")
    tips = []
    if needed is not None and needed <= cg_hi:
        tips.append(f"At IQ {iq:.0f}, the model's 50% line sits at a CGPA of "
                    f"about {max(needed, 0):.2f}. "
                    + ("You are above it, but the estimate can still be wrong."
                       if placed else f"That is {needed - cgpa:.2f} above yours."))
    tips.append("The model only sees IQ and CGPA. Skills, projects, "
                "communication and the job market matter too, and it knows "
                "nothing about them.")
    st.markdown("\n".join(f"- {t}" for t in tips))


# --------------------------------------------------------------------------
# Compare mode
# --------------------------------------------------------------------------
def render_compare_mode(model, scaler, df):
    pal = current_palette()
    rng = data_ranges(df)
    (iq_lo, iq_hi), (cg_lo, cg_hi) = rng["IQ"], rng["CGPA"]
    _, a, _ = raw_coefficients(model, scaler)

    st.subheader("Compare two students")
    st.caption("Move the sliders to see which difference matters most.")
    q = df[FEATURES].quantile([0.35, 0.75])
    people = {}
    for col, tag, qi in zip(st.columns(2), ("A", "B"), (0.35, 0.75)):
        with col:
            st.markdown(f"**Student {tag}**")
            iq = st.slider("IQ", iq_lo, iq_hi,
                           clamp(round(q.loc[qi, "IQ"]), iq_lo, iq_hi), 1.0,
                           format="%.0f", key=f"cmp_iq_{tag}")
            cg = st.slider("CGPA", cg_lo, cg_hi,
                           clamp(round(q.loc[qi, "CGPA"], 2), cg_lo, cg_hi), 0.05,
                           format="%.2f", key=f"cmp_cg_{tag}")
            people[tag] = (iq, cg)

    probs = {t: float(predict(model, scaler,
                              pd.DataFrame({"IQ": [v[0]], "CGPA": [v[1]]}))[0])
             for t, v in people.items()}
    for col, tag in zip(st.columns(2), ("A", "B")):
        col.metric(f"Student {tag}", f"{probs[tag] * 100:.1f}%")
        col.caption("Predicted placed" if probs[tag] >= 0.5 else "Predicted not placed")

    fig = go.Figure(go.Bar(
        x=["Student A", "Student B"], y=[probs["A"] * 100, probs["B"] * 100],
        marker_color=pal["accent"], cliponaxis=False, textposition="outside",
        text=[f"{probs['A'] * 100:.1f}%", f"{probs['B'] * 100:.1f}%"]))
    fig.add_hline(y=50, line=dict(color=pal["muted"], width=1, dash="dot"))
    fig = style_figure(fig, pal, height=300)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0, 110], ticksuffix="%")
    show_figure(fig)

    # What explains the gap? Split it into an IQ part and a CGPA part.
    (iq_a, cg_a), (iq_b, cg_b) = people["A"], people["B"]
    d_iq, d_cg = a[0] * (iq_b - iq_a), a[1] * (cg_b - cg_a)
    gap = probs["B"] - probs["A"]
    st.subheader("What explains the gap")
    if abs(gap) < 0.0005:
        st.markdown("The two students get essentially the same estimate.")
    else:
        leader = "B" if gap > 0 else "A"
        driver = "CGPA" if abs(d_cg) > abs(d_iq) else "IQ"
        st.markdown(f"Student {leader} is **{abs(gap) * 100:.1f} percentage "
                    f"points** more likely to be placed, and the bigger driver "
                    f"is the difference in **{driver}**.")
    show_figure(contribution_chart(["IQ difference", "CGPA difference"],
                                   [d_iq, d_cg], pal))
    st.caption("Each bar is how much that difference (B minus A) moves the "
               "model's log-odds score. Right favours Student B, left favours "
               "Student A.")


# --------------------------------------------------------------------------
# Batch mode
# --------------------------------------------------------------------------
def render_batch_mode(model, scaler, df):
    pal = current_palette()
    rng = data_ranges(df)
    st.subheader("Batch predictions")
    st.caption("Upload a CSV with IQ and CGPA columns. Other columns (for "
               "example a name or roll number) are kept in the results.")

    # Sample template, built from typical values in the dataset.
    template = pd.DataFrame({
        "Student": [f"Student {i}" for i in range(1, 6)],
        "IQ": df["IQ"].quantile([0.2, 0.4, 0.5, 0.6, 0.8]).round(0).values,
        "CGPA": df["CGPA"].quantile([0.2, 0.4, 0.5, 0.6, 0.8]).round(2).values,
    })
    st.download_button("Download sample CSV template",
                       template.to_csv(index=False).encode("utf-8"),
                       file_name="placement_template.csv", mime="text/csv")

    use_sample = st.toggle("Try it with built-in sample data (no upload needed)",
                           key="use_sample")
    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception:
            notice("This file could not be read as a CSV. Please save it as a "
                   "plain .csv file and try again.", "Check your file")
            return
    elif use_sample:
        raw = df[FEATURES].sample(min(40, len(df)), random_state=7).reset_index(drop=True)
        raw.insert(0, "Student", [f"Student {i:02d}" for i in range(1, len(raw) + 1)])
        st.caption("These sample rows come from the training dataset, so the "
                   "estimates look better than they would for new students.")
    else:
        notice("Upload a file, or switch on the sample data above.")
        return
    if raw.empty:
        notice("The file has no rows.", "Check your file")
        return

    # Match column names ignoring case and spaces ("iq", " CGPA " also work).
    lookup = {str(c).strip().upper(): c for c in raw.columns}
    missing = [f for f in FEATURES if f not in lookup]
    if missing:
        found = ", ".join(html.escape(str(c)) for c in raw.columns)
        notice(f"Missing column(s): {', '.join(missing)}. "
               f"Columns found: {found}.", "Check your file")
        return
    data = raw.rename(columns={lookup["IQ"]: "IQ", lookup["CGPA"]: "CGPA"})

    # Validate: numbers only, no blanks, values within sensible limits.
    for f in FEATURES:
        data[f] = pd.to_numeric(data[f], errors="coerce")
    bad = data[FEATURES].isna().any(axis=1)
    for f in FEATURES:
        lo, hi = HARD_LIMITS[f]
        bad |= (data[f] < lo) | (data[f] > hi)

    if bad.any():
        rows = ", ".join(str(i + 2) for i in data.index[bad][:10])  # +2: header row
        more = " ..." if bad.sum() > 10 else ""
        notice(f"{int(bad.sum())} row(s) were skipped because IQ or CGPA is "
               "blank, not a number, or outside the allowed range "
               f"(IQ {HARD_LIMITS['IQ'][0]:.0f} to {HARD_LIMITS['IQ'][1]:.0f}, "
               f"CGPA {HARD_LIMITS['CGPA'][0]:.0f} to {HARD_LIMITS['CGPA'][1]:.0f}). "
               f"File row numbers: {rows}{more}", "Heads up")
    results = data[~bad].copy()
    if results.empty:
        notice("No valid rows were left to score.", "Check your file")
        return

    # Recruiters can be stricter (or more lenient) than the model's 50% default.
    threshold = st.slider("Decision threshold", 30, 80, 50, 5, format="%d%%",
                          key="threshold") / 100
    st.caption("50% is the model's default. Raise it to shortlist more strictly.")

    probs = predict(model, scaler, results)
    results["Prediction"] = np.where(probs >= threshold, "Placed", "Not placed")
    results["Probability (%)"] = (probs * 100).round(1)
    results["Band"] = [band(x) for x in probs]

    outside = ((results["IQ"] < rng["IQ"][0]) | (results["IQ"] > rng["IQ"][1]) |
               (results["CGPA"] < rng["CGPA"][0]) | (results["CGPA"] > rng["CGPA"][1]))
    if outside.any():
        results["Note"] = np.where(outside, "Outside training range", "")
        notice(f"{int(outside.sum())} row(s) are outside the range the model was "
               "trained on, so their estimates may be unreliable. They are "
               "marked in the Note column.", "Heads up")

    st.subheader("Summary")
    placed_n = int((probs >= threshold).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students scored", len(results))
    c2.metric("Predicted placed", placed_n)
    c3.metric("Predicted not placed", len(results) - placed_n)
    c4.metric("Average probability", f"{probs.mean() * 100:.1f}%")

    st.subheader("Distribution of estimates")
    fig = go.Figure(go.Histogram(
        x=probs * 100, xbins=dict(start=0, end=100, size=10),
        marker=dict(color=pal["accent"], line=dict(color=pal["card"], width=1)),
        hovertemplate="%{x}%: %{y} students<extra></extra>"))
    fig.add_vline(x=threshold * 100, line=dict(color=pal["muted"], width=1, dash="dot"))
    fig = style_figure(fig, pal, height=300)
    fig.update_layout(showlegend=False, bargap=0.05)
    fig.update_xaxes(title="Estimated probability (%)", range=[0, 100])
    fig.update_yaxes(title="Students")
    show_figure(fig)

    st.subheader("Top 10 candidates")
    show_table(results.sort_values("Probability (%)", ascending=False).head(10))

    st.subheader("All results")
    show_table(results)
    st.download_button("Download results (CSV)",
                       results.to_csv(index=False).encode("utf-8"),
                       file_name="placement_predictions.csv", mime="text/csv")


# --------------------------------------------------------------------------
# Insights mode
# --------------------------------------------------------------------------
def render_insights_mode(model, scaler, df):
    pal = current_palette()
    w, _, _ = raw_coefficients(model, scaler)

    st.subheader("The dataset at a glance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students", len(df))
    k2.metric("Placement rate", f"{df['Placed'].mean() * 100:.0f}%")
    k3.metric("Average IQ", f"{df['IQ'].mean():.0f}")
    k4.metric("Average CGPA", f"{df['CGPA'].mean():.2f}")

    st.subheader("How the model sees the data")
    show_figure(build_map(df, model, scaler, pal, heatmap=True, height=440))
    st.caption("Background colour is the model's estimate (green = likely "
               "placed, red = unlikely). The dashed line is exactly 50%.")

    st.subheader("Model facts")
    ev = evaluate(model, scaler, df)
    m1, m2, m3 = st.columns(3)
    m1.metric("Held-out accuracy", f"{ev['acc'] * 100:.0f}%")
    if ev["cv_mean"] is not None:
        m2.metric("5-fold cross-validated", f"{ev['cv_mean'] * 100:.0f}%",
                  delta=f"± {ev['cv_std'] * 100:.0f} pts", delta_color="off")
    m3.metric("Students", len(df))
    st.caption(f"Held-out accuracy is measured on {ev['n_test']} students the "
               "model did not train on (same 80/20 split and random_state=42 as "
               "the training script). With so few rows it is noisy, so the "
               "cross-validated figure, which tests on every student in turn, "
               "is a steadier guide. Neither is a guarantee.")

    cm = ev["cm"]
    show_table(pd.DataFrame({
        "Held-out students": ["Actually not placed", "Actually placed"],
        "Predicted not placed": [cm[0, 0], cm[1, 0]],
        "Predicted placed": [cm[0, 1], cm[1, 1]],
    }))

    st.subheader("Which feature matters more")
    show_table(pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient (scaled)": np.round(w, 2),
        "Odds multiplier": np.round(np.exp(w), 2),
        "1 std dev equals": np.round(scaler.scale_, 2),
    }))
    top = int(np.argmax(np.abs(w)))
    other = 1 - top
    st.markdown(
        f"In this model **{FEATURES[top]}** matters more. One standard "
        f"deviation of {FEATURES[top]} (about {scaler.scale_[top]:.2f}) multiplies "
        f"the odds of placement by roughly {np.exp(w[top]):.1f}, compared with "
        f"{np.exp(w[other]):.1f} for {FEATURES[other]}. The two are comparable "
        "because the inputs were standardised before training.")

    st.subheader("Placement rate by CGPA band")
    bands = pd.qcut(df["CGPA"], 4, duplicates="drop")
    rate = df.groupby(bands, observed=True)["Placed"].mean() * 100
    fig = go.Figure(go.Bar(
        x=[f"{i.left:.1f} to {i.right:.1f}" for i in rate.index], y=rate.values,
        marker_color=pal["accent"], cliponaxis=False, textposition="outside",
        text=[f"{v:.0f}%" for v in rate.values]))
    fig = style_figure(fig, pal, height=300)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="CGPA band (each holds a quarter of the students)")
    fig.update_yaxes(title="Placed", range=[0, 115], ticksuffix="%")
    show_figure(fig)

    notice("This is a small-data educational model trained on 100 records. "
           "Its weights can shift a lot with a few more or fewer students, so "
           "please do not use it for real admission or hiring decisions.",
           "Please note")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Placement Predictor", layout="centered",
                       initial_sidebar_state="collapsed")
    st.session_state.setdefault("dark_mode", True)  # default to dark
    st.session_state.setdefault("history", [])

    with st.sidebar:
        st.markdown("#### Appearance")
        st.toggle("Dark mode", key="dark_mode")
        render_history()
    apply_theme(st.session_state.dark_mode)

    st.markdown('<div class="ds-title">Placement Predictor</div>'
                '<div class="ds-sub">An estimate based on 100 student records. '
                'Not a guarantee.</div>', unsafe_allow_html=True)

    # Friendly errors for missing files or an incompatible pickle.
    missing = [p.name for p in (MODEL_PATH, SCALER_PATH, DATA_PATH) if not p.exists()]
    if missing:
        notice(f"Missing file(s): {', '.join(missing)}. Put them in the same "
               "folder as app.py and restart the app.", "Setup needed")
        st.stop()
    try:
        model, scaler = load_assets()
        df = load_data()
    except Exception as exc:
        notice("The model files could not be loaded. This usually means "
               "scikit-learn here is a different version from the one used "
               f"for training. Details: {html.escape(str(exc))}", "Setup needed")
        st.stop()

    mode = st.segmented_control(
        "Mode", ["Student", "Compare", "Batch", "Insights"], default="Student",
        key="mode", label_visibility="collapsed") or "Student"

    if mode == "Student":
        render_student_mode(model, scaler, df)
    elif mode == "Compare":
        render_compare_mode(model, scaler, df)
    elif mode == "Batch":
        render_batch_mode(model, scaler, df)
    else:
        render_insights_mode(model, scaler, df)

    with st.expander("About this model"):
        st.markdown(
            "- **Model:** logistic regression using two inputs, IQ and CGPA, "
            "standardised before use.\n"
            "- **Data:** 100 student records, trained on an 80/20 split.\n"
            "- **Output:** a probability of placement. At 50% or above it is "
            "shown as Placed, otherwise Not placed.\n"
            "- **Explanations:** the driver bars show each feature's pull on "
            "the model's score; they describe the model, not real-world causes.\n"
            "- **Limits:** it is an estimate, not a guarantee, and it cannot "
            "see skills, projects or market conditions.\n"
            "- **Privacy:** nothing is stored beyond this browser session; "
            "uploads and history disappear when you close the tab.")


main()