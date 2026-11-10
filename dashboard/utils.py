"""
Shared helpers for the dashboard: file paths, colours, page styling,
cached data loaders and the two forecasting models.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
REPO = DASHBOARD_DIR.parent
DATA = DASHBOARD_DIR / "data"
EDA_FIG = REPO / "EDA_doc" / "figures"
EDA_TAB = REPO / "EDA_doc" / "tables"
MODELING = REPO / "models" / "modeling"
TS_MODEL = REPO / "models" / "time_series" / "model_1"
TS_DATA = REPO / "data" / "processed" / "time_series"
ARM_FINDINGS = REPO / "data" / "processed" / "arm_findings"
ARM_PREP = REPO / "data" / "processed" / "association_rules"

# The team's time-series model code lives in models/time_series/model_1/tsa.
sys.path.insert(0, str(TS_MODEL))
from tsa.baselines import adaptive_week, seasonal_mean, seasonal_naive, tsb  # noqa: E402

# ---------------------------------------------------------------------------
# Colours
# Chart palette: fixed order, checked for colour-blind safety (the first three
# also stay distinct on maps, where any two zones can touch).
# ---------------------------------------------------------------------------
INDIGO, AMBER, TEAL, COPPER = "#3456c4", "#d08c00", "#00917f", "#b8622a"
VIOLET, ROSE, SKY, OLIVE = "#8a63c9", "#d2475e", "#3f97d6", "#6f8f00"
SERIES = [INDIGO, AMBER, TEAL, COPPER, VIOLET, ROSE, SKY, OLIVE]
YELLOW, GREEN = AMBER, TEAL  # Yellow and Green taxi
GRAY = "#7c8798"
INK, INK_2, INK_3 = "#0f172a", "#475569", "#64748b"
NAVY = "#0f1b33"
LINE = "#e2e8f0"
SEQ = ["#e3e9f8", "#bccbf0", "#93a9e5", "#6a87d8", "#4566c9", "#2f4ea6", "#1f3577"]  # light -> dark indigo

# Fixed colours per entity so the same thing always has the same colour.
MODEL_COLORS = {
    "Naive 168-hour baseline": GRAY,
    "Linear Regression": INDIGO,
    "Random Forest": TEAL,
    "HistGradientBoosting": AMBER,
}
DEMAND_COLORS = {"High": AMBER, "Medium": INDIGO, "Low": TEAL}
DEMAND_PALE = {"High": "#f5dfb3", "Medium": "#c9d4f2", "Low": "#b3e0d9"}  # light versions of the above
CLUSTER_COLORS = {
    "Core business and airports (daytime)": INDIGO,
    "Nightlife (late night and weekends)": AMBER,
    "Residential (morning commute)": TEAL,
    "Near-empty (not clustered)": "#cbd5e1",
}


def tint(hex_color, alpha):
    """A see-through version of a colour, e.g. for shaded areas."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


# Plotly template used by every chart.
pio.templates["taxi"] = go.layout.Template(
    layout=dict(
        colorway=SERIES,
        font=dict(family="Inter, -apple-system, 'Segoe UI', Roboto, sans-serif", size=13, color=INK_2),
        title=dict(font=dict(size=15, color=INK, weight=600), x=0, xanchor="left", y=1, yref="container",
                   yanchor="top", pad=dict(t=12, l=4)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, linecolor="#cbd5e1", ticks="outside", tickcolor="#cbd5e1", ticklen=4,
                   zeroline=False, automargin=True, title_font=dict(size=12, color=INK_3)),
        yaxis=dict(gridcolor="#edf1f7", zeroline=False, linecolor="rgba(0,0,0,0)", automargin=True,
                   title_font=dict(size=12, color=INK_3)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="",
                    font=dict(size=12, color=INK_2)),
        hoverlabel=dict(bgcolor="white", font=dict(size=12, color=INK), bordercolor=LINE),
        margin=dict(l=10, r=10, t=80, b=10),
        bargap=0.3,
        modebar=dict(bgcolor="rgba(0,0,0,0)", color="#94a3b8", activecolor=INDIGO),
    )
)
pio.templates.default = "taxi"

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TIME_BANDS = {
    "late_night": "Late night (00–04)",
    "early_morning": "Early morning (05–06)",
    "am_rush": "AM rush (07–09)",
    "midday": "Midday (10–15)",
    "pm_rush": "PM rush (16–19)",
    "evening": "Evening (20–23)",
}

# ---------------------------------------------------------------------------
# Page styling and small layout components
# ---------------------------------------------------------------------------
CSS = """
<style>
:root {
  --ink: #0f172a; --ink-2: #475569; --ink-3: #64748b; --line: #e2e8f0; --soft: #f6f8fb;
  --indigo: #3456c4; --amber: #d08c00; --teal: #00917f; --navy: #0f1b33;
  --shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 2px 8px rgba(15, 23, 42, .04);
}
[data-testid="stMainBlockContainer"] {padding-top: 3.6rem !important; padding-bottom: 3rem !important; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.015em; color: var(--ink);}
[data-testid="stMain"] h2 {font-size: 1.4rem; font-weight: 650; margin-top: 1.4rem;}
[data-testid="stMain"] h2::before {
  content: ""; display: block; width: 28px; height: 3px; border-radius: 2px;
  background: var(--amber); margin-bottom: .55rem;
}
[data-testid="stMain"] h3 {font-size: 1.1rem; font-weight: 650;}
[data-testid="stCaptionContainer"] {color: var(--ink-3) !important; opacity: 1 !important;}
hr {border-color: var(--line) !important;}

/* Page header */
.hero {
  position: relative; overflow: hidden; background: var(--navy); color: #fff;
  border-radius: 16px; padding: 1.7rem 2rem 2.1rem 2rem; margin-bottom: 1.4rem;
}
.hero::after {   /* taxi checker strip */
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 8px;
  background: repeating-linear-gradient(90deg, var(--amber) 0 8px, transparent 8px 16px) top / 100% 4px no-repeat,
              repeating-linear-gradient(90deg, transparent 0 8px, var(--amber) 8px 16px) bottom / 100% 4px no-repeat;
  opacity: .9;
}
.hero .kicker {
  display: inline-block; text-transform: uppercase; font-size: .72rem; font-weight: 600;
  letter-spacing: .12em; color: #f5c35b;
}
.hero h1 {color: #fff; margin: .35rem 0 .5rem 0; font-size: 2.05rem; font-weight: 700; padding: 0;}
.hero p {margin: 0; color: #c8d2e3; font-size: 1.02rem; line-height: 1.6; max-width: 920px;}
.hero b {color: #fff;}

/* Cards */
.kpi {
  background: #fff; border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow);
  padding: .95rem 1.1rem; min-height: 122px; border-top: 3px solid var(--indigo);
}
.kpi .label {font-size: .7rem; font-weight: 600; color: var(--ink-3); text-transform: uppercase; letter-spacing: .08em;}
.kpi .value {font-size: 1.7rem; font-weight: 700; color: var(--ink); line-height: 1.3; margin-top: .15rem;}
.kpi .note {font-size: .8rem; color: var(--ink-3);}
.callout {
  background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--indigo);
  border-radius: 10px; box-shadow: var(--shadow); padding: .85rem 1.1rem; margin: .5rem 0 1rem 0;
  color: var(--ink-2); line-height: 1.6;
}
.callout.use {border-left-color: var(--teal); background: #f4fbf9;}
.callout.warn {border-left-color: var(--amber); background: #fffaf0;}
.callout b {color: var(--ink);}
.step {
  background: #fff; border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow);
  padding: 1rem 1.1rem; min-height: 148px;
}
.step .num {
  display: inline-flex; align-items: center; justify-content: center; width: 1.9rem; height: 1.9rem;
  border-radius: 8px; background: #e8edfb; color: var(--indigo); font-weight: 700; font-size: .85rem;
  margin-bottom: .55rem;
}
.step .title {font-weight: 650; color: var(--ink); margin-bottom: .2rem;}
.step .body {font-size: .88rem; color: var(--ink-2); line-height: 1.55;}
.big-number {font-size: 3rem; font-weight: 800; color: var(--ink); line-height: 1.1;}
.pill {
  display: inline-block; padding: .2rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
  margin: 0 .3rem .3rem 0; background: #eef2fb; color: #2f4ea6; border: 1px solid #dbe3f7;
}

/* Streamlit elements as cards */
[data-testid="stPlotlyChart"] {
  background: #fff; border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow);
  padding: 0 .5rem; overflow: hidden;
}
[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden; box-shadow: var(--shadow);}
[data-testid="stExpander"] details {border-radius: 12px; border-color: var(--line); background: #fff;}
[data-testid="stImage"] img {border-radius: 8px; border: 1px solid var(--line);}
[data-testid="stTabs"] button[role="tab"] p {font-weight: 600;}

/* Sidebar */
[data-testid="stSidebar"] {border-right: none;}
[data-testid="stSidebarNav"] a {border-radius: 8px;}
.sidebar-brand {padding: .2rem 0 .6rem 0;}
.sidebar-brand .name {color: #fff; font-weight: 700; font-size: 1.02rem; letter-spacing: -0.01em;}
.sidebar-brand .sub {color: #93a3bf; font-size: .78rem; line-height: 1.45; margin-top: .2rem;}
.sidebar-brand .bar {width: 32px; height: 3px; background: #f5c35b; border-radius: 2px; margin-bottom: .6rem;}
</style>
"""


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(kicker, title, text):
    st.markdown(
        f'<div class="hero"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    """items: list of (label, value, note) tuples shown as cards in one row."""
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        column.markdown(
            f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="note">{note}</div></div>',
            unsafe_allow_html=True,
        )


def callout(text, kind="info"):
    """kind: 'info' (indigo), 'use' (teal, how it helps users) or 'warn' (amber)."""
    css_class = {"info": "callout", "use": "callout use", "warn": "callout warn"}[kind]
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def steps(items):
    """items: list of (title, body) shown as numbered cards in one row."""
    columns = st.columns(len(items))
    for i, (column, (title, body)) in enumerate(zip(columns, items), start=1):
        column.markdown(
            f'<div class="step"><div class="num">{i:02d}</div><div class="title">{title}</div>'
            f'<div class="body">{body}</div></div>',
            unsafe_allow_html=True,
        )


def show_chart(fig, height=None, toolbar=True):
    fig.update_layout(legend_title_text="")  # the chart title already names what is shown
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False, "displayModeBar": "hover" if toolbar else False})


def figure(path, caption):
    """Show one of the saved analysis figures from the repository."""
    if Path(path).exists():
        st.image(str(path), caption=caption, width="stretch")



# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
def check_data_built():
    if not (DATA / "zones.csv").exists():
        st.error("Dashboard data not found. Run `python dashboard/build_data.py` from the repository root first.")
        st.stop()


@st.cache_data
def read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


@st.cache_data
def load_zones():
    return pd.read_csv(DATA / "zones.csv").sort_values("Zone").reset_index(drop=True)


@st.cache_data
def load_features():
    """The final modelling table: one row per zone and hour (8 Sep 2025 - 28 Feb 2026)."""
    return pd.read_csv(REPO / "data" / "processed" / "features.csv", parse_dates=["pickup_hour"])


@st.cache_data
def load_test_predictions():
    return pd.read_csv(DATA / "test_predictions.csv", parse_dates=["pickup_hour"])


@st.cache_data
def load_zone_hourly():
    """Hourly pickups, one column per zone (column names are zone IDs as text)."""
    return pd.read_csv(DATA / "zone_hourly_wide.csv", index_col=0, parse_dates=True)


@st.cache_data
def load_aggregate_hourly():
    return pd.read_csv(DATA / "aggregate_hourly.csv", parse_dates=["ds"])


@st.cache_data
def load_zone_shapes():
    """Official taxi-zone outlines (saved by build_data.py)."""
    return json.loads((DATA / "taxi_zones.geojson").read_text())


def zone_map(frame, color, color_map=None, hover_data=None, order=None, title="", labels=None):
    """Draw the taxi zones as shapes coloured by one column. Needs no map tiles or internet."""
    fig = px.choropleth(
        frame, geojson=load_zone_shapes(), locations="PULocationID", featureidkey="properties.id",
        color=color, color_discrete_map=color_map, category_orders={color: order} if order else None,
        hover_name="Zone", hover_data=hover_data, title=title, labels=labels,
    )
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
    fig.update_traces(marker_line_color="white", marker_line_width=0.6)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), dragmode=False)
    return fig


@st.cache_data
def load_rules():
    rules = pd.read_csv(ARM_FINDINGS / "arm_rules_final_non_redundant.csv")
    rules["demand"] = rules["consequent_text"].str.replace("demand=", "", regex=False)
    rules["conditions"] = rules["antecedent_text"].str.split(", ")
    return rules


# ---------------------------------------------------------------------------
# Model 3 (HistGradientBoosting) - the best prediction model from Review 1
# ---------------------------------------------------------------------------
MODEL_FEATURES = [
    "PULocationID", "hour", "day_of_week", "month",
    "apparent_temperature", "rain", "snowfall", "precipitation_probability", "is_holiday",
    "trip_count_lag_1h", "trip_count_lag_24h", "trip_count_lag_168h",
    "trip_count_roll_3h", "trip_count_roll_24h",
]


@st.cache_resource
def load_hgb_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # saved with scikit-learn 1.9.0; predictions are identical
        return joblib.load(MODELING / "model_3_hist_gradient_boosting.joblib")


def predict_demand(rows):
    """Predict hourly pickups for a DataFrame that has the 14 model features."""
    model = load_hgb_model()
    prediction = model.predict(rows[MODEL_FEATURES])
    return np.clip(prediction, 0, None)


@st.cache_data
def load_zone_normal():
    """Average pickups per zone for each weekday and hour in the training period (no holidays)."""
    return pd.read_csv(DATA / "zone_normal.csv")


LEVEL_MEANING = {"High": "busier than usual", "Medium": "about usual", "Low": "quieter than usual"}


def compare_with_normal(predicted, normal):
    """High / Medium / Low compared with the zone's normal demand for the same weekday and hour.
    Within ±15% of normal counts as Medium. Adding 1 to both sides stops near-empty zones
    from flipping on one or two pickups."""
    ratio = (predicted + 1) / (normal + 1)
    if ratio >= 1.15:
        return "High"
    if ratio <= 0.85:
        return "Low"
    return "Medium"


def demand_tier(value, zone_row):
    """Low / Medium / High compared with the zone's own training history (ARM terciles)."""
    if not zone_row["arm_active"]:
        return "Low"
    if value <= zone_row["q_low"]:
        return "Low"
    if value <= zone_row["q_high"]:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Time-series Model 1 - the four forecasting methods (team code in tsa/)
# ---------------------------------------------------------------------------
METHOD_NAMES = {
    "adaptive_week": "Adaptive seasonal",
    "mean_4weeks": "Four-week mean",
    "tsb": "TSB (intermittent)",
    "naive_168": "Weekly naive",
}


def run_method(method, history, horizon):
    """Forecast `horizon` hours from a 1-D history array using one Model 1 method."""
    y = np.asarray(history, dtype=float)
    if method == "adaptive_week":
        return adaptive_week(y, horizon)
    if method == "mean_4weeks":
        return seasonal_mean(y, horizon)
    if method == "naive_168":
        return seasonal_naive(y, horizon, 168)
    if method == "tsb":
        return tsb(y, horizon)[:, 0]
    raise ValueError(method)


def forecast_bottom_up(zone_hourly, zones, origin, horizon, zone_ids=None):
    """Forecast every chosen zone with its selected method and add them up."""
    history = zone_hourly[zone_hourly.index < origin]
    method_of = dict(zip(zones["PULocationID"].astype(str), zones["ts_method"]))
    columns = zone_ids if zone_ids is not None else list(zone_hourly.columns)
    total = np.zeros(horizon)
    for zone_id in columns:
        total += run_method(method_of[zone_id], history[zone_id].values, horizon)
    return total


def lead_bucket(lead, horizon):
    """Lead-time groups used for the calibrated prediction intervals."""
    if horizon == 24:
        return "01-06" if lead <= 6 else ("07-12" if lead <= 12 else "13-24")
    return "001-024" if lead <= 24 else ("025-072" if lead <= 72 else "073-168")


# ---------------------------------------------------------------------------
# Association rules - plain-English wording
# ---------------------------------------------------------------------------
ITEM_TEXT = {
    "day_type=weekday": "weekday", "day_type=weekend": "weekend",
    "holiday=regular": "regular day", "holiday=federal": "federal holiday",
    "rain=no": "no rain", "rain=yes": "raining",
    "snow=no": "no snow", "snow=yes": "snowing",
    "temperature=lower": "cold (< −0.9 °C)", "temperature=middle": "mild (−0.9 to 9.9 °C)",
    "temperature=upper": "warmer (> 9.9 °C)",
    "zone_cluster=daytime": "daytime/business zone", "zone_cluster=morning_commute": "commuter zone",
    "zone_cluster=nightlife": "nightlife zone", "zone_cluster=near_empty": "near-empty zone",
}
for _key, _label in TIME_BANDS.items():
    ITEM_TEXT[f"time_band={_key}"] = _label


def describe_rule(conditions):
    """['time_band=am_rush', 'day_type=weekday'] -> 'AM rush (07–09) + weekday'"""
    return " + ".join(ITEM_TEXT.get(item, item) for item in conditions)
