from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (INDIGO, GRAY, INK, AMBER, REPO, TS_DATA, TS_MODEL, callout,
                   figure, hero, kpi_row, read_csv, show_chart, steps)

TABLES = TS_MODEL / "tables"
FIGS = TS_MODEL / "figures"
ASSETS = Path(__file__).resolve().parent.parent / "assets"  # images saved with the dashboard

hero(
    "Chapter 4 · Review 2",
    "Forecasting a whole day or week ahead",
    "The ML model needs the actual demand of the previous hour, so it only sees one hour ahead. "
    "Shift rosters and fleet plans need <b>24-hour and 7-day forecasts</b> issued before any of those "
    "hours are known. That is a time-series problem.",
)

# ---------------------------------------------------------------------------
st.header("Preparing a complete time series")
steps([
    ("Restore 1–7 September", "features.csv starts on 8 Sep. 23,184 zone-hours were recovered exactly from the 1-week lag column."),
    ("Complete zone-hour grid", "138 zones × 4,344 hours = 599,472 rows. True zero hours stay zero; no hour is missing or repeated."),
    ("Handle daylight saving", "01:00 on 2 Nov happened twice. trip_count_adj halves only that hour (pickups per physical hour)."),
    ("Modelling levels", "Hourly and daily series for each zone plus six totals: all zones, Manhattan, Queens, Yellow Zone, Boro Zone, Airports."),
])
st.caption("Check: all levels add up to exactly 22,579,793 cleaned trips. Training-only zone profiles: "
           "83 regular, 53 intermittent (many zero hours) and 2 zero-demand zones.")

st.header("What the demand history showed")
acf = read_csv(TS_DATA / "ts_acf_key_lags.csv")
strength = read_csv(TABLES / "seasonal_strength.csv")
kpi_row([
    ("Daily seasonality", "0.94", "strength, all zones (0 – 1)"),
    ("Weekly seasonality", "0.76", "strength, all zones"),
    ("Similarity 24 h apart", "0.88", "autocorrelation"),
    ("Similarity 168 h apart", "0.89", "autocorrelation"),
])
st.write("")
left, right = st.columns(2)
with left:
    fig = px.bar(acf, x="lag_hours", y="autocorrelation", text="autocorrelation",
                 labels={"lag_hours": "Lag (hours)", "autocorrelation": "Autocorrelation"},
                 title="Demand repeats every day and every week")
    fig.update_xaxes(type="category")
    fig.update_traces(marker_color=[AMBER if lag in (24, 168) else INDIGO for lag in acf["lag_hours"]],
                      texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    show_chart(fig, height=340)
with right:
    strength = strength[strength["component"] != "trend"]
    strength["component"] = strength["component"].str.title()
    fig = px.bar(strength, x="series_id", y="strength", color="component", barmode="group",
                 color_discrete_map={"Daily": INDIGO, "Weekly": AMBER},
                 labels={"series_id": "", "strength": "Seasonal strength", "component": ""},
                 title="Seasonal strength by area")
    fig.update_traces(marker_line_color="white", marker_line_width=2)
    show_chart(fig, height=340)
callout("Busy and quiet hours repeat each day and week, so <b>seasonal profiles</b> are a natural "
        "base for forecasting. 53 zones have long gaps between pickups, so an <b>intermittent-demand</b> "
        "method (TSB) was added. Comparing each hour with the same hour last week removed the weekly "
        "pattern, and two stationarity tests (ADF, KPSS) agreed on the result.")

# ---------------------------------------------------------------------------
st.header("Model 1: four forecasting methods, chosen zone by zone")
st.caption("Each zone uses whichever method had the lowest error across four earlier weekly tests. "
           "F(h) = predicted pickups h hours ahead; y₁…y₄ = the same weekday and hour 1–4 weeks earlier.")
m1, m2, m3, m4 = st.columns(4)
with m1:
    with st.container(border=True, height=330):
        st.markdown("**1 · Adaptive seasonal** – 88 zones")
        st.latex(r"F(h)=B(h)\,[1+(r-1)e^{-h/48}]")
        st.latex(r"r=\mathrm{clip}\!\left(\tfrac{L+1}{E+1},0.5,1.5\right)")
        st.caption("The weekly pattern B(h), raised or lowered by recent demand. L = latest 24-hour mean, "
                   "E = the same window in the four prior weeks. The adjustment fades over 48 h.")
with m2:
    with st.container(border=True, height=330):
        st.markdown("**2 · Four-week mean** – 25 zones")
        st.latex(r"F(h)=\frac{y_1+y_2+y_3+y_4}{4}")
        st.caption("Average of the matching hour in the last four weeks. Smooths out one unusual week.")
with m3:
    with st.container(border=True, height=330):
        st.markdown("**3 · TSB** – 17 zones")
        st.latex(r"F(h)=p\times z")
        st.caption("For intermittent zones with many zero hours. p = probability of any pickup "
                   "(updated each hour, β = 0.05); z = average count when pickups happen (α = 0.10).")
with m4:
    with st.container(border=True, height=330):
        st.markdown("**4 · Weekly naive** – 8 zones")
        st.latex(r"F(h)=y_1")
        st.caption("Repeat the same weekday and hour from last week. Also the baseline every method must beat.")

# ---------------------------------------------------------------------------
st.header("Forecast results on the test period (25 Jan – 28 Feb 2026)")
results = pd.DataFrame({
    "Evaluation level": ["Each zone, each hour", "City total, forecast directly", "City total, sum of zone forecasts"],
    "Last-week forecast MAE": [9.31, 777.71, 777.71],
    "Model 1 MAE": [7.82, 678.21, 649.37],
})
results["Less error"] = 100 * (1 - results["Model 1 MAE"] / results["Last-week forecast MAE"])
left, right = st.columns([1, 1.15])
with left:
    st.dataframe(results, hide_index=True, width="stretch",
                 column_config={"Less error": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=25)})
    callout("Adding up the 138 zone forecasts gives the <b>most accurate city total</b>: 16.5% lower "
            "error than repeating last week. For the city total, 84% of actual "
            "values fell inside the calibrated 80% prediction band.")
with right:
    improvement = read_csv(TABLES / "bottom_up_improvement.csv")
    improvement["Horizon"] = improvement["schedule"].map({24: "Day-ahead (24 h)", 168: "Week-ahead (168 h)"})
    fig = px.bar(improvement, x="series_id", y="MAE_reduction_pct", color="Horizon", barmode="group",
                 color_discrete_map={"Day-ahead (24 h)": INDIGO, "Week-ahead (168 h)": AMBER},
                 labels={"series_id": "", "MAE_reduction_pct": "Error reduction vs last week (%)"},
                 title="Model 1 beats the weekly baseline in every area")
    fig.update_traces(marker_line_color="white", marker_line_width=2, hovertemplate="%{x}<br>%{y:.1f}% less error<extra></extra>")
    show_chart(fig, height=330)

st.subheader("Actual vs forecast")
predictions = pd.read_csv(TABLES / "bottom_up_predictions.csv.gz", parse_dates=["ds", "origin"])
col_a, col_b, col_c = st.columns(3)
series = col_a.selectbox("Area", ["All zones", "Manhattan", "Queens", "Yellow Zone", "Boro Zone", "Airports"])
horizon = col_b.segmented_control("Forecast issued", ["Day-ahead (24 h)", "Week-ahead (168 h)"], default="Day-ahead (24 h)", required=True)
weeks = pd.date_range("2026-01-25", "2026-02-22", freq="7D")
week = col_c.selectbox("Week", weeks, index=4, format_func=lambda d: f"{d:%d %b} – {d + pd.Timedelta(days=6):%d %b %Y}")
schedule = 24 if horizon.startswith("Day") else 168
view = predictions[(predictions["series_id"] == series) & (predictions["schedule"] == schedule)
                   & (predictions["ds"] >= week + pd.Timedelta(hours=4)) & (predictions["ds"] < week + pd.Timedelta(days=7, hours=4))]
fig = go.Figure()
fig.add_scatter(x=view["ds"], y=view["actual"], name="Actual", line=dict(color=INK, width=2))
fig.add_scatter(x=view["ds"], y=view["baseline"], name="Last-week forecast", line=dict(color=GRAY, width=1.5, dash="dot"))
fig.add_scatter(x=view["ds"], y=view["prediction"], name="Model 1 (sum of zones)", line=dict(color=AMBER, width=2))
fig.update_layout(hovermode="x unified", yaxis_title="Pickups per hour", title=f"{series}: hourly pickups")
show_chart(fig, height=400)
if len(view):
    mae_model = (view["prediction"] - view["actual"]).abs().mean()
    mae_base = (view["baseline"] - view["actual"]).abs().mean()
    reduction = 100 * (1 - mae_model / mae_base)
    st.caption(f"This week: Model 1 MAE **{mae_model:,.1f}** vs last-week forecast **{mae_base:,.1f}** "
               f"({abs(reduction):.1f}% {'less' if reduction >= 0 else 'more'} error). Single weeks vary a lot "
               "(from 20% worse to 55% better); over the whole test period Model 1 has 16.5% less error.")

callout("<b>Limitation:</b> during the 22–23 Feb blizzard, the error of the direct city-total forecast was "
        "about <b>four times higher</b> (MAE 2,181 vs 545 per hour). A seasonal forecast cannot see a "
        "storm coming, so weather alerts must still override it. Only six months of data were available.", "warn")

# ---------------------------------------------------------------------------
st.header("Other time-series models we tested")
st.caption("Teammate models fitted per zone on a 70 : 15 : 15 chronological split (a different test window), "
           "so their numbers are compared with their own weekly-naive baseline, not with Model 1 directly.")
dhr = read_csv(REPO / "timeSeries" / "results" / "model_2" / "Split751515" / "metrics.csv")
mstl = read_csv(REPO / "timeSeries" / "results" / "model_3" / "metrics.csv")
other = pd.concat([dhr, mstl])
other["model"] = other["model"].replace({"DynamicHarmonicRegression": "Dynamic Harmonic Regression (ARIMA + Fourier)",
                                         "MSTL_ETS": "MSTL + ETS", "SeasonalNaive168": "Weekly naive (same split)"})
left, right = st.columns([1.2, 1])
with left:
    st.dataframe(other[["split", "model", "MAE", "RMSE", "R2"]].sort_values(["split", "MAE"]),
                 hide_index=True, width="stretch",
                 column_config={"MAE": st.column_config.NumberColumn(format="%.2f"),
                                "RMSE": st.column_config.NumberColumn(format="%.2f"),
                                "R2": st.column_config.NumberColumn("R²", format="%.3f")})
    callout("Fitting heavier models (Fourier + ARIMA errors, MSTL + ETS) to 138 noisy zone series did "
            "<b>not</b> beat the simple weekly naive forecast on the test window. That is why the final "
            "Model 1 uses robust seasonal profiles chosen zone by zone.")
with right:
    folds = read_csv(REPO / "timeSeries" / "results" / "model_2_rolling" / "rolling_origin_metrics.csv")
    folds["Week"] = pd.to_datetime(folds["forecast_start"]).dt.strftime("%d %b")
    fig = px.bar(folds, x="Week", y="MAE", text="MAE", title="DHR rolling-origin test: the blizzard week fails",
                 labels={"Week": "Forecast week starting", "MAE": "MAE (pickups)"})
    fig.update_traces(marker_color=[AMBER if f == 3 else INDIGO for f in folds["fold"]],
                      texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    show_chart(fig, height=330)
    st.caption("Fold 3 (16–23 Feb) covers the 22–23 Feb blizzard and travel ban: daily MAE rose from 13.7 to 25.5.")

with st.expander("Original analysis figures from the time-series notebook"):
    gallery = [
        ("02_seasonal_profile_decomposition.png", "Seasonal profile decomposition"),
        ("04_weekly_heatmaps.png", "Weekly demand heatmaps"),
        ("07_forecast_intervals.png", "Forecast with prediction intervals"),
        ("11_event_stress_test.png", "Event stress test"),
        ("10_zone_errors.png", "Zone-level forecast errors"),
        ("12_next_week_forecast.png", "Next-week forecast (1–7 Mar 2026)"),
    ]
    for i in range(0, len(gallery), 2):
        cols = st.columns(2)
        for col, (file_name, caption) in zip(cols, gallery[i:i + 2]):
            with col:
                figure(FIGS / file_name, caption)

    # Teammate Model 2 (Dynamic Harmonic Regression), shown full width because the charts are wide.
    figure(ASSETS / "model_2_dhr_actual_vs_predicted.png", "Model 2 (DHR): actual vs predicted demand, test period")
    figure(ASSETS / "model_2_dhr_anomaly.png", "Model 2 (DHR): actual vs predicted around the 23 Feb blizzard")

callout("<b>How it helps users:</b> managers can plan driver rosters and maintenance 1–7 days ahead from "
        "the forecast instead of reacting day by day. Generate your own forecast in <b>Forecast planner</b>.", "use")
