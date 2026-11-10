import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (TEAL, INDIGO, GRAY, INK, METHOD_NAMES, AMBER, COPPER, TS_MODEL, callout, forecast_bottom_up,
                   hero, kpi_row, lead_bucket, load_zone_hourly, load_zones, read_csv, run_method, show_chart, tint)

hero(
    "Try it live · Time series",
    "Forecast planner",
    "Generate a day-ahead or week-ahead forecast with our time-series Model 1. Only the history "
    "<b>before</b> the forecast start is used, just like in real planning. Choose a start inside the data "
    "to compare with what actually happened, or <b>1 March 2026</b> to forecast the unseen next week.",
)

zones = load_zones()
hourly = load_zone_hourly()
last_hour = hourly.index.max()

AREAS = {
    "All zones": zones,
    "Manhattan": zones[zones["Borough"] == "Manhattan"],
    "Queens": zones[zones["Borough"] == "Queens"],
    "Yellow Zone": zones[zones["service_zone"] == "Yellow Zone"],
    "Boro Zone": zones[zones["service_zone"] == "Boro Zone"],
    "Airports": zones[zones["service_zone"] == "Airports"],
}

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
col_level, col_target, col_start, col_hour, col_horizon = st.columns([1, 1.6, 1, 0.8, 1.2])
level = col_level.radio("Forecast for", ["Area total", "Single zone"])
if level == "Area total":
    area = col_target.selectbox("Area", list(AREAS))
    members = AREAS[area]
    title = area
else:
    default_zone = int(zones.index[zones["Zone"] == "JFK Airport"][0])
    zone_label = col_target.selectbox("Zone", zones["label"], index=default_zone)
    members = zones[zones["label"] == zone_label]
    title = members["Zone"].iloc[0]
start_date = col_start.date_input("Forecast start", dt.date(2026, 2, 15), min_value=dt.date(2025, 9, 29), max_value=dt.date(2026, 3, 1))
start_hour = col_hour.selectbox("Start hour", list(range(24)), index=4, format_func=lambda h: f"{h:02d}:00")
horizon_label = col_horizon.segmented_control("Horizon", ["24 hours", "7 days"], default="7 days", required=True)
horizon = 24 if horizon_label == "24 hours" else 168

origin = pd.Timestamp(start_date) + pd.Timedelta(hours=start_hour)
if origin > last_hour + pd.Timedelta(hours=1):
    origin = last_hour + pd.Timedelta(hours=1)
    st.info("History ends on 28 Feb 2026 23:00, so the forecast starts on 1 Mar 2026 00:00.")
future_hours = pd.date_range(origin, periods=horizon, freq="h")
zone_ids = members["PULocationID"].astype(str).tolist()

# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------
forecast = forecast_bottom_up(hourly, zones, origin, horizon, zone_ids)
history = hourly[hourly.index < origin][zone_ids].sum(axis=1)
baseline = run_method("naive_168", history.values, horizon)
actual = hourly[zone_ids].sum(axis=1).reindex(future_hours)
has_actual = actual.notna().all()

# Calibrated 80% range for area totals (the team's interval calibration table).
lower = upper = None
if level == "Area total":
    calibration = read_csv(TS_MODEL / "tables" / "interval_calibration.csv")
    calibration = calibration[(calibration["series_id"] == area) & (calibration["schedule"] == horizon)
                              & (calibration["coverage"] == 0.8)].set_index("bucket")["radius"]
    radius = np.array([calibration[lead_bucket(lead, horizon)] for lead in range(1, horizon + 1)])
    lower, upper = np.clip(forecast - radius, 0, None), forecast + radius

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
peak_index = int(np.argmax(forecast))
cards = [
    ("Forecast pickups", f"{forecast.sum():,.0f}", f"next {horizon_label}"),
    ("Peak hour", f"{future_hours[peak_index]:%a %H:00}", f"{forecast[peak_index]:,.0f} pickups"),
]
if not has_actual:
    cards += [("Actual data", "not yet", "this is a true future forecast"),
              ("Method", "Model 1", "selected per zone")]
kpi_row(cards)
st.write("")

fig = go.Figure()
recent = history[history.index >= origin - pd.Timedelta(days=3)]
fig.add_scatter(x=recent.index, y=recent.values, name="History (known)", line=dict(color=GRAY, width=1.5))
if lower is not None:
    fig.add_scatter(x=future_hours, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=future_hours, y=lower, fill="tonexty", fillcolor=tint(AMBER, 0.15), line=dict(width=0),
                    name="80% range", hoverinfo="skip")
if has_actual:
    fig.add_scatter(x=future_hours, y=actual.values, name="Actual", line=dict(color=INK, width=2))
fig.add_scatter(x=future_hours, y=baseline, name="Last-week forecast", line=dict(color=INDIGO, width=1.5, dash="dot"))
fig.add_scatter(x=future_hours, y=forecast, name="Model 1 forecast", line=dict(color=AMBER, width=2.5))
fig.add_vline(x=origin, line_color=GRAY, line_dash="dash", line_width=1)
fig.add_annotation(x=origin, y=1, yref="paper", text="forecast issued", showarrow=False, xanchor="left", font=dict(size=11))
fig.update_layout(title=f"{title}: hourly pickups", hovermode="x unified", yaxis_title="Pickups per hour")
show_chart(fig, height=430)

if level == "Single zone":
    zone = members.iloc[0]
    st.caption(f"Method selected for this zone in validation: **{METHOD_NAMES.get(zone['ts_method'], zone['ts_method'])}** "
               f"(demand pattern: {zone['demand_pattern']}, volume tier: {zone['volume_tier']}).")
    with st.expander("Compare all four Model 1 methods for this zone"):
        zone_history = hourly[hourly.index < origin][zone_ids[0]].values
        comparison = go.Figure()
        if has_actual:
            comparison.add_scatter(x=future_hours, y=actual.values, name="Actual", line=dict(color=INK, width=2))
        rows = []
        for (method, name), color in zip(METHOD_NAMES.items(), [AMBER, TEAL, COPPER, INDIGO]):
            values = run_method(method, zone_history, horizon)
            comparison.add_scatter(x=future_hours, y=values, name=name, line=dict(color=color, width=1.8))
            if has_actual:
                rows.append({"Method": name, "MAE": np.abs(values - actual.values).mean(),
                             "Selected": "✓" if method == zone["ts_method"] else ""})
        comparison.update_layout(hovermode="x unified", yaxis_title="Pickups per hour", height=360)
        show_chart(comparison)
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("MAE"), hide_index=True, width="stretch",
                         column_config={"MAE": st.column_config.NumberColumn(format="%.2f")})
else:
    st.caption(f"Model 1 forecasts each of the {len(zone_ids)} zones with its own selected method and adds them up "
               "(bottom-up). The shaded 80% range uses the team's calibration from two pre-test weeks.")

# Daily plan table
plan = pd.DataFrame({"hour": future_hours, "forecast": forecast, "last_week": baseline})
if has_actual:
    plan["actual"] = actual.values
plan["day"] = plan["hour"].dt.strftime("%a %d %b")
daily = plan.groupby("day", sort=False).agg(forecast=("forecast", "sum"), last_week=("last_week", "sum"),
                                            peak_hour=("forecast", lambda s: plan.loc[s.idxmax(), "hour"].strftime("%H:00")))
if has_actual:
    daily["actual"] = plan.groupby("day", sort=False)["actual"].sum()
daily = daily.reset_index()

left, right = st.columns([1.2, 1])
with left:
    melted = daily.melt(id_vars="day", value_vars=[c for c in ["forecast", "actual"] if c in daily], var_name="Series", value_name="Pickups")
    melted["Series"] = melted["Series"].map({"forecast": "Model 1 forecast", "actual": "Actual"})
    fig = px.bar(melted, x="day", y="Pickups", color="Series", barmode="group",
                 color_discrete_map={"Model 1 forecast": AMBER, "Actual": INK}, title="Daily totals")
    fig.update_traces(marker_line_color="white", marker_line_width=2)
    fig.update_layout(xaxis_title="")
    show_chart(fig, height=320)
with right:
    st.markdown("**Shift-planning table**")
    st.dataframe(daily, hide_index=True, width="stretch",
                 column_config={"forecast": st.column_config.NumberColumn("Forecast", format="%.0f"),
                                "last_week": st.column_config.NumberColumn("Last week", format="%.0f"),
                                "actual": st.column_config.NumberColumn("Actual", format="%.0f"),
                                "peak_hour": "Peak hour", "day": "Day"})
    st.download_button("Download hourly forecast (CSV)", plan.drop(columns="day").to_csv(index=False).encode(),
                       file_name=f"forecast_{title.replace(' ', '_')}_{origin:%Y%m%d_%H}.csv", mime="text/csv")

callout("<b>How it helps users:</b> fleet managers can size driver shifts for each day and hour of the coming "
        "week, plan maintenance in the quietest hours, and know in advance where the peaks will be. "
        "Remember that a seasonal forecast cannot see storms or one-off events: adjust it when a weather "
        "alert is issued.", "use")
