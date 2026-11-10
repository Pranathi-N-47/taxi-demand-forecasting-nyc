import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (INDIGO, DEMAND_COLORS, DEMAND_PALE, GRAY, INK, LEVEL_MEANING, AMBER, callout, compare_with_normal, hero,
                   kpi_row, load_features, load_zone_normal, load_zone_shapes, load_zones, predict_demand,
                   show_chart)

hero(
    "Try it live · Machine learning",
    "Demand predictor",
    "Our best model (HistGradientBoosting, R² 0.963) predicts how many pickups a zone will have in a "
    "given hour. Pick a zone and time: inputs are filled in from the real data, and you can change the "
    "weather or recent demand to run a <b>what-if scenario</b>.",
)

zones = load_zones()
features = load_features()

# ---------------------------------------------------------------------------
# 1. Choose zone and time
# ---------------------------------------------------------------------------
st.subheader("1 · Choose a zone and an hour")
col_zone, col_date, col_hour = st.columns([1.6, 1, 1])
default_zone = int(zones.index[zones["Zone"] == "Midtown Center"][0])
zone_label = col_zone.selectbox("Pickup zone", zones["label"], index=default_zone)
date = col_date.date_input("Date", dt.date(2026, 2, 12), min_value=dt.date(2025, 9, 8), max_value=dt.date(2026, 2, 28))
hour = col_hour.selectbox("Hour", list(range(24)), index=18, format_func=lambda h: f"{h:02d}:00 – {h:02d}:59")

zone = zones[zones["label"] == zone_label].iloc[0]
zone_id = int(zone["PULocationID"])
timestamp = pd.Timestamp(date) + pd.Timedelta(hours=hour)
row = features[(features["PULocationID"] == zone_id) & (features["pickup_hour"] == timestamp)]
if row.empty:
    st.warning("No data for this zone and hour.")
    st.stop()
actual_row = row.iloc[0]

# ---------------------------------------------------------------------------
# 2. Inputs (pre-filled, editable)
# ---------------------------------------------------------------------------
st.subheader("2 · Check or change the inputs")
st.caption(f"Pre-filled with what really happened on {timestamp:%A %d %b %Y, %H:00}. "
           "Change any value to see how the prediction responds.")
key = f"{zone_id}_{timestamp}"  # new widgets (and fresh defaults) whenever zone or time changes

col_w, col_d = st.columns(2)
with col_w:
    with st.container(border=True):
        st.markdown("**:material/partly_cloudy_day: Weather in the zone**")
        temperature = st.slider("Feels-like temperature (°C)", -25.0, 35.0, float(actual_row["apparent_temperature"]), 0.5, key="t" + key)
        rain = st.slider("Rain (mm in the hour)", 0.0, 15.0, float(min(actual_row["rain"], 15)), 0.1, key="r" + key)
        snow = st.slider("Snowfall (cm in the hour)", 0.0, 5.0, float(min(actual_row["snowfall"], 5)), 0.05, key="s" + key)
        precip = st.slider("Precipitation probability (%)", 0, 100, int(actual_row["precipitation_probability"]), key="p" + key)
        holiday = st.toggle("Federal holiday", bool(actual_row["is_holiday"]), key="h" + key)
with col_d:
    with st.container(border=True):
        st.markdown("**:material/history: Recent demand in the zone**")
        lag_1 = st.number_input("Pickups 1 hour earlier", 0, 2000, int(actual_row["trip_count_lag_1h"]), key="l1" + key)
        lag_24 = st.number_input("Pickups at this hour yesterday", 0, 2000, int(actual_row["trip_count_lag_24h"]), key="l24" + key)
        lag_168 = st.number_input("Pickups at this hour last week", 0, 2000, int(actual_row["trip_count_lag_168h"]), key="l168" + key)
        roll_3 = st.number_input("Average of the last 3 hours", 0.0, 2000.0, float(round(actual_row["trip_count_roll_3h"], 1)), key="r3" + key)
        roll_24 = st.number_input("Average of the last 24 hours", 0.0, 2000.0, float(round(actual_row["trip_count_roll_24h"], 1)), key="r24" + key)

scenario = row.copy()
scenario["apparent_temperature"] = temperature
scenario["rain"] = rain
scenario["snowfall"] = snow
scenario["precipitation_probability"] = precip
scenario["is_holiday"] = holiday
scenario["trip_count_lag_1h"] = lag_1
scenario["trip_count_lag_24h"] = lag_24
scenario["trip_count_lag_168h"] = lag_168
scenario["trip_count_roll_3h"] = roll_3
scenario["trip_count_roll_24h"] = roll_24

predicted_real = float(predict_demand(row)[0])
predicted = float(predict_demand(scenario)[0])
changed = abs(predicted - predicted_real) > 1e-9

# ---------------------------------------------------------------------------
# 3. Output
# ---------------------------------------------------------------------------
st.subheader("3 · Prediction")
normal_table = load_zone_normal()
normal = float(normal_table[(normal_table["PULocationID"] == zone_id) & (normal_table["day_of_week"] == timestamp.dayofweek)
                            & (normal_table["hour"] == hour)]["normal"].iloc[0])
level = compare_with_normal(predicted, normal)
change = 100 * ((predicted + 1) / (normal + 1) - 1)
kpi_row([
    ("Predicted pickups" + (" (your scenario)" if changed else ""), f"{predicted:,.1f}", f"{zone['Zone']}, {timestamp:%a %d %b %H:00}"),
    ("Actual pickups", f"{int(actual_row['trip_count']):,}", "what really happened"),
    (f"Normal for a {timestamp:%A} {hour:02d}:00", f"{normal:,.1f}", "this zone's average at this hour"),
    ("Demand level", level, f"{LEVEL_MEANING[level]} ({change:+.0f}% vs normal)"),
])
st.write("")
if changed:
    delta = predicted - predicted_real
    st.info(f"Your changes move the prediction from **{predicted_real:,.1f}** to **{predicted:,.1f}** pickups "
            f"({delta:+,.1f}, {100 * delta / max(predicted_real, 1e-9):+.1f}%).", icon=":material/tune:")

actions = {
    "High": "More passengers than a normal hour like this: send extra cabs here before the hour starts.",
    "Medium": "Demand looks normal for this hour: keep the usual number of cabs here.",
    "Low": "Fewer passengers than a normal hour like this: move some idle cabs to busier zones.",
}
callout(f"<b>Recommended action ({level} demand, {LEVEL_MEANING[level]}):</b> {actions[level]}", "use")

# ---------------------------------------------------------------------------
# The whole day for this zone
# ---------------------------------------------------------------------------
day = features[(features["PULocationID"] == zone_id) & (features["pickup_hour"].dt.date == date)].copy()
day["predicted"] = predict_demand(day)
fig = go.Figure()
fig.add_scatter(x=day["pickup_hour"].dt.hour, y=day["trip_count"], name="Actual", line=dict(color=INK, width=2))
fig.add_scatter(x=day["pickup_hour"].dt.hour, y=day["predicted"], name="Model prediction", line=dict(color=AMBER, width=2))
fig.add_scatter(x=day["pickup_hour"].dt.hour, y=day["trip_count_lag_168h"], name="Last-week baseline",
                line=dict(color=GRAY, width=1.5, dash="dot"))
fig.add_scatter(x=[hour], y=[predicted], mode="markers", name="Selected hour" + (" (scenario)" if changed else ""),
                marker=dict(size=13, color=INDIGO, line=dict(color="white", width=2)))
fig.update_layout(title=f"{zone['Zone']} on {date:%a %d %b %Y}: hour by hour", hovermode="x unified",
                  xaxis=dict(title="Hour of day", dtick=2), yaxis_title="Pickups")
show_chart(fig, height=380)
day_mae = (day["predicted"] - day["trip_count"]).abs().mean()
st.caption(f"Model MAE for this zone and day: **{day_mae:.2f}** pickups per hour "
           f"(last-week baseline: {(day['trip_count_lag_168h'] - day['trip_count']).abs().mean():.2f}).")

# ---------------------------------------------------------------------------
# Citywide view for the same hour
# ---------------------------------------------------------------------------
st.subheader(f"Where to send cabs at {timestamp:%H:00} on {timestamp:%a %d %b}")
st.caption("All 138 zones for the date and hour chosen above. Each zone's prediction is compared with its normal "
           "demand for this weekday and hour.")

city = features[features["pickup_hour"] == timestamp].copy()
weather_changed = (temperature != float(actual_row["apparent_temperature"]) or rain != float(min(actual_row["rain"], 15))
                   or snow != float(min(actual_row["snowfall"], 5)) or precip != int(actual_row["precipitation_probability"])
                   or holiday != bool(actual_row["is_holiday"]))

col_show, col_toggle = st.columns([1.3, 1])
view = col_show.segmented_control(
    "Show", ["Need extra cabs", "Busiest zones", "Can spare cabs"], default="Need extra cabs", required=True,
    help="Need extra cabs = most above normal. Busiest = most predicted pickups. Can spare cabs = most below normal.")
col_toggle.write("")
use_scenario = col_toggle.toggle("Apply my weather changes to every zone", value=weather_changed, disabled=not weather_changed,
                                 help="Change the weather sliders above to run a citywide what-if, e.g. 2 cm of snow everywhere.")
if use_scenario:
    city["apparent_temperature"] = temperature
    city["rain"] = rain
    city["snowfall"] = snow
    city["precipitation_probability"] = precip
    city["is_holiday"] = holiday

city["predicted"] = predict_demand(city)
city = city.merge(zones, on="PULocationID")
city = city.merge(normal_table, on=["PULocationID", "day_of_week", "hour"], how="left")
city["level"] = [compare_with_normal(p, n) for p, n in zip(city["predicted"], city["normal"])]
city["difference"] = city["predicted"] - city["normal"]

# The 10 zones for the chosen view: shown in the table and numbered on the map.
if view == "Need extra cabs":
    top = city[city["difference"] > 0].sort_values("difference", ascending=False).head(10)
elif view == "Busiest zones":
    top = city.sort_values("predicted", ascending=False).head(10)
else:
    top = city[city["difference"] < 0].sort_values("difference").head(10)
top = top.assign(rank=range(1, len(top) + 1))
highlight = set(top["PULocationID"])

st.markdown(
    "**How to read it:** "
    f"<span style='color:{DEMAND_COLORS['High']}'>■</span> **High** = busier than usual · "
    f"<span style='color:{DEMAND_COLORS['Medium']}'>■</span> **Medium** = about usual (within ±15%) · "
    f"<span style='color:{DEMAND_COLORS['Low']}'>■</span> **Low** = quieter than usual. "
    f"Numbers 1–{len(top)} mark the zones in the table (strong colours); other zones are pale. "
    "The map zooms to the numbered zones.",
    unsafe_allow_html=True,
)

left, right = st.columns([0.85, 1.5])
with left:
    fig = go.Figure()
    for name in ["High", "Medium", "Low"]:
        for in_top in [False, True]:
            part = city[(city["level"] == name) & (city["PULocationID"].isin(highlight) == in_top)]
            if part.empty:
                continue
            color = DEMAND_COLORS[name] if in_top else DEMAND_PALE[name]
            fig.add_choropleth(
                geojson=load_zone_shapes(), featureidkey="properties.id", locations=part["PULocationID"],
                z=[1] * len(part), colorscale=[[0, color], [1, color]], showscale=False,
                marker_line_color="white", marker_line_width=0.6,
                customdata=part[["Zone", "predicted", "normal", "difference", "trip_count"]].values,
                hovertemplate="<b>%{customdata[0]}</b><br>Predicted: %{customdata[1]:.0f} pickups<br>"
                              "Normal for this hour: %{customdata[2]:.0f}<br>Difference: %{customdata[3]:+.0f}<br>"
                              "Actual: %{customdata[4]}<extra>" + name + "</extra>",
            )
    fig.add_scattergeo(
        lat=top["latitude"], lon=top["longitude"], text=top["rank"].astype(str), mode="markers+text",
        marker=dict(size=18, color="white", line=dict(color=INK, width=1.5)),
        textfont=dict(size=10, color=INK), hoverinfo="skip",
    )
    # Zoom to the 10 numbered zones (with some margin) so the numbers do not overlap.
    if len(top):
        lat_pad = max(0.015, 0.2 * (top["latitude"].max() - top["latitude"].min()))
        lon_pad = max(0.015, 0.2 * (top["longitude"].max() - top["longitude"].min()))
        fig.update_geos(lataxis_range=[top["latitude"].min() - lat_pad, top["latitude"].max() + lat_pad],
                        lonaxis_range=[top["longitude"].min() - lon_pad, top["longitude"].max() + lon_pad])
    else:
        fig.update_geos(fitbounds="locations")
    fig.update_geos(visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), dragmode=False, showlegend=False)
    show_chart(fig, height=440, toolbar=False)
with right:
    show_difference = st.toggle("Show difference column", value=False)
    columns = ["rank", "Zone", "level", "predicted", "normal"] + (["difference"] if show_difference else []) + ["trip_count"]
    st.dataframe(
        top[columns],
        hide_index=True, width="stretch", height=410,
        column_config={"rank": st.column_config.NumberColumn("#", width="small"),
                       "level": st.column_config.TextColumn("Level", width="small"),
                       "predicted": st.column_config.NumberColumn("Predicted", format="%.0f"),
                       "normal": st.column_config.NumberColumn("Normal", format="%.0f"),
                       "difference": st.column_config.NumberColumn("Difference", format="%+.0f"),
                       "trip_count": st.column_config.NumberColumn("Actual", format="%d")})
    if top.empty:
        st.info("No zone is in this group at this hour.")

counts = city["level"].value_counts()
total_pred, total_actual = city["predicted"].sum(), city["trip_count"].sum()
st.caption(f"Right now: **{counts.get('High', 0)}** zones High, **{counts.get('Medium', 0)}** Medium, "
           f"**{counts.get('Low', 0)}** Low. All 138 zones: predicted **{total_pred:,.0f}** pickups vs actual "
           f"**{total_actual:,.0f}** ({100 * (total_pred / max(total_actual, 1) - 1):+.1f}%). 'Normal' = the zone's average "
           "for this weekday and hour in the training period. Hover a zone for details."
           + (" **Your weather scenario is applied to every zone.**" if use_scenario else ""))

callout("<b>How it helps users:</b> every hour a dispatcher sees which zones are about to be busy compared with "
        "their normal level, and moves idle cabs there before passengers start waiting. The model uses the "
        "actual pickups of the previous hour, so for planning further ahead use the <b>Forecast planner</b>.", "warn")
