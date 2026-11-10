import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (INDIGO, GREEN, AMBER, REPO, SEQ, YELLOW, callout, hero, kpi_row,
                   load_features, load_zones, read_csv, show_chart, steps, zone_map)

hero(
    "Chapter 1",
    "Building a trustworthy dataset",
    "Before any model, we needed one clean table: how many taxi pickups happened in each zone "
    "in each hour, together with the weather in that zone at that hour.",
)

kpi_row([
    ("Raw trips collected", "23,318,979", "3.18 GB CSV, private Kaggle dataset"),
    ("Clean trips kept", "22,579,793", "96.83% retained"),
    ("Zone-hour grid", "599,472", "138 zones × 4,344 hours"),
    ("Final modelling rows", "576,288", "after the 7-day lag warm-up"),
])

# ---------------------------------------------------------------------------
st.header("Where the data comes from")
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("#### :material/local_taxi: NYC TLC trip records")
        st.markdown(
            "Official record of every Yellow and Green taxi trip in New York City. Twelve monthly "
            "files (6 Yellow, 6 Green) were downloaded with Python HTTP requests from the TLC "
            "trip-data endpoint, together with the zone lookup table."
        )
with right:
    with st.container(border=True):
        st.markdown("#### :material/partly_cloudy_day: Open-Meteo weather API")
        st.markdown(
            "A REST API that returns hourly weather for any latitude/longitude. We queried it at "
            "the centre point of **each of the 138 zones**, so every zone gets its own local "
            "weather instead of one city-wide station."
        )

steps([
    ("Identify zones", "138 Manhattan + Queens zones and their centre points."),
    ("Get weather", "Hourly Open-Meteo weather for every zone."),
    ("Filter trips", "Keep trips picked up in those zones, Sep 2025 – Feb 2026."),
    ("Join", "Match each trip to the weather of its pickup zone and pickup hour."),
])

st.write("")
zones = load_zones()
col_map, col_month = st.columns([1.1, 1])
with col_map:
    st.subheader("The 138 study zones")
    fig = zone_map(zones, "Borough", color_map={"Manhattan": INDIGO, "Queens": AMBER},
                   hover_data={"total_trips": ":,", "service_zone": True, "PULocationID": False},
                   labels={"total_trips": "Total pickups", "service_zone": "Service area"})
    show_chart(fig, height=430, toolbar=False)
    st.caption("Official NYC TLC taxi-zone outlines. Hover a zone to see its total pickups.")

with col_month:
    st.subheader("Trips collected per month")
    summary = read_csv(REPO / "data" / "raw" / "taxi_collection_summary.csv")
    summary["taxi_type"] = summary["taxi_type"].str.title() + " taxi"
    fig = px.bar(
        summary, x="month", y="rows_kept", color="taxi_type",
        color_discrete_map={"Yellow taxi": YELLOW, "Green taxi": GREEN},
        labels={"rows_kept": "Trips kept", "month": ""},
    )
    fig.update_traces(marker_line_color="white", marker_line_width=2,
                      hovertemplate="%{x}<br>%{y:,} trips<extra>%{fullData.name}</extra>")
    show_chart(fig, height=430)
    st.caption("Green taxis are a small share (~1%) but are kept because they serve outer areas.")

# ---------------------------------------------------------------------------
st.header("Choosing the fields that matter")
left, right = st.columns(2)
with left:
    st.markdown("**Taxi trips – 5 of 20 columns kept**")
    st.dataframe(pd.DataFrame([
        ["Taxi type", "Yellow or Green"],
        ["Pickup time", "When demand happens"],
        ["Dropoff time", "Trip duration; helps spot bad records"],
        ["Pickup zone (PULocationID)", "Where demand happens; links to the weather"],
        ["Dropoff zone (DOLocationID)", "Where cabs end up after a trip"],
    ], columns=["Kept", "Why"]), hide_index=True, width="stretch")
    st.caption("Left out: fares, tips, tolls (revenue, not demand), passenger count (often missing), distance, payment type.")
with right:
    st.markdown("**Weather – 4 of 50+ variables kept**")
    st.dataframe(pd.DataFrame([
        ["Apparent temperature (°C)", "How cold or hot it feels, incl. wind and humidity"],
        ["Rain (mm)", "People avoid walking and waiting in the rain"],
        ["Snowfall (cm)", "Different effect from rain; also slows traffic"],
        ["Precipitation probability (%)", "What the forecast said; people plan around it"],
    ], columns=["Kept", "Why"]), hide_index=True, width="stretch")
    st.caption("Left out: plain temperature, humidity, wind (already in apparent temperature), cloud cover, pressure.")

# ---------------------------------------------------------------------------
st.header("Cleaning: from 23.3 M raw trips to 22.6 M verified trips")
left, right = st.columns([1, 1])
with left:
    funnel = pd.DataFrame({
        "stage": ["All NYC trips in the 12 TLC files", "Picked up in Manhattan or Queens",
                  "Inside the study period", "After the 4 cleaning rules"],
        "trips": [24_562_021, 23_319_012, 23_318_979, 22_579_793],
    })
    fig = go.Figure(go.Funnel(
        y=funnel["stage"], x=funnel["trips"], texttemplate="%{value:,}<br>%{percentInitial:.1%}",
        marker=dict(color=[SEQ[2], SEQ[3], SEQ[4], SEQ[5]]),
        hovertemplate="%{y}<br>%{x:,} trips<extra></extra>",
    ))
    fig.update_layout(title="Trip funnel", margin=dict(l=10, r=10, t=50, b=10))
    show_chart(fig, height=340)
with right:
    rules = pd.DataFrame({
        "rule": ["Exact duplicate trips", "Dropoff at/before pickup", "Trip longer than 6 hours",
                 "Unknown / outside-NYC dropoff"],
        "removed": [276_749, 329_654, 6_826, 125_957],
    })
    fig = px.bar(rules, x="removed", y="rule", orientation="h", text="removed",
                 labels={"removed": "Trips removed", "rule": ""}, title="Records removed by each rule")
    fig.update_traces(marker_color=AMBER, texttemplate="%{text:,}", textposition="outside",
                      hovertemplate="%{y}<br>%{x:,} trips removed<extra></extra>", cliponaxis=False)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 420_000])
    show_chart(fig, height=340)

callout(
    "<b>739,186 invalid records (3.17%) were removed.</b> The 9,166 trips in the repeated 01:00 hour "
    "of the 2 Nov 2025 daylight-saving change were <b>kept</b>, because they are real passengers. "
    "Hours with no pickups were filled with 0 so quiet periods are not lost: 27.7% of all zone-hours "
    "have zero pickups.",
)

# ---------------------------------------------------------------------------
st.header("Feature engineering: what the models learn from")
features_table = pd.DataFrame([
    ["Target", "trip_count", "Pickups in the zone during the hour"],
    ["Location", "PULocationID", "Pickup zone (138 values)"],
    ["Calendar", "hour, day_of_week, month", "When the hour happens"],
    ["Calendar", "is_weekend, is_holiday", "Saturday/Sunday flag; US federal holiday flag"],
    ["Weather", "apparent_temperature, rain, snowfall, precipitation_probability", "Zone-level hourly weather"],
    ["Recent demand", "trip_count_lag_1h / 24h / 168h", "Pickups 1 hour, 1 day and 1 week earlier"],
    ["Recent demand", "trip_count_roll_3h / roll_24h", "Average of the previous 3 and 24 hours (no leakage)"],
], columns=["Group", "Feature(s)", "Meaning"])
st.dataframe(features_table, hide_index=True, width="stretch")
st.caption(
    "The first 168 hours (1–7 Sep) were dropped because the 1-week lag is empty there: "
    "599,472 − 23,184 = 576,288 rows × 17 columns, saved as features.csv (plus Manhattan and Queens files)."
)

with st.expander("Peek at the final dataset (features.csv)"):
    features = load_features()
    zone_name = st.selectbox("Zone", zones["label"], index=int(zones.index[zones["Zone"] == "Midtown Center"][0]))
    zone_id = int(zones.loc[zones["label"] == zone_name, "PULocationID"].iloc[0])
    st.dataframe(features[features["PULocationID"] == zone_id].head(48), hide_index=True, width="stretch")

callout(
    "<b>Why this matters for users:</b> every number in this dashboard is traceable back to a "
    "documented, reproducible cleaning step, so operators can trust the demand figures they plan with.",
    "use",
)
