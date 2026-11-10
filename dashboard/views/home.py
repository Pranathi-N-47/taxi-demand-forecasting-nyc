import pandas as pd
import streamlit as st

from utils import callout, hero, kpi_row, steps

hero(
    "23CSE452 Business Analytics · Team 12 · Final project",
    "Where and when will New Yorkers need a taxi?",
    "We combined 23.3 million Yellow and Green taxi trips with hourly weather for 138 zones in "
    "Manhattan and Queens (Sep 2025 – Feb 2026). Then we built models that predict, forecast and "
    "explain hourly pickup demand so fleet operators can put cabs in the right place at the right time.",
)

kpi_row([
    ("Trips analysed", "23.3 M", "Yellow + Green, 6 months"),
    ("Taxi zones", "138", "69 Manhattan · 69 Queens"),
    ("Zone-hours modelled", "576,288", "one row per zone per hour"),
    ("Best model accuracy", "R² 0.963", "next-hour prediction"),
    ("Forecast gain", "−16.5%", "error vs. last-week forecast"),
])

st.write("")
left, right = st.columns([1.1, 1])
with left:
    st.subheader("The business problem")
    st.markdown(
        "**Who:** NYC taxi fleet operators and dispatchers.  \n"
        "**Problem:** demand swings a lot across zones, hours and weather. Busy areas run short of "
        "cabs while taxis sit idle in quiet areas.  \n"
        "**Decisions we support:**\n"
        "- **Fleet positioning** – when and where to stage empty taxis before a peak.\n"
        "- **Shift scheduling** – how many drivers to put on the road each hour.\n"
        "- **Weather planning** – how to react to rain, snow and cold."
    )
with right:
    st.subheader("What the project delivers")
    st.markdown(
        "- **Prediction:** hourly pickups per zone for the next hour (machine learning).\n"
        "- **Forecast:** day-ahead and week-ahead pickups per zone (time series).\n"
        "- **Insight:** the main drivers of demand – recent demand, location, time and weather.\n"
        "- **Rules:** *if* time + place + weather, *then* demand is likely High / Medium / Low "
        "(association rule mining)."
    )

st.divider()
st.subheader("How the story unfolds")
st.caption("Use the sidebar to move through the chapters in order, or jump straight to the live tools.")
steps([
    ("Data", "Collect 23.3 M trips and hourly weather, clean them and build a zone-hour table."),
    ("Explore", "Find where, when and under which weather taxi demand happens."),
    ("Predict", "Train three ML models to predict next-hour pickups per zone."),
])
st.write("")
steps([
    ("Forecast", "Forecast 24 hours and 7 days ahead with seasonal time-series methods."),
    ("Rules", "Mine if-then rules that link conditions to High / Low demand."),
    ("Act", "Turn the evidence into fleet and staffing recommendations."),
])

st.write("")
callout(
    "<b>Try it live.</b> The <b>Demand predictor</b> runs our best ML model on any zone, hour and "
    "weather you choose. The <b>Forecast planner</b> produces a day-ahead or week-ahead forecast. "
    "The <b>Dispatch advisor</b> tells a dispatcher what demand to expect under given conditions.",
    "use",
)

st.divider()
st.subheader("Project team")
team = pd.DataFrame([
    ["CB.SC.U4CSE23102", "Adithya Ajay", "Data collection & dataset preparation, time-series data preparation, time-series Model 1"],
    ["CB.SC.U4CSE23105", "Amrith B", "Exploratory data analysis – I, interactive dashboard"],
    ["CB.SC.U4CSE23406", "Arun N", "Exploratory data analysis – II, data preparation for association rule mining"],
    ["CB.SC.U4CSE23643", "Pranathi Nibhanipudi", "Data preprocessing & feature engineering, association rule mining"],
    ["CB.SC.U4CSE23441", "Sadhana T P", "Predictive modelling & evaluation, time-series analysis (DHR, MSTL + ETS)"],
], columns=["Roll no.", "Name", "Responsibility"])
st.dataframe(team, hide_index=True, width="stretch")
