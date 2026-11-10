import streamlit as st

from utils import callout, hero, kpi_row, steps

hero(
    "Chapter 6",
    "From evidence to action",
    "What a taxi fleet operator should do differently because of this project, the value it brings, "
    "and the limits to keep in mind.",
)

kpi_row([
    ("Targeted deployment", "16 / 138", "zones carry 50% of all trips"),
    ("Peak planning", "10,770", "trips per hour on Thursday 6 pm"),
    ("Snow > 1 cm", "−75%", "pickups; rain > 5 mm: +14%"),
    ("Night shifts", "4.7×", "weekend nights vs weekday nights"),
])

st.header("Recommendations")
recommendations = [
    ("Concentrate the fleet where demand is", "EDA",
     "Manhattan drives 90.6% of pickups; 55 Yellow-Zone areas capture 86.3% of volume.",
     "Station 85–90% of active vehicles in core Manhattan / Yellow-Zone areas; stop cruising the 42 near-empty outer zones.",
     "Fewer unpaid empty miles and less fuel burnt.", "May miss hidden demand in outer areas."),
    ("Stage cabs before airport peaks", "EDA",
     "Two airport zones produce 6.5% of pickups. LaGuardia peaks at 14:00 (259/hr), JFK at 16:00 (381/hr).",
     "Stage airport vehicles 30–60 minutes before 14:00 (LGA) and 16:00 (JFK).",
     "Captures high-fare airport trips with shorter passenger waits.", "Flight delays; empty return trips."),
    ("React to the weather", "EDA + ML",
     "Morning rain adds 14.6–22.1% demand; heavy snow (> 0.5 cm) cuts demand by 40–70%.",
     "Call in extra drivers during morning rain; cut the active fleet by about half on heavy-snow warnings.",
     "Captures rain surges and avoids idle cost and accident risk in snow.", "Changing weather forecasts cause short mismatches."),
    ("Plan shifts 1–7 days ahead", "Time series",
     "Strong 24-hour and 168-hour cycles; Model 1 cuts forecast error by 16.5% vs repeating last week.",
     "Build driver rosters and maintenance windows from the day-ahead and week-ahead forecast.",
     "Better vehicle availability; maintenance scheduled in quiet hours.", "Cannot foresee sudden disruptions (storms, strikes)."),
    ("Pre-position for the morning commute", "Association rules",
     "Weekday AM rush in commuter zones → High demand with 70–73% confidence (2.6–2.7× lift).",
     "Pre-position cabs in residential / commuter hubs before 07:00 on weekdays.",
     "Cabs are there when a surge is 2.6–2.7× more likely than usual.", "Gives a demand level, not an exact count."),
    ("Follow the nightlife", "Association rules",
     "Evenings (20–23) in nightlife zones → High demand with 58–63% confidence (2.2–2.3× lift); weekends add more. East Village leads Friday/Saturday nights.",
     "Move evening cabs into nightlife zones from 20:00, and keep them there until about 04:00 on Friday and Saturday.",
     "Captures late-night trips when public transit runs less often.", "Strong ride-hailing competition at night."),
]
for i in range(0, len(recommendations), 2):
    cols = st.columns(2)
    for col, (title, source, evidence, action, benefit, limit) in zip(cols, recommendations[i:i + 2]):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {title}")
                st.markdown(f'<span class="pill">Evidence: {source}</span>', unsafe_allow_html=True)
                st.markdown(f"**What we found** – {evidence}")
                st.markdown(f"**What to do** – {action}")
                st.markdown(f":green[**Benefit**] – {benefit}  \n:orange[**Watch out**] – {limit}")

st.header("A dispatcher's day with this dashboard")
steps([
    ("Weekly planning", "<b>Forecast planner</b>: week-ahead pickups per zone to size driver rosters."),
    ("Each morning", "<b>Dispatch advisor</b>: check today's conditions (rain? weekend? commuter zone?) for expected demand levels."),
    ("Every hour", "<b>Demand predictor</b>: next-hour pickups per zone, with what-if weather, to reposition cabs."),
    ("Afterwards", "Compare forecasts with actual pickups and feed new data back into the models."),
])

st.header("Conclusion")
left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.markdown("#### :material/check_circle: Main outcome")
        st.markdown(
            "An end-to-end, reproducible demand analytics workflow for 138 zones in Manhattan and Queens:\n"
            "- clean zone-hour dataset with local weather (Unit 1)\n"
            "- next-hour prediction with **R² = 0.963** (Unit 2)\n"
            "- day/week-ahead forecasts with **16.5% less error** than repeating last week (Unit 3)\n"
            "- **152 validated rules** linking conditions to demand levels (Unit 3)\n"
            "- this interactive dashboard for business users"
        )
with right:
    with st.container(border=True):
        st.markdown("#### :material/warning: Limitations")
        st.markdown(
            "- Only six months of data (Sep 2025 – Feb 2026): autumn and winter, no annual seasonality.\n"
            "- Unusual events (the Feb blizzard) cause large forecast errors.\n"
            "- MAPE is hard to interpret when actual demand is zero or very low.\n"
            "- Pickups measure *served* demand, not people who could not find a cab.\n"
            "- The Review 2 holdout was examined earlier; a fresh period is needed for blind confirmation."
        )
with st.container(border=True):
    st.markdown("#### :material/rocket_launch: Future scope")
    st.markdown(
        "- Add multiple years of data and special-event calendars (concerts, games, flight schedules).\n"
        "- Improve forecasts during sudden shocks, e.g. by switching to weather-alert scenarios.\n"
        "- Deploy a live version with regularly refreshed forecasts and automatic fleet-allocation suggestions."
    )

callout("<b>Bottom line:</b> recent demand, location and time explain most of taxi demand; weather matters "
        "mainly in extreme conditions such as heavy snow. Combining short-term prediction, multi-day "
        "forecasts and if-then rules lets operators move from reacting to planning.", "use")
