import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (DEMAND_COLORS, INK_2, LINE, TIME_BANDS, callout, describe_rule, hero, kpi_row, load_rules,
                   load_zones, show_chart)

hero(
    "Try it live · Association rules",
    "Dispatch advisor",
    "Describe the situation: zone, day, hour and weather. The advisor finds every validated demand rule "
    "that matches and tells you whether to expect <b>High, Medium or Low</b> demand for that zone, and what to do.",
)

zones = load_zones()
rules = load_rules()

CLUSTER_ITEM = {"daytime": "daytime/business", "morning_commute": "commuter", "nightlife": "nightlife"}
ACTIONS = {
    "High": "Send extra cabs to this zone ahead of time and hold them nearby.",
    "Medium": "Keep normal coverage. No repositioning needed.",
    "Low": "Expect a quiet period. Move idle cabs to busier zones or schedule breaks and maintenance.",
}


def time_band_of(hour):
    if hour <= 4:
        return "late_night"
    if hour <= 6:
        return "early_morning"
    if hour <= 9:
        return "am_rush"
    if hour <= 15:
        return "midday"
    if hour <= 19:
        return "pm_rush"
    return "evening"


def temperature_band(celsius):
    if celsius <= -0.9:
        return "lower"
    if celsius <= 9.9:
        return "middle"
    return "upper"


def conditions_for(hour, day_type, holiday, temperature, rain, snow, cluster):
    return {
        f"time_band={time_band_of(hour)}", f"day_type={day_type}",
        f"holiday={'federal' if holiday else 'regular'}", f"temperature={temperature_band(temperature)}",
        f"rain={'yes' if rain else 'no'}", f"snow={'yes' if snow else 'no'}", f"zone_cluster={cluster}",
    }


def matching_rules(conditions):
    """A rule applies when all of its IF-conditions are part of the situation.
    Rules are ranked by test lift: how much more likely the demand level is than usual."""
    mask = rules["conditions"].apply(lambda items: set(items) <= conditions)
    return rules[mask].sort_values(["test_lift", "test_confidence"], ascending=False)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
active = zones[zones["arm_active"] == True]  # noqa: E712  (rules were mined on the 96 active zones)
col_zone, col_day, col_hour = st.columns([1.6, 1, 1.4])
default_zone = list(active["Zone"]).index("Astoria")
zone_label = col_zone.selectbox("Zone", active["label"], index=default_zone,
                                help="Only the 96 zones averaging at least one pickup per hour were mined.")
day_type = col_day.radio("Day", ["weekday", "weekend"], horizontal=True, format_func=str.title)
hour = col_hour.slider("Hour of day", 0, 23, 8, format="%d:00")

col_t, col_r, col_s, col_h = st.columns(4)
temperature = col_t.slider("Feels-like temperature (°C)", -20, 30, 2)
rain = col_r.toggle("Raining")
snow = col_s.toggle("Snowing")
holiday = col_h.toggle("Federal holiday")

zone = active[active["label"] == zone_label].iloc[0]
cluster = zone["arm_cluster"]
situation = conditions_for(hour, day_type, holiday, temperature, rain, snow, cluster)
matches = matching_rules(situation)

st.markdown(
    f'<span class="pill">{TIME_BANDS[time_band_of(hour)]}</span>'
    f'<span class="pill">{day_type.title()}</span>'
    f'<span class="pill">{CLUSTER_ITEM[cluster]} zone</span>'
    f'<span class="pill">{temperature_band(temperature)} temperature</span>'
    + ('<span class="pill">rain</span>' if rain else "")
    + ('<span class="pill">snow</span>' if snow else "")
    + ('<span class="pill">holiday</span>' if holiday else ""),
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
pickup_ranges = {
    "Low": f"≤ {zone['q_low']:.0f} pickups/hour",
    "Medium": f"{zone['q_low']:.0f}–{zone['q_high']:.0f} pickups/hour",
    "High": f"> {zone['q_high']:.0f} pickups/hour",
}
if matches.empty:
    kpi_row([
        ("Expected demand", "Typical", "no strong rule applies"),
        ("Matching rules", "0", "of 152 validated rules"),
        ("Zone type", CLUSTER_ITEM[cluster].title(), zone["Zone"]),
    ])
    callout("No validated rule covers this exact situation, so demand should be close to normal for this "
            "zone and time. Use the <b>Demand predictor</b> or <b>Forecast planner</b> for a numeric estimate.", "warn")
else:
    best = matches.iloc[0]
    expected = best["demand"]
    kpi_row([
        ("Expected demand", expected, f"for {zone['Zone']}: {pickup_ranges[expected]}"),
        ("Confidence", f"{100 * best['test_confidence']:.0f}%", "of matching hours in the test period"),
        ("Lift", f"{best['test_lift']:.2f}×", "more likely than usual"),
        ("Matching rules", f"{len(matches)}", "of 152 validated rules"),
    ])
    st.write("")
    callout(f"<b>Recommended action:</b> {ACTIONS[expected]}<br><span style='color:{INK_2}'>Strongest rule: "
            f"IF {describe_rule(best['conditions'])} THEN demand = {expected}.</span>", "use")

    # Best matching rule for each demand level, so conflicting evidence stays visible.
    per_level = matches.drop_duplicates("demand")
    if len(per_level) > 1:
        other = per_level.iloc[1:]
        text = "; ".join(f"<b>{r['demand']}</b> (lift {r['test_lift']:.2f}×, IF {describe_rule(r['conditions'])})"
                         for _, r in other.iterrows())
        callout(f"<b>Mixed signals:</b> weaker rules also point to {text}. The verdict follows the rule "
                "with the highest lift.", "warn")

    table = matches.assign(rule=matches["conditions"].apply(describe_rule))
    st.dataframe(
        table[["rule", "demand", "test_confidence", "test_lift", "test_antecedent_count"]],
        hide_index=True, width="stretch",
        column_config={
            "rule": st.column_config.TextColumn("IF …", width="large"),
            "demand": "THEN demand",
            "test_confidence": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
            "test_lift": st.column_config.NumberColumn("Lift", format="%.2f×"),
            "test_antecedent_count": st.column_config.NumberColumn("Test hours matched", format="%d"),
        },
    )

# ---------------------------------------------------------------------------
# 24-hour outlook for the same zone and conditions
# ---------------------------------------------------------------------------
st.subheader(f"24-hour outlook for {zone['Zone']} ({day_type})")
outlook = []
for h in range(24):
    found = matching_rules(conditions_for(h, day_type, holiday, temperature, rain, snow, cluster))
    if found.empty:
        outlook.append({"hour": h, "demand": "No rule", "confidence": None})
    else:
        outlook.append({"hour": h, "demand": found.iloc[0]["demand"], "confidence": found.iloc[0]["test_confidence"]})
outlook = pd.DataFrame(outlook)

colors = {**DEMAND_COLORS, "No rule": LINE}
fig = go.Figure()
for demand in ["High", "Medium", "Low", "No rule"]:
    part = outlook[outlook["demand"] == demand]
    fig.add_bar(x=part["hour"], y=[1] * len(part), name=demand, marker_color=colors[demand],
                marker_line_color="white", marker_line_width=2,
                customdata=part["confidence"].fillna(0) * 100,
                hovertemplate="%{x}:00 → " + demand + "<br>confidence %{customdata:.0f}%<extra></extra>")
fig.update_layout(barmode="stack", bargap=0.05, height=200, yaxis=dict(visible=False), legend_traceorder="normal",
                  xaxis=dict(title="Hour of day", dtick=1, range=[-0.5, 23.5]), margin=dict(l=10, r=10, t=40, b=10))
show_chart(fig)
st.caption("Each block shows the strongest matching rule for that hour. Change the day, weather or zone above "
           "to see how the daily plan shifts.")

callout("<b>How it helps users:</b> a dispatcher needs no model output. Simple, validated IF-THEN rules "
        "(each checked on unseen data) say when a zone is likely to be unusually busy or quiet compared "
        "with its normal level.", "info")
