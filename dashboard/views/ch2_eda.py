import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (ROSE, TEAL, INDIGO, CLUSTER_COLORS, DAY_NAMES, EDA_FIG, EDA_TAB, GRAY, INK_2,
                   AMBER, SEQ, callout, figure, hero, kpi_row, load_aggregate_hourly,
                   load_zone_hourly, load_zones, read_csv, show_chart, tint, zone_map)

hero(
    "Chapter 2",
    "Exploring demand: where, when and in what weather",
    "Exploratory analysis of 21.7 million clean trips (8 Sep 2025 – 28 Feb 2026). "
    "Each tab answers one question a dispatcher would ask.",
)

zones = load_zones()
agg = load_aggregate_hourly()
agg = agg[agg["ds"] >= "2025-09-08"]  # same period as the EDA (after the lag warm-up week)

tab_where, tab_when, tab_weather, tab_events, tab_zone = st.tabs(
    ["Where is demand?", "When does it happen?", "Weather", "Holidays & events", "Zone explorer"]
)

# ---------------------------------------------------------------------------
with tab_where:
    kpi_row([
        ("Half of all trips", "16 zones", "out of 138"),
        ("80% of all trips", "33 zones", "very concentrated"),
        ("Manhattan share", "90.6%", "19.7 M trips"),
        ("Inequality (Gini)", "0.71", "JFK is the only Queens zone in the top 10"),
    ])
    st.write("")
    left, right = st.columns(2)
    with left:
        ranked = zones.sort_values("total_trips", ascending=False).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        ranked["cumulative_share"] = 100 * ranked["total_trips"].cumsum() / ranked["total_trips"].sum()
        fig = px.area(ranked, x="rank", y="cumulative_share", hover_name="Zone",
                      labels={"rank": "Zones ranked by pickups", "cumulative_share": "Cumulative share of trips (%)"},
                      title="A few zones carry most of the demand")
        fig.update_traces(line_color=INDIGO, line_width=2, fillcolor=tint(INDIGO, 0.12),
                          hovertemplate="Top %{x} zones<br>%{y:.1f}% of trips<extra>%{hovertext}</extra>")
        for rank, share in [(16, 50), (33, 80)]:
            fig.add_scatter(x=[rank], y=[ranked.loc[rank - 1, "cumulative_share"]], mode="markers+text",
                            marker=dict(size=10, color=AMBER, line=dict(color="white", width=2)),
                            text=[f"  {rank} zones → {share}%"], textposition="middle right",
                            textfont=dict(color=INK_2), showlegend=False, hoverinfo="skip")
        show_chart(fig, height=380)
    with right:
        top = ranked.head(15)
        fig = px.bar(top, x="total_trips", y="Zone", orientation="h", color="Borough",
                     color_discrete_map={"Manhattan": INDIGO, "Queens": AMBER},
                     labels={"total_trips": "Total pickups", "Zone": ""}, title="Top 15 pickup zones")
        fig.update_traces(hovertemplate="%{y}<br>%{x:,} pickups<extra></extra>")
        fig.update_yaxes(categoryorder="total ascending")
        show_chart(fig, height=380)

    st.subheader("Three kinds of zones")
    st.caption("Zones were grouped (k-means) by the shape of their weekly demand profile.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**:blue[Core business + airports]**  \n58 zones · 84% of trips · busy in the daytime")
    c2.markdown("**:orange[Nightlife]**  \n10 zones · 11% of trips · huge Friday and Saturday nights")
    c3.markdown("**:green[Residential / commute]**  \n30 zones · 5% of trips · sharp 7–9 am peak")
    fig = zone_map(zones, "cluster_name", color_map=CLUSTER_COLORS, order=list(CLUSTER_COLORS),
                   hover_data={"total_trips": ":,", "avg_per_hour": ":.1f", "PULocationID": False},
                   labels={"cluster_name": "Zone type", "total_trips": "Total pickups", "avg_per_hour": "Pickups per hour"})
    show_chart(fig, height=520, toolbar=False)
    callout("<b>Use:</b> keep most cabs around the 33 zones that make up 80% of demand, and plan the "
            "nightlife and commuter zones on their own timetables.", "use")

# ---------------------------------------------------------------------------
with tab_when:
    kpi_row([
        ("Busiest hour", "Thu 6 pm", "10,770 trips per hour"),
        ("Quietest hour", "Tue 3 am", "~280 trips per hour"),
        ("Weekend nights", "4.7×", "busier than weekday nights (00–03)"),
        ("Winter drop", "−15%", "last 4 weeks vs first 4 weeks"),
    ])
    st.write("")
    series = st.segmented_control("Area", ["All zones", "Manhattan", "Queens", "Airports"],
                                  default="All zones", key="when_series", required=True)
    data = agg[(agg["series_id"] == series) & (agg["holiday_name"] == "none")]
    grid = data.pivot_table(index="day_of_week", columns="hour", values="trip_count_adj", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[f"{h:02d}:00" for h in grid.columns], y=DAY_NAMES,
        colorscale=[[i / 6, c] for i, c in enumerate(SEQ)], xgap=2, ygap=2,
        hovertemplate="%{y} %{x}<br>%{z:,.0f} pickups per hour<extra></extra>",
        colorbar=dict(title="Pickups / hour", thickness=12),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"Average pickups per hour – {series} (regular, non-holiday days)")
    show_chart(fig, height=360)

    daily = agg[agg["series_id"] == series].copy()
    daily["date"] = daily["ds"].dt.normalize()
    daily = daily.groupby("date", as_index=False)["trip_count_adj"].sum()
    fig = px.line(daily, x="date", y="trip_count_adj", labels={"trip_count_adj": "Pickups per day", "date": ""},
                  title=f"Daily pickups – {series}")
    fig.update_traces(line_color=INDIGO, line_width=2, hovertemplate="%{x|%a %d %b %Y}<br>%{y:,.0f} pickups<extra></extra>")
    fig.update_layout(hovermode="x unified")
    for day, label in [("2025-11-27", "Thanksgiving"), ("2025-12-25", "Christmas"), ("2026-02-23", "Snowstorm")]:
        row = daily[daily["date"] == day]
        if len(row):
            fig.add_annotation(x=day, y=float(row["trip_count_adj"].iloc[0]), text=label, showarrow=True,
                               arrowhead=0, ay=35, font=dict(size=11, color=INK_2))
    show_chart(fig, height=340)

    airports = read_csv(EDA_TAB / "C5_airport_hourly.csv")
    left, right = st.columns([1.2, 1])
    with left:
        fig = px.line(airports, x="hour", y=["JFK Airport", "LaGuardia Airport"], markers=True,
                      labels={"value": "Average pickups / hour", "hour": "Hour of day", "variable": ""},
                      color_discrete_sequence=[INDIGO, AMBER], title="Airports peak earlier than Manhattan")
        fig.update_traces(line_width=2, marker_size=6)
        fig.update_layout(hovermode="x unified")
        show_chart(fig, height=330)
    with right:
        st.write("")
        callout("<b>Peak hours differ by place:</b> LaGuardia peaks at <b>2 pm</b>, JFK at <b>4 pm</b>, "
                "Manhattan at <b>6 pm</b>. Weekend nights 00–03 are 4.7× busier than weekday nights.")
        callout("<b>Use:</b> stage airport cabs 30–60 minutes before 2 pm (LGA) and 4 pm (JFK), "
                "then move them into Manhattan for the 6 pm peak.", "use")

# ---------------------------------------------------------------------------
with tab_weather:
    kpi_row([
        ("Heavy snow (>0.5 cm)", "−40% to −70%", "fewer pickups"),
        ("Rain, AM rush", "+14.6%", "Manhattan (Queens +14.7%)"),
        ("Temperature alone", "< 3%", "change in demand"),
    ])
    effects = read_csv(EDA_TAB / "weather_effects.csv")
    condition = st.segmented_control("Weather condition", effects["condition"].unique().tolist(),
                                     default="Heavy snow (> 0.5 cm)", key="weather_condition", required=True)
    chosen = effects[effects["condition"] == condition]
    fig = px.bar(chosen, x="time_band", y="change_%", color="borough", barmode="group",
                 color_discrete_map={"Manhattan": INDIGO, "Queens": AMBER},
                 labels={"change_%": "Change vs a normal dry hour (%)", "time_band": ""},
                 title=f"{condition}: change in pickups by time of day",
                 hover_data={"zone_hours": ":,"})
    fig.update_traces(marker_line_color="white", marker_line_width=2)
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    show_chart(fig, height=380)
    st.caption("Compared with a normal dry hour in the same zone, hour, weekday/weekend and month. "
               "Hover a bar to see how many zone-hours support the estimate.")
    left, right = st.columns(2)
    with left:
        figure(EDA_FIG / "F12_extreme_weather_days.png", "Extreme weather days (EDA figure F12)")
    with right:
        figure(EDA_FIG / "F10_rain_sensitivity_by_zone.png", "Rain sensitivity by zone (EDA figure F10)")
    callout("<b>Use:</b> call in extra drivers for morning rain; cut the active fleet sharply when a "
            "heavy-snow warning is issued.", "use")

# ---------------------------------------------------------------------------
with tab_events:
    holidays = read_csv(EDA_TAB / "holiday_effects.csv")
    special = read_csv(EDA_TAB / "E4_special_day_effects.csv")
    events = pd.concat([holidays.assign(kind="Federal holiday"), special.assign(kind="Special day")])
    events = events[events["scope"] == "All"].copy()
    events = pd.concat([events, pd.DataFrame([{
        "date": "2025-12-31", "event": "New Year's Eve night", "pct_change": 62.0, "kind": "Special day"}])])
    events["label"] = events["event"] + " (" + pd.to_datetime(events["date"]).dt.strftime("%d %b") + ")"
    events = events.sort_values("pct_change")
    events["direction"] = np.where(events["pct_change"] >= 0, "More pickups", "Fewer pickups")
    fig = px.bar(events, x="pct_change", y="label", orientation="h", color="direction",
                 color_discrete_map={"More pickups": INDIGO, "Fewer pickups": ROSE},
                 labels={"pct_change": "Change vs matched normal days (%)", "label": ""},
                 title="Every unusual day has a reason", hover_data={"kind": True, "direction": False})
    fig.update_traces(texttemplate="%{x:+.0f}%", textposition="outside", cliponaxis=False)
    show_chart(fig, height=430)
    kpi_row([
        ("23 Feb snowstorm", "−78%", "worst day: 23 K trips"),
        ("Christmas Day", "−58%", "vs a normal Thursday"),
        ("Pre-Christmas peak", "+32%", "170 K trips a day"),
        ("New Year's Eve night", "+62%", "20:00 – 04:59"),
    ])
    st.write("")
    figure(EDA_FIG / "E5_anomaly_days.png", "Anomaly days detected in the daily series (EDA figure E5)")

# ---------------------------------------------------------------------------
with tab_zone:
    st.caption("Pick any zone to see its own demand pattern.")
    default = int(zones.index[zones["Zone"] == "East Village"][0])
    label = st.selectbox("Zone", zones["label"], index=default, key="explore_zone")
    zone = zones[zones["label"] == label].iloc[0]
    hourly = load_zone_hourly()[str(zone["PULocationID"])]
    hourly = hourly[hourly.index >= "2025-09-08"]
    frame = pd.DataFrame({"pickups": hourly.values, "hour": hourly.index.hour, "dow": hourly.index.dayofweek})
    peak = frame.groupby(["dow", "hour"])["pickups"].mean().idxmax()

    kpi_row([
        ("Average", f"{zone['avg_per_hour']:.1f}", "pickups per hour"),
        ("Total pickups", f"{int(zone['total_trips']):,}", "8 Sep – 28 Feb"),
        ("Busiest slot", f"{DAY_NAMES[peak[0]]} {peak[1]:02d}:00", "on average"),
        ("Zone type", str(zone["cluster_name"]).split(" (")[0], f"volume tier: {zone['tier']}"),
    ])
    grid = frame.pivot_table(index="dow", columns="hour", values="pickups", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[f"{h:02d}" for h in grid.columns], y=DAY_NAMES,
        colorscale=[[i / 6, c] for i, c in enumerate(SEQ)], xgap=2, ygap=2,
        hovertemplate="%{y} %{x}:00<br>%{z:.1f} pickups per hour<extra></extra>",
        colorbar=dict(title="Pickups / hour", thickness=12),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"{zone['Zone']}: average pickups by weekday and hour")
    show_chart(fig, height=330)

    frame["day_type"] = np.where(frame["dow"] >= 5, "Weekend", "Weekday")
    profile = frame.groupby(["day_type", "hour"], as_index=False)["pickups"].mean()
    fig = px.line(profile, x="hour", y="pickups", color="day_type", markers=True,
                  color_discrete_map={"Weekday": INDIGO, "Weekend": TEAL},
                  labels={"pickups": "Average pickups / hour", "hour": "Hour of day", "day_type": ""},
                  title="Weekday vs weekend daily shape")
    fig.update_traces(line_width=2, marker_size=6)
    fig.update_layout(hovermode="x unified")
    show_chart(fig, height=320)

with st.expander("More figures from the original EDA notebooks"):
    gallery = [
        ("D7_hour_weekday_heatmap.png", "Hour × weekday heatmap"),
        ("C2_lorenz_curve.png", "Lorenz curve of zone demand"),
        ("C8_cluster_weekly_profiles.png", "Weekly profiles of the three zone clusters"),
        ("D2_daily_demand_trend.png", "Daily demand trend"),
        ("I5_night_hotspot_map.png", "Night hotspots"),
        ("G1_feature_correlations.png", "Feature correlations"),
    ]
    for i in range(0, len(gallery), 2):
        cols = st.columns(2)
        for col, (file_name, caption) in zip(cols, gallery[i:i + 2]):
            with col:
                figure(EDA_FIG / file_name, caption)

