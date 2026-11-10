import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (ARM_FINDINGS, INDIGO, DEMAND_COLORS, GRAY, ITEM_TEXT, AMBER, SEQ, callout,
                   describe_rule, hero, kpi_row, load_rules, read_csv, show_chart)

hero(
    "Chapter 5 · Review 2",
    "Demand rules: if these conditions, then this demand",
    "Association rule mining finds combinations of time, place and weather that reliably come with "
    "High, Medium or Low demand. Each rule reads like an instruction a dispatcher can follow.",
)

# ---------------------------------------------------------------------------
st.header("Turning zone-hours into transactions")
left, right = st.columns([1, 1.1])
with left:
    st.markdown(
        "**One transaction = one zone in one hour**, described by 8 category items. "
        "All categories were defined on the **training period only** and then applied unchanged "
        "to the test period:\n"
        "- **Demand** is Low / Medium / High *compared with the zone's own history* "
        "(zone-specific thirds), so a quiet zone can still have a *High* hour.\n"
        "- **Temperature** is split at −0.9 °C and 9.9 °C (training thirds).\n"
        "- **Zone pattern** comes from k-means on each zone's weekly profile.\n\n"
        "Only the **96 zones** averaging ≥ 1 pickup per hour are mined."
    )
with right:
    items = pd.DataFrame([
        ["Time band", "Late night, early morning, AM rush, midday, PM rush, evening"],
        ["Day type", "Weekday / weekend"],
        ["Holiday", "Regular / federal holiday"],
        ["Rain · Snow", "Present / absent (two separate flags)"],
        ["Temperature", "Lower / middle / upper"],
        ["Zone pattern", "Daytime / morning commute / nightlife"],
        ["Demand (target)", "Low / Medium / High per zone"],
    ], columns=["Item group", "Categories"])
    st.dataframe(items, hide_index=True, width="stretch")

kpi_row([
    ("Training transactions", "320,544", "Sep 2025 – Jan 2026"),
    ("Test transactions", "80,256", "late Jan – Feb 2026"),
    ("Min support", "0.005", "keeps niche nightlife peaks"),
    ("Min confidence / lift", "0.50 / 1.20", "≥ 50% likely, ≥ 20% above chance"),
])

# ---------------------------------------------------------------------------
st.header("Apriori vs FP-Growth")
algos = read_csv(ARM_FINDINGS / "arm_algorithm_comparison.csv")
left, right = st.columns([1.2, 1])
with left:
    fig = px.line(algos, x="min_support", y="runtime_seconds", color="algorithm", markers=True,
                  color_discrete_map={"Apriori": AMBER, "FP-Growth": INDIGO}, log_x=True,
                  labels={"min_support": "Minimum support (log scale)", "runtime_seconds": "Runtime (seconds)", "algorithm": ""},
                  title="Runtime as the support threshold falls")
    fig.update_traces(line_width=2, marker_size=9)
    fig.update_xaxes(autorange="reversed")
    show_chart(fig, height=330)
with right:
    st.write("")
    callout("Both algorithms found <b>exactly the same</b> 6,719 itemsets and 897 demand rules. Apriori "
            "generates candidates and rescans the data, so it slowed from 4.5 s to 28.5 s. FP-Growth builds "
            "a compact tree in two passes and stayed near 1.3 s (<b>≈ 22× faster</b>). FP-Growth at support "
            "0.005 was selected.")

# ---------------------------------------------------------------------------
st.header("From 897 candidates to 152 trustworthy rules")
left, right = st.columns([1, 1.1])
with left:
    fig = go.Figure(go.Funnel(
        y=["Mined on training data", "Stable on the unseen test period", "Non-redundant (final)"],
        x=[897, 632, 152], texttemplate="%{value}<br>%{percentInitial:.0%}",
        marker=dict(color=[SEQ[2], SEQ[4], SEQ[6]]),
        hovertemplate="%{y}<br>%{x} rules<extra></extra>",
    ))
    fig.update_layout(title="Rule filtering", margin=dict(l=10, r=10, t=50, b=10))
    show_chart(fig, height=300)
with right:
    st.markdown(
        "**Holdout validation** – a rule is kept only if, on the test period, it still has\n"
        "- at least 100 matching transactions,\n- confidence ≥ 0.50 and lift ≥ 1.20.\n\n"
        "→ 265 seasonal or low-volume rules dropped.\n\n"
        "**Redundancy pruning** – a longer rule is removed when a shorter rule with the same "
        "result is at least as confident (e.g. adding *no snow* adds nothing).\n\n"
        "→ 480 redundant rules removed (83% total reduction). *Time band* was never dropped: it anchors every rule."
    )

# ---------------------------------------------------------------------------
st.header("Explore the 152 final rules")
rules = load_rules()
col_a, col_b, col_c = st.columns([1.2, 1, 1.3])
tiers = col_a.pills("Demand result", ["High", "Medium", "Low"], selection_mode="multi", default=["High", "Medium", "Low"])
min_lift = col_b.slider("Minimum test lift", 1.2, 3.0, 1.2, 0.1)
# Only offer conditions that really appear in the final rules.
present = sorted({item for items in rules["conditions"] for item in items})
keyword = col_c.selectbox("Rule must contain", ["(anything)"] + present,
                          format_func=lambda item: item if item == "(anything)" else ITEM_TEXT.get(item, item))
shown = rules[rules["demand"].isin(tiers or []) & (rules["test_lift"] >= min_lift)]
if keyword != "(anything)":
    shown = shown[shown["conditions"].apply(lambda items: keyword in items).astype(bool)]
shown = shown.assign(rule=shown["conditions"].apply(describe_rule))

if shown.empty:
    st.info("No rule matches these filters. Lower the minimum lift or pick another condition.", icon=":material/filter_alt_off:")
else:
    left, right = st.columns(2)
    with left:
        fig = px.scatter(shown, x="support", y="confidence", size="test_lift", color="demand",
                         color_discrete_map=DEMAND_COLORS, hover_name="rule", size_max=20,
                         category_orders={"demand": ["High", "Medium", "Low"]},
                         hover_data={"test_confidence": ":.2f", "test_lift": ":.2f", "support": ":.4f", "demand": False},
                         labels={"support": "Training support", "confidence": "Training confidence", "demand": "Demand"},
                         title="Support vs confidence (bubble size = test lift)")
        fig.update_traces(marker_line_color="white", marker_line_width=2)
        show_chart(fig, height=380)
    with right:
        fig = px.scatter(shown, x="lift", y="test_lift", color="demand", color_discrete_map=DEMAND_COLORS,
                         category_orders={"demand": ["High", "Medium", "Low"]}, hover_name="rule",
                         labels={"lift": "Training lift", "test_lift": "Test lift", "demand": "Demand"},
                         title="Rules hold up on unseen data (dashed line = equal lift)")
        fig.update_traces(marker=dict(size=9, line=dict(color="white", width=2)))
        fig.add_shape(type="line", x0=1.2, y0=1.2, x1=3.3, y1=3.3, line=dict(color=GRAY, dash="dash", width=1))
        show_chart(fig, height=380)

    st.dataframe(
        shown.sort_values("test_lift", ascending=False)[["rule", "demand", "confidence", "test_confidence", "test_lift", "test_antecedent_count"]],
        hide_index=True, width="stretch", height=320,
        column_config={
            "rule": st.column_config.TextColumn("IF these conditions …", width="large"),
            "demand": "… THEN demand is",
            "confidence": st.column_config.NumberColumn("Train confidence", format="%.2f"),
            "test_confidence": st.column_config.ProgressColumn("Test confidence", format="%.2f", min_value=0, max_value=1),
            "test_lift": st.column_config.NumberColumn("Test lift", format="%.2f×"),
            "test_antecedent_count": st.column_config.NumberColumn("Test matches", format="%d"),
        },
    )
st.caption(f"{len(shown)} of 152 rules shown. Lift 2.7× = this demand level is 2.7 times more likely than usual under these conditions.")
callout("<b>Why is there no 'raining' or 'snowing' rule?</b> Rain and snow hours are rare in the data, so rules "
        "built on them either fell below the minimum support or did not hold up on the test period. Weather "
        "appears only as <i>no rain</i> / <i>no snow</i>, and pruning showed it adds little. This matches the "
        "machine-learning result that weather adds little beyond time, place and recent demand.", "warn")

st.subheader("Which conditions drive each demand level?")
families = ["time_band", "day_type", "holiday", "temperature", "rain", "snow", "zone_cluster"]
rows = []
for demand, group in rules.groupby("demand"):
    for family in families:
        share = group["conditions"].apply(lambda items: any(i.startswith(family + "=") for i in items)).mean()
        rows.append({"demand": demand, "family": family, "share": 100 * share})
heat = pd.DataFrame(rows).pivot(index="demand", columns="family", values="share").loc[["High", "Medium", "Low"], families]
fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, xgap=2, ygap=2,
                           colorscale=[[i / 6, c] for i, c in enumerate(SEQ)], text=heat.values.round(0),
                           texttemplate="%{text:.0f}%", hovertemplate="%{y} demand rules<br>%{z:.0f}% contain %{x}<extra></extra>",
                           colorbar=dict(title="% of rules", thickness=12)))
fig.update_layout(title="Share of rules that use each condition")
show_chart(fig, height=280)

st.header("Headline rules")
c1, c2 = st.columns(2)
with c1:
    callout("<b>Morning commute surge:</b> weekday AM rush (07–09) in commuter zones → <b>High</b> demand "
            "with 70–73% confidence and <b>2.6–2.7× lift</b>. Pre-position cabs in residential hubs before 07:00.", "use")
with c2:
    callout("<b>Nightlife surge:</b> evenings (20–23) in nightlife zones → <b>High</b> demand with 58–63% "
            "confidence and <b>2.2–2.3× lift</b>; weekends in nightlife zones → High with 55% confidence (2.0× lift). "
            "Move evening cabs into nightlife zones, especially on Friday and Saturday.", "use")
st.caption("Limitation: rules give a demand *level* (Low / Medium / High for that zone), not an exact count of pickups.")
