import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (INDIGO, INK, MODEL_COLORS, MODELING, AMBER, callout, hero, kpi_row,
                   load_test_predictions, load_zones, read_csv, show_chart)

hero(
    "Chapter 3 · Review 1",
    "Predicting next-hour demand with machine learning",
    "A regression task: predict <b>trip_count</b>, the number of pickups in a zone during an hour, "
    "from recent demand, location, calendar and weather. We compared three models against a simple baseline.",
)

# ---------------------------------------------------------------------------
st.header("The prediction setup")
left, right = st.columns([1, 1.2])
with left:
    st.markdown(
        "**Inputs (14 features)**\n"
        "- **Recent demand:** pickups 1 h, 24 h and 168 h ago; 3 h and 24 h rolling averages\n"
        "- **Location & time:** pickup zone, hour, day of week, month, holiday flag\n"
        "- **Weather:** apparent temperature, rain, snowfall, precipitation probability\n\n"
        "**Split:** chronological 80 : 20 – no shuffling, because demand depends on time. "
        "All zones at the same hour fall in the same partition."
    )
with right:
    fig = go.Figure()
    fig.add_bar(y=["Split"], x=[3340], orientation="h", name="Train · 460,920 rows · 8 Sep → 25 Jan",
                marker_color=INDIGO, hovertemplate="Training: 3,340 hours<extra></extra>")
    fig.add_bar(y=["Split"], x=[836], orientation="h", name="Test · 115,368 rows · 25 Jan → 28 Feb",
                marker_color=AMBER, hovertemplate="Testing: 836 hours<extra></extra>")
    fig.update_layout(barmode="stack", height=170, showlegend=True, xaxis_title="Hours",
                      yaxis=dict(showticklabels=False), margin=dict(l=10, r=10, t=40, b=10))
    show_chart(fig)

st.subheader("Four approaches, from simple to strong")
c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "Naive weekly baseline", "Repeats the pickups of the same zone and hour one week (168 h) earlier. The benchmark to beat."),
    (c2, "Linear Regression", "Interpretable baseline with one-hot encoded zones and calendar. Captures linear relationships only."),
    (c3, "Random Forest", "150 trees, max depth 20. Captures non-linear effects and interactions between time, place and demand."),
    (c4, "HistGradientBoosting", "Boosted trees with Poisson loss for count data and native categorical handling. **Selected model.**"),
]
for col, title, body in cards:
    with col:
        with st.container(border=True, height=190):
            st.markdown(f"**{title}**")
            st.caption(body)

# ---------------------------------------------------------------------------
st.header("Results on the unseen test period")
results = pd.DataFrame({
    "Model": ["Naive 168-hour baseline", "Linear Regression", "Random Forest", "HistGradientBoosting"],
    "MAE": [9.3074, 7.4861, 6.7993, 5.7841],
    "RMSE": [24.0566, 15.7036, 15.2593, 13.1632],
    "R²": [0.8776, 0.9478, 0.9507, 0.9633],
    "WAPE (%)": [27.6246, 22.2189, 20.1805, 17.1674],
})
results["MAE vs baseline"] = (results["MAE"] / results["MAE"].iloc[0] - 1) * 100
short_names = {"Naive 168-hour baseline": "Naive<br>baseline", "Linear Regression": "Linear<br>Regression",
               "Random Forest": "Random<br>Forest", "HistGradientBoosting": "HistGradient-<br>Boosting"}

kpi_row([
    ("Best model", "HistGB", "HistGradientBoosting, Poisson loss"),
    ("Test MAE", "5.78", "pickups per zone-hour"),
    ("Test R²", "0.963", "96% of the variation explained"),
    ("Error reduction", "−37.9%", "MAE vs the naive baseline"),
])
st.write("")
left, right = st.columns([1, 1])
with left:
    metric = st.segmented_control("Metric", ["MAE", "RMSE", "R²"], default="MAE", key="metric_choice", required=True)
    fig = px.bar(results, x="Model", y=metric, color="Model", color_discrete_map=MODEL_COLORS, text=metric,
                 title=f"Test {metric} by model ({'higher' if metric == 'R²' else 'lower'} is better)")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False,
                      marker_line_color="white", marker_line_width=2)
    fig.update_layout(showlegend=False, xaxis_title="")
    fig.update_xaxes(tickvals=results["Model"], ticktext=[short_names[m] for m in results["Model"]], tickangle=0)
    if metric == "R²":
        fig.update_yaxes(range=[0.8, 1])
    show_chart(fig, height=380)
with right:
    st.write("")
    st.write("")
    st.dataframe(
        results, hide_index=True, width="stretch", placeholder="–",
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.3f"),
            "RMSE": st.column_config.NumberColumn(format="%.3f"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
            "WAPE (%)": st.column_config.NumberColumn(format="%.1f", help="Total absolute error as a share of total pickups"),
            "MAE vs baseline": st.column_config.NumberColumn(format="%+.1f%%"),
        },
    )
    callout("<b>MAE</b> = average error in pickups. <b>RMSE</b> punishes big misses more. "
            "<b>R²</b> = share of demand variation explained. <b>WAPE</b> = total error as a % of total pickups (lower is better).")

# Train vs test check
gaps = read_csv(MODELING / "model_train_test_comparison.csv")
gaps["Model"] = ["Linear Regression", "Random Forest", "HistGradientBoosting"]
gap_long = gaps.melt(id_vars="Model", value_vars=["MAE_train", "MAE_test"], var_name="Dataset", value_name="MAE")
gap_long["Dataset"] = gap_long["Dataset"].map({"MAE_train": "Train", "MAE_test": "Test"})
left, right = st.columns(2)
with left:
    fig = px.bar(gap_long, x="Model", y="MAE", color="Dataset", barmode="group",
                 color_discrete_map={"Train": INDIGO, "Test": AMBER}, title="No major overfitting: train vs test MAE")
    fig.update_traces(marker_line_color="white", marker_line_width=2, hovertemplate="%{x}<br>MAE %{y:.3f}<extra></extra>")
    fig.update_layout(xaxis_title="")
    show_chart(fig, height=340)
with right:
    importance = read_csv(MODELING / "model_3_permutation_importance.csv").sort_values("importance_mean")
    fig = px.bar(importance, x="importance_mean", y="feature", orientation="h", error_x="importance_std",
                 labels={"importance_mean": "Increase in MAE when the feature is shuffled", "feature": ""},
                 title="What drives the HistGradientBoosting prediction")
    fig.update_traces(marker_color=INDIGO, hovertemplate="%{y}<br>+%{x:.3f} MAE<extra></extra>")
    show_chart(fig, height=340)

callout("<b>Insight:</b> the previous hour's demand is by far the strongest predictor, followed by the "
        "24-hour lag, 24-hour rolling average, 1-week lag, pickup zone and hour of day. Rankings describe "
        "predictive usefulness, not causal effects.")

# ---------------------------------------------------------------------------
st.header("Where and when the models make mistakes")
by_hour = read_csv(MODELING / "model_error_by_hour.csv")
by_hour = by_hour.rename(columns={"baseline_mae": "Naive 168-hour baseline", "model_1_mae": "Linear Regression",
                                  "model_2_mae": "Random Forest", "model_3_mae": "HistGradientBoosting"})
long = by_hour.melt(id_vars="hour", value_vars=list(MODEL_COLORS), var_name="Model", value_name="MAE")
fig = px.line(long, x="hour", y="MAE", color="Model", color_discrete_map=MODEL_COLORS, markers=True,
              labels={"hour": "Hour of day", "MAE": "Test MAE (pickups)"}, title="Error by hour of day")
fig.update_traces(line_width=2, marker_size=6)
fig.update_layout(hovermode="x unified")
show_chart(fig, height=360)
callout("Errors are largest in the <b>evening rush</b>, when demand is highest and changes fastest. "
        "The skewed target (mean 37.7, median 4) also pulls predictions toward typical demand, so "
        "extreme peaks are under-predicted.", "warn")

st.subheader("Actual vs predicted – pick a zone and a week")
zones = load_zones()
preds = load_test_predictions()
col_a, col_b, col_c = st.columns([1.3, 1, 1.6])
default_zone = int(zones.index[zones["Zone"] == "JFK Airport"][0])
zone_label = col_a.selectbox("Zone", zones["label"], index=default_zone, key="m_zone")
week_starts = pd.date_range("2026-01-26", "2026-02-23", freq="7D")
week = col_b.selectbox("Week starting", week_starts, format_func=lambda d: d.strftime("%a %d %b %Y"), index=2)
model_names = {"baseline": "Naive 168-hour baseline", "linear_regression": "Linear Regression",
               "random_forest": "Random Forest", "hist_gradient_boosting": "HistGradientBoosting"}
shown = col_c.multiselect("Models", list(model_names.values()), default=["Naive 168-hour baseline", "HistGradientBoosting"])

zone_id = int(zones.loc[zones["label"] == zone_label, "PULocationID"].iloc[0])
view = preds[(preds["PULocationID"] == zone_id) & (preds["pickup_hour"] >= week)
             & (preds["pickup_hour"] < week + pd.Timedelta(days=7))]
fig = go.Figure()
fig.add_scatter(x=view["pickup_hour"], y=view["actual"], name="Actual", line=dict(color=INK, width=2))
for column, name in model_names.items():
    if name in shown:
        fig.add_scatter(x=view["pickup_hour"], y=view[column], name=name,
                        line=dict(color=MODEL_COLORS[name], width=2, dash="dot" if column == "baseline" else "solid"))
fig.update_layout(hovermode="x unified", yaxis_title="Pickups per hour", title=f"{zone_label}: actual vs predicted")
show_chart(fig, height=380)
if len(view) and shown:
    scores = {name: (view[col] - view["actual"]).abs().mean() for col, name in model_names.items() if name in shown}
    st.caption("MAE for this zone and week: " + " · ".join(f"**{n}** {v:.2f}" for n, v in scores.items()))

callout("<b>How it helps users:</b> with an average error of under 6 pickups per zone-hour, a dispatcher "
        "can see an hour in advance which zones will need cabs. Try it in <b>Demand predictor</b>.", "use")
