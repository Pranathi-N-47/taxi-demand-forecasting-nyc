"""Reproducible Review 1 EDA for ArunN2005 (issues 63--66).

Run from any directory: python scripts/eda_arun.py --issues 63
Inputs are read only; outputs go exclusively to EDA_doc/.
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EDA_doc"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
DST = pd.Timestamp("2025-11-02 01:00")
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# Recurring US federal calendar holidays used by the upstream feature pipeline.
# Source: https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/
HOLIDAYS = {
    "2025-09-01": "Labor Day", "2025-10-13": "Columbus Day",
    "2025-11-11": "Veterans Day", "2025-11-27": "Thanksgiving Day",
    "2025-12-25": "Christmas Day", "2026-01-01": "New Year's Day",
    "2026-01-19": "Martin Luther King Jr. Day", "2026-02-16": "Washington's Birthday",
}
COLORS = {"Manhattan": "#2878ad", "Queens": "#e78632",
          "Yellow Zone": "#d2a900", "Boro Zone": "#27836a", "Airports": "#9056a0"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .18, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def table(frame, name, index=False):
    frame.to_csv(TABLES / name, index=index, float_format="%.8g")


def md_table(frame, columns=None, limit=None):
    """Small Markdown tables without an optional tabulate dependency."""
    if columns:
        frame = frame[columns]
    if limit:
        frame = frame.head(limit)
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
    return "| " + " | ".join(map(str, frame.columns)) + " |\n|" + "---|" * len(frame.columns) + "\n" + "\n".join(rows)


def finding(code, title, text, evidence, implication):
    links = []
    for file in evidence:
        folder = "figures" if file.endswith(".png") else "tables"
        links.append(f"[{file}]({folder}/{file})")
    return (f"\n**[{code}] {title}**\n\n- **Finding:** {text}\n"
            f"- **Evidence:** {', '.join(links)}\n- **Implication:** {implication}\n")


def write_section(issue, title, content):
    path = OUT / "EDA_findings.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = """# EDA Findings

ArunN2005's Review 1 EDA for issues #63--#66. Reproduce with
`python scripts/eda_arun.py` or the companion `scripts/EDA_Arun.ipynb`.
Inputs: `data/processed/features.csv` and the existing zone/weather lookup files.
Analysis covers **8 Sep 2025 to 28 Feb 2026**, 174 complete local calendar days,
138 zones and 4,176 recorded hour labels per zone. Counts combine yellow and green
taxi pickups; `service_zone` describes an area, not a taxi fleet.

Zero counts are retained. All timestamps are naive New York local time. The
repeated 01:00 on 2 Nov 2025 combines two physical hours; rates here use recorded
hour labels unless explicitly stated. Partial weeks and September's missing first
week are labelled. Observed pickups measure completed service, not unmet rider
demand or required fleet size. Descriptive comparisons do not establish causation.

The other team EDA lives on its own branch; this file contains only Arun's assigned
IDs. Generated sections can be rerun without replacing unrelated sections.
"""
    start, end = f"<!-- arun-issue-{issue}:start -->", f"<!-- arun-issue-{issue}:end -->"
    block = f"{start}\n\n## {title} (#{issue})\n{content}\n{end}"
    pattern = re.escape(start) + r".*?" + re.escape(end)
    if re.search(pattern, text, flags=re.S):
        text = re.sub(pattern, lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def load_data():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data/processed/features.csv", parse_dates=["pickup_hour"])
    zones = pd.read_csv(ROOT / "data/raw/taxi_zones_manhattan_queens.csv")
    data = data.merge(zones, left_on="PULocationID", right_on="LocationID", how="left", validate="many_to_one")
    if data["Zone"].isna().any():
        raise ValueError("Unmatched pickup zones")
    data = data.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)
    data["date"] = data.pickup_hour.dt.normalize()
    data["year_month"] = data.pickup_hour.dt.to_period("M").astype(str)
    data["week_start"] = data.date - pd.to_timedelta(data.day_of_week, unit="D")
    return data, zones


def zone_stats(data):
    result = data.groupby("PULocationID").trip_count.agg(
        rows="size", total="sum", mean="mean", variance="var", maximum="max",
        zero_share=lambda x: x.eq(0).mean(), p95=lambda x: x.quantile(.95), p99=lambda x: x.quantile(.99))
    result["variance_mean_ratio"] = result.variance.div(result["mean"].replace(0, np.nan))
    return result.join(data.groupby("PULocationID")[["Zone", "Borough", "service_zone", "latitude", "longitude"]].first())


def issue63(data, zones):
    features = pd.read_csv(ROOT / "data/processed/features.csv", parse_dates=["pickup_hour"])
    columns = ["PULocationID", "pickup_hour", "trip_count", "apparent_temperature", "rain", "snowfall",
               "precipitation_probability", "hour", "day_of_week", "month", "is_weekend", "is_holiday",
               "trip_count_lag_1h", "trip_count_lag_24h", "trip_count_lag_168h", "trip_count_roll_3h", "trip_count_roll_24h"]
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})

    check("A1 schema and nulls", features.columns.tolist() == columns and not features.isna().any().any(), "17 expected columns; zero nulls")
    check("A2 unique zone-hours", not features.duplicated(["PULocationID", "pickup_hour"]).any(), f"{len(features):,} rows")
    hours = pd.date_range("2025-09-08", "2026-02-28 23:00", freq="h")
    sizes = features.groupby("PULocationID").size()
    check("A2 complete rectangular grid", set(features.PULocationID) == set(zones.LocationID)
          and len(features) == len(zones) * len(hours) and sizes.eq(len(hours)).all()
          and set(features.pickup_hour) == set(hours), "138 zones x 4,176 local hour labels = 576,288")
    ints = ["PULocationID", "trip_count", "precipitation_probability", "hour", "day_of_week", "month",
            "trip_count_lag_1h", "trip_count_lag_24h", "trip_count_lag_168h"]
    check("A1 numeric and Boolean types", all(pd.api.types.is_integer_dtype(features[c]) for c in ints)
          and all(pd.api.types.is_float_dtype(features[c]) for c in ["apparent_temperature", "rain", "snowfall", "trip_count_roll_3h", "trip_count_roll_24h"])
          and all(pd.api.types.is_bool_dtype(features[c]) for c in ["is_weekend", "is_holiday"]), "integer counts/calendar, float weather/rolling, Boolean flags")
    numeric = features.select_dtypes("number")
    check("A3 finite values", np.isfinite(numeric).all().all(), "all numeric fields finite")
    nonnegative = [c for c in numeric if c not in ["apparent_temperature"]]
    check("A3 count and weather ranges", (features[nonnegative] >= 0).all().all()
          and features.precipitation_probability.between(0, 100).all()
          and features.hour.between(0, 23).all() and features.day_of_week.between(0, 6).all()
          and features.month.isin([9, 10, 11, 12, 1, 2]).all(), "nonnegative counts/rain/snow; probability 0--100; calendar ranges valid")
    check("A3 calendar derivation", features.hour.eq(features.pickup_hour.dt.hour).all()
          and features.day_of_week.eq(features.pickup_hour.dt.dayofweek).all()
          and features.month.eq(features.pickup_hour.dt.month).all(), "matches pickup_hour")
    expected_holidays = features.pickup_hour.dt.strftime("%Y-%m-%d").isin(HOLIDAYS)
    check("A5 weekend and holiday flags", features.is_weekend.eq(features.day_of_week.ge(5)).all()
          and features.is_holiday.eq(expected_holidays).all(), f"{features.is_holiday.sum():,} holiday rows; 7 dates; Labor Day is in warm-up")
    for borough in ["Manhattan", "Queens"]:
        split = pd.read_csv(ROOT / f"data/processed/features_{borough.lower()}.csv", parse_dates=["pickup_hour"])
        expected = features.loc[features.PULocationID.isin(zones.loc[zones.Borough.eq(borough), "LocationID"])]
        split = split.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)
        expected = expected.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)
        check(f"A6 {borough} exact split", split.equals(expected), f"{len(split):,} rows; all columns and values compared")
    report = (ROOT / "data/processed/cleaning_report.md").read_text(encoding="utf-8")
    clean_total = int(re.search(r"Final clean rows:\*\* ([\d,]+)", report).group(1).replace(",", ""))
    retained = int(features.trip_count.sum())
    # These lags recover Sep 1--7 once each, rather than summing overlapping lags.
    first_week = features.loc[features.pickup_hour.lt(pd.Timestamp("2025-09-15"))]
    warmup = int(first_week.trip_count_lag_168h.sum())
    check("A7 trip reconciliation", retained + warmup == clean_total,
          f"{retained:,} retained + {warmup:,} warm-up = {clean_total:,} cleaned pickups")
    raw_total = int(re.search(r"\| Raw input \| -- \| ([\d,]+)", report).group(1).replace(",", ""))
    audit = pd.read_csv(ROOT / "data/raw/taxi_collection_summary.csv")
    check("A7 raw collection audit", int(audit.rows_kept.sum()) == raw_total, f"{raw_total:,} raw pickups; {raw_total-clean_total:,} removed")
    weather = pd.read_csv(ROOT / "data/raw/weather_hourly_by_zone.csv", parse_dates=["weather_hour"])
    weather_cols = ["apparent_temperature", "rain", "snowfall", "precipitation_probability"]
    joined = features.merge(weather, left_on=["PULocationID", "pickup_hour"], right_on=["LocationID", "weather_hour"],
                            how="left", validate="one_to_one", suffixes=("", "_weather"))
    check("A8 weather join integrity", all(np.allclose(joined[c], joined[c+"_weather"], atol=1e-5, rtol=1e-6) for c in weather_cols), "every retained zone-hour matches source weather")
    cells = weather[["LocationID", "grid_latitude", "grid_longitude", "grid_elevation_m"]].drop_duplicates()
    # Grid identity uses latitude/longitude. API elevation metadata can differ
    # for zone centroids in the same horizontal model cell.
    cell_groups = cells.groupby(["grid_latitude", "grid_longitude"]).agg(
        LocationID=("LocationID", list), elevation_min=("grid_elevation_m", "min"),
        elevation_max=("grid_elevation_m", "max")).reset_index()
    cell_groups["zone_count"] = cell_groups.LocationID.map(len)
    cell_groups["zone_ids"] = cell_groups.LocationID.map(lambda x: ",".join(map(str, sorted(x))))
    table(cell_groups.drop(columns="LocationID"), "A8_shared_weather_cells.csv")
    check("A8 stable grid assignment", len(cells) == len(zones), f"{len(cell_groups)} cells for {len(zones)} zones; max {cell_groups.zone_count.max()} zones/cell")
    dst_total = int(features.loc[features.pickup_hour.eq(DST), "trip_count"].sum())
    report_dst = int(re.search(r"([\d,]+) trips carry", report).group(1).replace(",", ""))
    check("A8 DST audit", dst_total == report_dst, f"{dst_total:,} pickups in combined 01:00 label")
    table(pd.DataFrame(checks), "A1_A8_data_checks.csv")
    table(pd.DataFrame({"column": features.columns, "dtype": features.dtypes.astype(str).values,
                        "nulls": features.isna().sum().values}), "A1_schema.csv")
    table(numeric.agg(["min", "max"]).T.rename_axis("column").reset_index(), "A3_value_ranges.csv")
    table(pd.DataFrame({"stage": ["Raw", "Removed in cleaning", "Clean", "Warm-up (recovered from 168h lag)", "Retained features"],
                        "trips": [raw_total, raw_total-clean_total, clean_total, warmup, retained]}), "A7_trip_reconciliation.csv")
    failures = [c for c in checks if not c["passed"]]
    if failures:
        raise ValueError(f"Data checks failed: {failures}")

    stats = zone_stats(data)
    table(stats.reset_index(), "B4_zone_distribution.csv")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(sizes)), sizes.values, color=[COLORS[zones.set_index("LocationID").loc[x, "Borough"]] for x in sizes.index])
    ax.axhline(len(hours), color="black", linestyle="--", label="Expected 4,176")
    ticks = np.arange(0, len(sizes), 7)
    ax.set_xticks(ticks, sizes.index[ticks], rotation=90)
    ax.set(xlabel="Pickup zone ID (all 138 zones)", ylabel="Rows", title="A2 | Every zone retains the same 4,176 hour labels")
    ax.legend(); save(fig, "A2_rows_per_zone.png")
    zeros = data.assign(zero=data.trip_count.eq(0)).pivot_table(index="PULocationID", columns="date", values="zero", aggfunc="sum")
    table(zeros.rename_axis("PULocationID").reset_index(), "A2_zero_hours_by_zone_date.csv")
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(zeros, aspect="auto", interpolation="nearest", vmin=0, vmax=24, cmap="magma")
    xt = np.arange(0, len(zeros.columns), 14); yt = np.arange(0, len(zeros), 7)
    ax.set_xticks(xt, [zeros.columns[i].strftime("%d %b") for i in xt], rotation=45)
    ax.set_yticks(yt, zeros.index[yt]); ax.set(xlabel="Local date", ylabel="Pickup zone ID", title="A2 | Zero-demand hour labels by zone and date")
    fig.colorbar(im, ax=ax, label="Zero hours per day (0--24)"); save(fig, "A2_zero_hour_heatmap.png")
    sundays = pd.to_datetime(["2025-10-19", "2025-10-26", "2025-11-02", "2025-11-09", "2025-11-16"])
    dst_compare = data.loc[data.date.isin(sundays)].groupby(["date", "hour"]).trip_count.sum().unstack("date")
    table(dst_compare.reset_index(), "A8_dst_comparison.csv")
    normal_01 = dst_compare.loc[1].drop(pd.Timestamp("2025-11-02")).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    for date in sundays:
        ax.plot(dst_compare.index, dst_compare[date], label=date.strftime("%d %b"), linewidth=2.5 if date.day == 2 else 1.2)
    ax.axvspan(.7, 1.3, color="red", alpha=.12)
    ax.set(xlabel="Local hour", ylabel="Trips across 138 zones", title="A8 | DST Sunday: two physical 01:00 hours share one label")
    ax.legend(ncol=5); save(fig, "A8_dst_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(data.trip_count, bins=80, color="#2878ad"); axes[0].set(xlabel="Trips per zone-hour", ylabel="Zone-hours", title="Linear target")
    axes[1].hist(np.log1p(data.trip_count), bins=60, color="#27836a"); axes[1].set(xlabel="log(1 + trips)", ylabel="Zone-hours", title="Zero-preserving log transform")
    fig.suptitle("B1/B5 | Highly skewed pickup-count distribution"); save(fig, "B1_B5_target_histograms.png")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for col, row in [("Borough", 0), ("service_zone", 1)]:
        names = sorted(data[col].unique())
        groups = [np.log1p(data.loc[data[col].eq(n), "trip_count"].to_numpy()) for n in names]
        axes[row, 0].boxplot(groups, tick_labels=names, showfliers=False)
        rng = np.random.default_rng(2005)
        sampled = [rng.choice(g, min(8000, len(g)), replace=False) for g in groups]
        axes[row, 1].violinplot(sampled, showmedians=True)
        axes[row, 1].set_xticks(range(1, len(names)+1), names)
        for ax in axes[row]: ax.set_ylabel("log(1 + trips per zone-hour)")
    axes[0, 0].set_title("Box: all rows; outlier dots hidden")
    axes[0, 1].set_title("Violin: <=8,000 sampled rows/group; seed 2005")
    fig.suptitle("B3 | Target shapes by borough and service area"); save(fig, "B3_borough_service_shapes.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, group in [("All", data), *list(data.groupby("Borough"))]:
        values, counts = np.unique(group.trip_count, return_counts=True)
        ax.step(np.log1p(values), np.cumsum(counts)/len(group), where="post", label=label)
    ax.set(xlabel="log(1 + trips per zone-hour)", ylabel="Cumulative fraction", title="B1/B3 | ECDF retains the zero-count mass")
    ax.legend(); save(fig, "B1_B3_ecdf.png")
    zero_hour = data.assign(zero=data.trip_count.eq(0)).groupby(["hour", "Borough"]).zero.mean().unstack()
    table(zero_hour.reset_index(), "B2_zero_share_by_hour.csv")
    fig, ax = plt.subplots(figsize=(12, 4)); (zero_hour * 100).plot.bar(ax=ax, color=COLORS)
    ax.set(xlabel="Local hour", ylabel="Zero zone-hours (%)", title="B2 | Zero-demand prevalence by hour and borough")
    save(fig, "B2_zero_share_by_hour.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    positive = stats.loc[stats["mean"].gt(0)]
    for borough, group in positive.groupby("Borough"):
        ax.scatter(group["mean"], group.variance, label=borough, color=COLORS[borough], alpha=.75)
    bounds = np.geomspace(positive["mean"].min(), positive["mean"].max(), 100)
    ax.plot(bounds, bounds, "k--", label="Poisson: variance = mean")
    ax.set(xscale="log", yscale="log", xlabel="Mean pickups / zone-hour", ylabel="Sample variance", title="B4 | Pooled zone counts are overdispersed")
    ax.legend(); save(fig, "B4_mean_variance.png")
    summary = data.groupby("Borough").trip_count.agg(rows="size", total="sum", mean="mean", median="median", zero_share=lambda x: x.eq(0).mean())
    table(summary.reset_index(), "B3_borough_distribution.csv")
    quiet = stats.loc[stats["mean"].lt(1)].sort_values("total")
    table(quiet.reset_index(), "A8_near_empty_zones.csv")
    content = f"""
All **{len(checks)} checks pass**. No missing values or duplicate zone-hours; both borough
files exactly match their combined-file subsets (all 17 columns). The schema, finite
values, integer/Boolean types, calendar derivations, ranges and source-weather joins
were checked before any downstream analysis. Expected temperature limits are reported,
not clipped to the raw-data range: the retained period differs from the raw period.
Holiday validation follows the upstream recurring federal calendar definition, rather
than adding separate government closure days. [Calendar reference](https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/).

{md_table(pd.DataFrame(checks))}
"""
    content += finding("A1/A2/A3", "Schema, coverage and ranges", f"{len(features):,} rows across 138 zones, with exactly 4,176 rows per zone. All expected local hour labels are present; zero counts are observations.",
                       ["A1_schema.csv", "A3_value_ranges.csv", "A2_rows_per_zone.png", "A2_zero_hour_heatmap.png"], "Use the complete grid; dropping zeros would bias averages and temporal profiles.")
    content += finding("A5/A6", "Flags and borough splits", f"Weekend flags match Saturday/Sunday; {features.is_holiday.sum():,} rows cover seven federal holiday dates. Each borough has 69 zones and 288,144 rows.",
                       ["A1_A8_data_checks.csv"], "Labor Day is absent because Sep 1--7 was removed for lag warm-up; service area must not be interpreted as taxi type.")
    content += finding("A7", "Clean-trip reconciliation", f"{retained:,} retained trips + {warmup:,} warm-up trips = {clean_total:,} cleaned trips exactly. Warm-up demand was recovered once from lag_168h in Sep 8--14; these rows refer to Sep 1--7. Raw audit = {raw_total:,}; cleaning removed {raw_total-clean_total:,}.",
                       ["A7_trip_reconciliation.csv"], "Use retained-period totals for EDA; a direct comparison with all six months of cleaned trips would appear to lose demand.")
    content += finding("A8", "DST, sparse zones and weather cells", f"DST 01:00 has {dst_total:,} trips versus {normal_01:,.0f} on four nearby normal Sundays ({dst_total/normal_01:.2f}x, not an assumed 2x). There are {len(quiet)} zones with mean <1 pickup/hour; zones 103 and 104 are entirely zero. The 138 zones use {len(cell_groups)} weather cells, with up to {cell_groups.zone_count.max()} zones sharing a cell.",
                       ["A8_dst_comparison.png", "A8_near_empty_zones.csv", "A8_shared_weather_cells.csv"], "Do not remove sparse areas as errors. Flag DST and its lag copies; nearby Sundays differ in events/weather, and the marathon also occurs on DST day. Grid identity uses latitude/longitude; elevation metadata varies within some cells, so shared cells do not guarantee identical weather values and are not independent spatial observations.")
    content += finding("B1/B2/B3/B5", "Target distribution and zero prevalence", f"Mean={data.trip_count.mean():.2f}, median={data.trip_count.median():.0f}, maximum={data.trip_count.max():,}; {data.trip_count.eq(0).mean():.1%} of zone-hours are zero. Manhattan zero share={summary.loc['Manhattan','zero_share']:.1%}; Queens={summary.loc['Queens','zero_share']:.1%}. log1p compresses the tail while preserving zeros.",
                       ["B1_B5_target_histograms.png", "B3_borough_service_shapes.png", "B1_B3_ecdf.png", "B2_zero_share_by_hour.png", "B3_borough_distribution.csv"], "Compare count-based and zero-aware approaches. A log-target model needs retransformation validation; exp(prediction)-1 alone need not estimate the mean count.")
    content += finding("B4", "Overdispersion", f"{positive.variance.gt(positive['mean']).sum()} of {len(positive)} nonzero-demand zones have variance > mean; median variance/mean={positive.variance_mean_ratio.median():.2f}. All-zero zones have undefined variance/mean and are excluded from the log plot.",
                       ["B4_mean_variance.png", "B4_zone_distribution.csv"], "A constant-rate Poisson assumption is inadequate for pooled counts. Calendar/seasonal variation also contributes; this does not prove conditional overdispersion after modelling.")
    write_section(63, "Data checks and target distribution", content)
    print(f"#63: {len(checks)} checks passed; retained {retained:,} + warm-up {warmup:,} = clean {clean_total:,}.", flush=True)


def label_zone(zone_id, name, width=30):
    return f"{zone_id} {name[:width]}" + ("..." if len(name) > width else "")


def issue64(data, zones):
    content = "\nTotals use all retained pickups. Group means are per zone-hour and include zeros.\nWeekend ratios compare mean pickups per observed weekend/weekday hour (not totals).\n"
    for col, code in [("Borough", "C3"), ("service_zone", "C4")]:
        summary = data.groupby(col).trip_count.agg(total="sum", mean="mean", rows="size")
        summary["zones"] = data.groupby(col).PULocationID.nunique()
        summary["trip_share_pct"] = 100 * summary.total / data.trip_count.sum()
        table(summary.reset_index(), f"{code}_group_demand.csv")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(summary.index, summary.total/1e6, color=[COLORS[n] for n in summary.index])
        axes[1].bar(summary.index, summary["mean"], color=[COLORS[n] for n in summary.index])
        axes[0].set(ylabel="Retained pickups (millions)", title="Total pickups")
        axes[1].set(ylabel="Mean pickups / zone-hour", title="Normalised for number of zones")
        fig.suptitle(f"{code} | Demand by {col}"); save(fig, f"{code}_group_demand.png")
        rows = "; ".join(f"{n}: {r.total:,.0f} ({r.trip_share_pct:.1f}%), mean {r['mean']:.1f}/zone-hour across {int(r.zones)} zones" for n, r in summary.iterrows())
        content += finding(code, f"Demand by {col}", rows + ".", [f"{code}_group_demand.png", f"{code}_group_demand.csv"], "Compare both volume and per-zone rates. Service areas are geography; this table cannot separate yellow from green taxis.")
    borough_hour = data.groupby(["hour", "Borough"]).trip_count.sum().unstack()
    shares = borough_hour.div(borough_hour.sum(axis=1), axis=0) * 100
    table(shares.reset_index(), "C3_borough_hour_share.csv")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(shares.index, *[shares[c] for c in shares], labels=shares.columns, colors=[COLORS[c] for c in shares])
    ax.set(xlim=(0, 23), ylim=(0, 100), xlabel="Local hour", ylabel="Share of recorded pickups (%)", title="C3 | Borough share shifts through the day")
    ax.legend(loc="lower right"); save(fig, "C3_borough_hour_share.png")
    airport = data.loc[data.service_zone.eq("Airports")].copy()
    airport["hour_of_week"] = airport.day_of_week * 24 + airport.hour
    ap_hour = airport.groupby(["hour", "Zone"]).trip_count.mean().unstack()
    ap_week = airport.groupby(["hour_of_week", "Zone"]).trip_count.mean().unstack()
    ap_totals = airport.groupby(["week_start", "Zone"]).trip_count.sum().unstack()
    ap_days = airport.groupby("week_start").date.nunique()
    table(ap_hour.reset_index(), "C5_airport_hourly.csv")
    table(ap_week.reset_index(), "C5_airport_weekly_profile.csv")
    table(ap_totals.assign(observed_days=ap_days).reset_index(), "C5_airport_weekly_totals.csv")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    ap_hour.plot(ax=axes[0]); ap_week.plot(ax=axes[1]); ap_totals.plot(ax=axes[2])
    axes[0].set(xlabel="Local hour", ylabel="Mean pickups / hour", title="C5 | JFK and LaGuardia hourly profiles")
    axes[1].set(xlabel="Hour of week", ylabel="Mean pickups / hour", title="Weekly shape; Monday 00:00 = 0")
    axes[1].set_xticks(np.arange(0, 168, 24), DAY_NAMES)
    axes[2].set(xlabel="Week beginning Monday", ylabel="Pickups / observed week", title="Weekly totals; final week is six days")
    save(fig, "C5_airport_profiles.png")
    content += finding("C5", "Airport timing", "; ".join(f"{n} peaks at {int(ap_hour[n].idxmax()):02d}:00 ({ap_hour[n].max():.1f} pickups/hour)" for n in ap_hour) + ".",
                       ["C5_airport_profiles.png", "C5_airport_hourly.csv", "C5_airport_weekly_profile.csv", "C5_airport_weekly_totals.csv"], "Plan airport pickup positioning by its own hourly and weekday pattern; flight schedules and supply are not measured here.")
    ratio = data.groupby(["PULocationID", "is_weekend"]).trip_count.agg(mean="mean", total="sum", hours="size").unstack("is_weekend")
    ratios = pd.DataFrame({"weekday_mean": ratio["mean"][False], "weekend_mean": ratio["mean"][True],
                           "weekday_hours": ratio["hours"][False], "weekend_hours": ratio["hours"][True]})
    ratios["weekend_weekday_ratio"] = ratios.weekend_mean / ratios.weekday_mean.replace(0, np.nan)
    ratios = ratios.join(zone_stats(data)[["Zone", "Borough", "mean"]]).sort_values("weekend_weekday_ratio", ascending=False, na_position="last")
    table(ratios.reset_index(), "C9_weekend_weekday_by_zone.csv")
    fig, axes = plt.subplots(1, 2, figsize=(19, 18))
    for ax, group in zip(axes, [ratios.iloc[:69], ratios.iloc[69:]]):
        values = group.weekend_weekday_ratio
        ax.barh(range(len(group)), values.fillna(0), color=[COLORS[b] for b in group.Borough])
        ax.set_yticks(range(len(group)), [label_zone(i, n, 26) for i, n in group.Zone.items()], fontsize=8)
        ax.invert_yaxis(); ax.axvline(1, color="black", linestyle="--")
        for j, value in enumerate(values):
            if pd.isna(value): ax.text(.02, j, "undefined (zero weekday demand)", fontsize=7)
        ax.set(xlabel="Weekend mean / weekday mean", title="C9 | All zones ranked; includes sparse zones")
    save(fig, "C9_weekend_weekday_ranked.png")
    meaningful = ratios.loc[ratios["mean"].ge(1)]
    content += finding("C9", "Weekend-to-weekday ratios", f"Each zone has {int(ratios.weekday_hours.iloc[0]):,} weekday labels and {int(ratios.weekend_hours.iloc[0]):,} weekend labels. Among zones averaging >=1 pickup/hour, {meaningful.Zone.iloc[0]} has the highest ratio ({meaningful.weekend_weekday_ratio.iloc[0]:.2f}x); {meaningful.Zone.iloc[-1]} has the lowest ({meaningful.weekend_weekday_ratio.iloc[-1]:.2f}x). All-zero zones have an undefined ratio, not zero.",
                       ["C9_weekend_weekday_ranked.png", "C9_weekend_weekday_by_zone.csv"], "Use zone x weekend interactions; large ratios in near-empty zones may reflect tiny denominators. All flags, including holidays, are retained for this unadjusted comparison.")
    city_hour = data.groupby("pickup_hour").trip_count.sum().rename("trips").to_frame()
    city_hour["hour"] = city_hour.index.hour
    city_hour["day_of_week"] = city_hour.index.dayofweek
    city_hour["year_month"] = city_hour.index.to_period("M").astype(str)
    daily = city_hour.trips.resample("D").sum().rename("trips").to_frame()
    daily["year_month"] = daily.index.to_period("M").astype(str)
    daily["day_of_week"] = daily.index.dayofweek
    daily["week_start"] = daily.index - pd.to_timedelta(daily.day_of_week, unit="D")
    weekly = daily.groupby("week_start").trips.agg(total="sum", observed_days="size", mean_per_day="mean")
    weekly["complete_week"] = weekly.observed_days.eq(7)
    table(weekly.reset_index(), "D3_weekly_totals.csv")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(range(len(weekly)), weekly.total/1000, color=np.where(weekly.complete_week, "#2878ad", "#e78632"))
    ax.set_xticks(range(len(weekly)), [d.strftime("%d %b") + ("*" if not full else "") for d, full in zip(weekly.index, weekly.complete_week)], rotation=60)
    ax.set(ylabel="Pickups (thousands)", xlabel="Monday week start (* = six days)", title="D3 | Weekly pickup totals; incomplete weeks explicitly marked")
    save(fig, "D3_weekly_totals.png")
    monthly = daily.groupby("year_month").trips.agg(total="sum", observed_days="size", mean_per_day="mean")
    monthly["calendar_days"] = pd.PeriodIndex(monthly.index, freq="M").days_in_month
    monthly["complete_month"] = monthly.observed_days.eq(monthly.calendar_days)
    table(monthly.reset_index(), "D4_monthly_trips_per_day.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly.index, monthly.mean_per_day/1000, color="#27836a")
    for j, days in enumerate(monthly.observed_days): ax.text(j, monthly.mean_per_day.iloc[j]/1000+1, f"{days} observed days", ha="center", fontsize=9)
    ax.set(ylabel="Mean pickups / observed day (thousands)", xlabel="Year-month", title="D4 | Monthly daily demand; September covers only Sep 8--30")
    save(fig, "D4_monthly_trips_per_day.png")
    content += finding("D3/D4", "Calendar-adjusted totals", f"There are {weekly.complete_week.sum()} complete weeks and one final six-day week. September has 23 observed days, not 30. Average daily pickups range from {monthly.mean_per_day.min():,.0f} ({monthly.mean_per_day.idxmin()}) to {monthly.mean_per_day.max():,.0f} ({monthly.mean_per_day.idxmax()}).",
                       ["D3_weekly_totals.png", "D4_monthly_trips_per_day.png", "D3_weekly_totals.csv", "D4_monthly_trips_per_day.csv"], "Compare means per observed day across months; do not interpret partial-week or partial-month totals as a full-period fall.")
    hour_profile = city_hour.groupby("hour").trips.mean()
    weekday_profile = daily.groupby("day_of_week").trips.mean()
    table(hour_profile.rename("mean_trips").reset_index(), "D5_hourly_profile.csv")
    table(weekday_profile.rename("mean_daily_trips").reset_index(), "D6_weekday_profile.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hour_profile, marker="o"); axes[0].set(xlabel="Local hour", ylabel="Mean total pickups / hour", title="D5 | Mean hourly profile")
    axes[1].plot(weekday_profile, marker="o", color="#e78632"); axes[1].set_xticks(range(7), DAY_NAMES)
    axes[1].set(ylabel="Mean total pickups / day", title="D6 | Mean weekday profile")
    save(fig, "D5_D6_calendar_profiles.png")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    axes[0].boxplot([city_hour.loc[city_hour.hour.eq(h), "trips"] for h in range(24)], tick_labels=range(24), showfliers=False)
    axes[0].set(xlabel="Hour", ylabel="Total pickups / hour", title="D5 | Across-date hourly distributions; outlier dots hidden")
    axes[1].boxplot([daily.loc[daily.day_of_week.eq(d), "trips"] for d in range(7)], tick_labels=DAY_NAMES, showfliers=False)
    axes[1].set(xlabel="Weekday", ylabel="Total pickups / day", title="D6 | Daily totals by weekday")
    axes[2].boxplot([daily.loc[daily.year_month.eq(m), "trips"] for m in monthly.index], tick_labels=monthly.index, showfliers=False)
    axes[2].set(xlabel="Year-month", ylabel="Total pickups / day", title="D4 | Daily totals by month; Sep 8--30 only")
    save(fig, "D4_D5_D6_calendar_boxplots.png")
    content += finding("D5/D6", "Hourly and weekday profiles", f"Peak hour={hour_profile.idxmax():02d}:00 ({hour_profile.max():,.0f} pickups/hour); trough={hour_profile.idxmin():02d}:00 ({hour_profile.min():,.0f}). Busiest average day={DAY_NAMES[weekday_profile.idxmax()]} ({weekday_profile.max():,.0f} pickups/day); quietest={DAY_NAMES[weekday_profile.idxmin()]} ({weekday_profile.min():,.0f}).",
                       ["D5_D6_calendar_profiles.png", "D4_D5_D6_calendar_boxplots.png", "D5_hourly_profile.csv", "D6_weekday_profile.csv"], "Hour and weekday have different units here. Calendar variation and zone interactions should be evaluated with time-based validation.")
    band_names = ["Late night 00-04", "Early morning 05-06", "AM 07-09", "Midday 10-15", "PM 16-19", "Evening 20-23"]
    band_data = data.assign(time_band=pd.cut(data.hour, [-1, 4, 6, 9, 15, 19, 23], labels=band_names))
    band_totals = band_data.groupby(["year_month", "time_band"], observed=True).trip_count.sum().unstack().reindex(columns=band_names)
    band_share = band_totals.div(band_totals.sum(axis=1), axis=0) * 100
    assert np.allclose(band_share.sum(axis=1), 100)
    table(band_share.reset_index(), "D9_time_band_share.csv")
    fig, ax = plt.subplots(figsize=(12, 5)); band_share.plot.bar(stacked=True, ax=ax, colormap="viridis")
    ax.set(ylabel="Share of month's pickups (%)", xlabel="Year-month", title="D9 | Pickup shares in six disjoint time bands")
    ax.tick_params(axis="x", rotation=0); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "D9_time_band_shares.png")
    month_profiles = city_hour.groupby(["hour", "year_month"]).trips.mean().unstack()
    normalised = month_profiles.div(month_profiles.sum(axis=0), axis=1) * 100
    table(month_profiles.reset_index(), "D10_monthly_hourly_profiles.csv")
    table(normalised.reset_index(), "D10_monthly_hourly_shares.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    month_profiles.plot(ax=axes[0]); normalised.plot(ax=axes[1])
    axes[0].set(xlabel="Local hour", ylabel="Mean total pickups / hour", title="D10 | Monthly volume profiles")
    axes[1].set(xlabel="Local hour", ylabel="Share of average daily pickups (%)", title="Monthly shape after volume normalisation")
    save(fig, "D10_monthly_hourly_profiles.png")
    content += finding("D9/D10", "Time bands and changing shape", f"The PM 16--19 band carries {band_share['PM 16-19'].min():.1f}%--{band_share['PM 16-19'].max():.1f}% of monthly pickups; late night 00--04 carries {band_share['Late night 00-04'].min():.1f}%--{band_share['Late night 00-04'].max():.1f}%. The two monthly overlays distinguish overall volume changes from changes in time-of-day shares.",
                       ["D9_time_band_shares.png", "D10_monthly_hourly_profiles.png", "D9_time_band_share.csv", "D10_monthly_hourly_profiles.csv", "D10_monthly_hourly_shares.csv"], "Shares measure recorded trips, not rates per band-hour; the bands have unequal durations. September's partial month and changing weekday mix can affect the overlays.")
    write_section(64, "Borough, airport and calendar patterns", content)
    print(f"#64: borough, airport and calendar outputs saved; peak city hour {hour_profile.idxmax():02d}:00.", flush=True)


def issue66(data, zones):
    stats = zone_stats(data)
    content = "\nFleet planning charts describe pickup volume and timing. Counts alone do not estimate\nvehicle availability, trip duration, airport queues or unmet demand. Historical\npercentiles below are empirical pickup counts, not calibrated forecast intervals.\n"
    # Use only complete 168-label weeks for profile consistency.
    week_sizes = data.groupby("week_start").pickup_hour.nunique()
    complete = week_sizes.index[week_sizes.eq(168)]
    full = data.loc[data.week_start.isin(complete)].assign(hour_of_week=lambda x: x.day_of_week*24+x.hour)
    metrics, profiles = [], {}
    for zone_id, group in full.groupby("PULocationID"):
        matrix = group.pivot(index="week_start", columns="hour_of_week", values="trip_count").reindex(columns=range(168))
        assert matrix.shape == (len(complete), 168) and not matrix.isna().any().any()
        profiles[zone_id] = matrix
        values = matrix.to_numpy(dtype=float)
        other = (values.sum(axis=0)[None, :] - values) / (len(values)-1)
        v = values-values.mean(axis=1, keepdims=True)
        o = other-other.mean(axis=1, keepdims=True)
        denom = np.sqrt((v*v).sum(axis=1) * (o*o).sum(axis=1))
        corr = np.divide((v*o).sum(axis=1), denom, out=np.full(len(values), np.nan), where=denom>0)
        for week, correlation, mean in zip(matrix.index, corr, values.mean(axis=1)):
            metrics.append({"PULocationID": zone_id, "week_start": week, "profile_correlation": correlation, "mean_weekly_hourly_demand": mean})
    weekly_metrics = pd.DataFrame(metrics)
    consistency = weekly_metrics.groupby("PULocationID").agg(median_profile_correlation=("profile_correlation", "median"),
        valid_profile_weeks=("profile_correlation", "count"), weekly_mean=("mean_weekly_hourly_demand", "mean"),
        weekly_std=("mean_weekly_hourly_demand", "std"))
    consistency["weekly_level_cv"] = consistency.weekly_std / consistency.weekly_mean.replace(0, np.nan)
    consistency = consistency.join(stats[["Zone", "Borough", "mean"]])
    table(weekly_metrics, "H3_weekly_profile_correlations.csv")
    table(consistency.reset_index(), "H3_zone_consistency.csv")
    chosen = [243, 161, 79, 132, 138, 7]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, zone_id in zip(axes.flat, chosen):
        matrix = profiles[zone_id]
        for _, row in matrix.iterrows(): ax.plot(range(168), row, color="#2878ad", alpha=.13, linewidth=.7)
        ax.plot(range(168), matrix.mean(axis=0), color="#e78632", linewidth=2, label="Mean profile")
        ax.set_xticks(np.arange(0, 168, 24), DAY_NAMES)
        ax.set(ylabel="Pickups / hour", title=label_zone(zone_id, stats.loc[zone_id, "Zone"], 40))
    fig.suptitle(f"H3 | Weekly profiles: {len(complete)} full weeks (thin lines), mean (orange)")
    save(fig, "H3_weekly_profile_overlays.png")
    active = consistency.loc[consistency["mean"].ge(1)]
    content += finding("H3", "Week-to-week consistency", f"Across {len(complete)} complete weeks, median zone correlation with the other weeks' 168-hour template is {active.median_profile_correlation.median():.2f} among zones with mean >=1/hour; median CV of weekly levels={active.weekly_level_cv.median():.2f}. Each comparison excludes its own week from the template. Constant/all-zero profiles have undefined correlation; valid-week counts are exported.",
                       ["H3_weekly_profile_overlays.png", "H3_zone_consistency.csv", "H3_weekly_profile_correlations.csv"], "Separate repeatable timing from changing volume. This retrospective template uses other weeks and is descriptive only; a forecasting baseline must use prior weeks.")
    previous_time = data.groupby("PULocationID").pickup_hour.shift()
    delta = data.groupby("PULocationID").trip_count.diff()
    eligible = data.pickup_hour.sub(previous_time).eq(pd.Timedelta(hours=1)) & ~data.pickup_hour.eq(DST) & ~previous_time.eq(DST)
    changes = data.loc[eligible, ["PULocationID", "pickup_hour", "Borough", "hour"]].assign(change=delta.loc[eligible])
    change_stats = changes.groupby("Borough").change.agg(rows="size", mean="mean", median="median",
        p01=lambda x: x.quantile(.01), p99=lambda x: x.quantile(.99), mean_abs=lambda x: x.abs().mean(), max_abs=lambda x: x.abs().max())
    table(change_stats.reset_index(), "H4_hourly_change_summary.csv")
    change_hour = changes.groupby(["hour", "Borough"]).change.mean().unstack()
    table(change_hour.reset_index(), "H4_change_by_hour.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for borough, group in changes.groupby("Borough"):
        axes[0].hist(group.change, bins=np.arange(-650, 651, 10), color=COLORS[borough], alpha=.6, label=borough)
    axes[0].set(yscale="log", xlabel="Current minus previous hour pickups (per zone)", ylabel="Transitions (log scale)", title="H4 | Adjacent-hour changes; DST transitions excluded")
    axes[0].legend(); change_hour.plot(ax=axes[1], color=COLORS)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(xlabel="Local hour at end of transition", ylabel="Mean signed change / zone", title="Typical increases and decreases by hour")
    save(fig, "H4_hourly_changes.png")
    content += finding("H4", "Hour-over-hour changes", "; ".join(f"{b}: mean absolute change {r.mean_abs:.1f} pickups/zone, 1st--99th percentile {r.p01:.0f} to {r.p99:.0f}" for b, r in change_stats.iterrows()) + f". {len(changes):,} valid within-zone transitions; no cross-zone boundary differences, first-row or DST-adjacent differences.",
                       ["H4_hourly_changes.png", "H4_hourly_change_summary.csv", "H4_change_by_hour.csv"], "Repositioning needs to anticipate rising slots, with larger count changes in busy Manhattan zones. Changes remain signed; zero previous counts are valid.")
    stats["peak_average_ratio"] = stats.maximum / stats["mean"].replace(0, np.nan)
    table(stats.reset_index(), "H5_I3_zone_peak_demand.csv")
    selected = stats.loc[stats["mean"].ge(1)].nlargest(20, "peak_average_ratio")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(selected)), selected.peak_average_ratio, color=[COLORS[b] for b in selected.Borough])
    ax.set_yticks(range(len(selected)), [label_zone(i, n) for i, n in selected.Zone.items()]); ax.invert_yaxis()
    ax.set(xlabel="Maximum recorded hourly pickups / mean", title="H5 | Top 20 peak-to-average ratios among zones averaging >=1/hour")
    save(fig, "H5_peak_average_ratios.png")
    content += finding("H5", "Peak-to-average ratios", f"Among zones with mean >=1/hour, median maximum/mean={stats.loc[stats['mean'].ge(1), 'peak_average_ratio'].median():.2f}; {selected.Zone.iloc[0]} leads at {selected.peak_average_ratio.iloc[0]:.2f}x. Ratios for all 138 zones are exported; two all-zero zones are undefined.",
                       ["H5_peak_average_ratios.png", "H5_I3_zone_peak_demand.csv"], "Maximum/mean is sensitive to one event and tiny means. Use P95/P99 and conditional calendar profiles alongside maxima.")
    means = data.groupby(["PULocationID", "hour"]).trip_count.mean().unstack("hour").sort_index()
    # Deterministic tie-break: lower zone ID first.
    ranks = means.rank(axis=0, ascending=False, method="first")
    ranked = []
    for h in range(24):
        top = means[h].sort_values(ascending=False, kind="stable").head(10)
        for rank, (zone_id, value) in enumerate(top.items(), 1):
            ranked.append({"hour": h, "rank": rank, "PULocationID": zone_id, "Zone": stats.loc[zone_id, "Zone"], "mean_pickups_per_hour": value})
    table(pd.DataFrame(ranked), "I1_top10_zones_each_hour.csv")
    union = ranks.index[ranks.le(10).any(axis=1)]
    fig, ax = plt.subplots(figsize=(15, 7))
    palette = [*plt.get_cmap("tab20").colors, *plt.get_cmap("tab20b").colors]
    for j, zone_id in enumerate(union):
        ax.plot(range(24), ranks.loc[zone_id].where(ranks.loc[zone_id].le(10)), marker="o", markersize=3,
                color=palette[j], label=label_zone(zone_id, stats.loc[zone_id, "Zone"], 26), linewidth=1.5)
    ax.set(ylim=(10.5, .5), yticks=range(1, 11), xticks=range(24), xlabel="Local hour", ylabel="Rank (1 = busiest)", title="I1 | Top-ten ranks through the day; lines hidden when rank >10")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    save(fig, "I1_top_zone_bump_chart.png")
    coverage = pd.DataFrame(index=range(24))
    for n in [5, 10, 20]:
        coverage[f"top_{n}_share_pct"] = [100 * means[h].nlargest(n).sum() / means[h].sum() for h in range(24)]
    coverage.index.name = "hour"
    table(coverage.reset_index(), "I2_topN_hourly_coverage.csv")
    fig, ax = plt.subplots(figsize=(11, 4)); coverage.plot(ax=ax, marker="o")
    ax.set(xlabel="Local hour", ylabel="Share of pickups (%)", title="I2 | Coverage by the busiest N zones at each hour")
    save(fig, "I2_topN_coverage.png")
    content += finding("I1/I2", "Changing hotspots and top-N coverage", f"{len(union)} distinct zones enter the top ten at some hour. Top-five coverage ranges {coverage.top_5_share_pct.min():.1f}%--{coverage.top_5_share_pct.max():.1f}%; top-ten {coverage.top_10_share_pct.min():.1f}%--{coverage.top_10_share_pct.max():.1f}%; top-twenty {coverage.top_20_share_pct.min():.1f}%--{coverage.top_20_share_pct.max():.1f}%. Rankings use average zone pickups at each clock hour and ties use the lower zone ID.",
                       ["I1_top_zone_bump_chart.png", "I1_top10_zones_each_hour.csv", "I2_topN_coverage.png", "I2_topN_hourly_coverage.csv"], "Top-N sets change by hour and are selected retrospectively over the full period; a fixed deployment or forward test needs historical-only selection.")
    busy = stats.nlargest(15, "mean")
    fig, ax = plt.subplots(figsize=(13, 6))
    busy[["mean", "p95", "p99"]].set_axis([label_zone(i, n, 23) for i, n in busy.Zone.items()]).plot.bar(ax=ax, color=["#2878ad", "#e78632", "#9056a0"])
    ax.set(ylabel="Pickups / zone-hour", xlabel="Pickup zone", title="I3 | Mean vs empirical P95/P99 in the 15 busiest zones")
    ax.tick_params(axis="x", rotation=60); save(fig, "I3_mean_p95_p99.png")
    content += finding("I3", "Demand percentiles", f"{busy.Zone.iloc[0]} averages {busy['mean'].iloc[0]:.1f} pickups/hour, with P95={busy.p95.iloc[0]:.0f} and P99={busy.p99.iloc[0]:.0f}. All-zero and sparse zones remain in the exported percentile table; linear empirical quantiles can be fractional.",
                       ["I3_mean_p95_p99.png", "H5_I3_zone_peak_demand.csv"], "Sizing against the average misses recurring busy hours. These unconditional pickup quantiles are not taxi counts or predictive uncertainty bounds.")
    compare = pd.DataFrame({"Manhattan (all zones)": data.loc[data.Borough.eq("Manhattan")].groupby(["pickup_hour", "hour"]).trip_count.sum().groupby("hour").mean()})
    for zone_id in [132, 138]:
        compare[stats.loc[zone_id, "Zone"]] = means.loc[zone_id]
    relative = compare.div(compare.mean(axis=0), axis=1)
    table(compare.reset_index(), "I4_airport_manhattan_hourly.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    compare.plot(ax=axes[0], logy=True); relative.plot(ax=axes[1])
    axes[0].set(xlabel="Local hour", ylabel="Mean pickups / hour (log scale)", title="I4 | Airport vs Manhattan count levels")
    axes[1].set(xlabel="Local hour", ylabel="Hourly mean / group's overall hourly mean", title="Relative timing (each group's mean = 1)")
    save(fig, "I4_airport_manhattan_timing.png")
    content += finding("I4", "Airport demand versus Manhattan", "; ".join(f"{n} peaks {int(compare[n].idxmax()):02d}:00" for n in compare) + ". Manhattan represents 69 zones; each airport represents one zone. Relative profiles divide by each group's own mean.",
                       ["I4_airport_manhattan_timing.png", "I4_airport_manhattan_hourly.csv"], "Compare timing on a normalised scale; the borough's much larger level would otherwise hide the airport shapes.")
    # Operational night starts at 20:00. Fri/Sat evening and Sat/Sun 00--04
    # belong to weekend nights; early Friday belongs to Thursday night.
    nights = data.loc[data.hour.ge(20) | data.hour.le(4)].copy()
    nights["night_date"] = nights.date.where(nights.hour.ge(20), nights.date-pd.Timedelta(days=1))
    nights["weekend_night"] = nights.night_date.dt.dayofweek.isin([4, 5])
    periods = {
        "late_night_00_04": data.loc[data.hour.le(4)],
        "weekend_night_Fri_Sat_20_04": nights.loc[nights.weekend_night],
    }
    night_tables = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, (name, group) in zip(axes, periods.items()):
        result = group.groupby("PULocationID").trip_count.agg(total="sum", observed_hours="size", mean_per_hour="mean")
        result = result.join(stats[["Zone", "Borough", "latitude", "longitude"]]).sort_values("mean_per_hour", ascending=False)
        result["period"] = name; night_tables.append(result.reset_index())
        top = result.head(12)
        ax.barh(range(len(top)), top.mean_per_hour, color=[COLORS[b] for b in top.Borough])
        ax.set_yticks(range(len(top)), [label_zone(i, n, 26) for i, n in top.Zone.items()]); ax.invert_yaxis()
        ax.set(xlabel="Mean pickups / observed hour", title=name.replace("_", " "))
    fig.suptitle("I5 | Late-night and Friday/Saturday-night pickup hotspots")
    save(fig, "I5_night_hotspots.png")
    night_results = pd.concat(night_tables, ignore_index=True)
    table(night_results, "I5_night_hotspots.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, name in zip(axes, periods):
        group = night_results.loc[night_results.period.eq(name)]
        norm = matplotlib.colors.LogNorm(vmin=.01, vmax=night_results.mean_per_hour.max())
        im = ax.scatter(group.longitude, group.latitude, c=group.mean_per_hour.clip(lower=.01), cmap="magma", norm=norm,
                        s=12+2*group.mean_per_hour, edgecolors="white", linewidth=.3)
        labelled = group.head(8).sort_values("latitude")
        city_labels = labelled.loc[labelled.Borough.eq("Manhattan")]
        for (_, row), target_y in zip(city_labels.iterrows(), np.linspace(40.715, 40.79, len(city_labels))):
            ax.annotate(str(row.PULocationID), (row.longitude, row.latitude), xytext=(-74.07, target_y),
                        ha="right", va="center", fontsize=8, arrowprops={"arrowstyle": "-", "color": "#555555", "lw": .5})
        for row in labelled.loc[labelled.Borough.eq("Queens")].itertuples():
            ax.annotate(str(row.PULocationID), (row.longitude, row.latitude), xytext=(row.longitude+.015, row.latitude+.01),
                        fontsize=8, arrowprops={"arrowstyle": "-", "color": "#555555", "lw": .5})
        ax.set_xlim(-74.09, -73.70)
        ax.set(xlabel="Longitude", ylabel="Latitude", title=name.replace("_", " "))
        ax.set_aspect(1/np.cos(np.deg2rad(40.75)))
        fig.colorbar(im, ax=ax, label="Mean pickups / hour; zeros shown at 0.01")
    fig.suptitle("I5 | Zone-centroid map; labels are IDs of the top eight (no boundary map)")
    save(fig, "I5_night_hotspot_map.png")
    weekend_top = night_results.loc[night_results.period.eq("weekend_night_Fri_Sat_20_04")].iloc[0]
    late_top = night_results.loc[night_results.period.eq("late_night_00_04")].iloc[0]
    content += finding("I5", "Night hotspots", f"00--04 leader: {late_top.Zone} ({late_top.mean_per_hour:.1f} pickups/hour). Friday/Saturday 20:00--04:59 leader: {weekend_top.Zone} ({weekend_top.mean_per_hour:.1f}/hour). Weekend nights are assigned to their evening date: early Sat/Sun is included; early Fri is not. Means use actual available hour labels (boundary nights may be partial).",
                       ["I5_night_hotspots.png", "I5_night_hotspot_map.png", "I5_night_hotspots.csv"], "Nightlife positioning differs from commuter coverage. Centroids are location proxies and do not identify exact pickup points or driving routes.")
    outer = data.loc[data.service_zone.eq("Boro Zone")]
    boro = outer.groupby(["hour", "Borough"]).trip_count.mean().unstack()
    table(boro.reset_index(), "I6_boro_zone_hourly.csv")
    fig, ax = plt.subplots(figsize=(11, 4)); boro.plot(ax=ax, color=COLORS)
    ax.set(xlabel="Local hour", ylabel="Mean pickups / Boro zone-hour (zeros retained)", title="I6 | Outer service-area demand by borough")
    save(fig, "I6_boro_zone_hourly.png")
    content += finding("I6", "Boro Zone demand", f"The Boro Zone service area contains {outer.PULocationID.nunique()} zones. " + "; ".join(f"{b} peaks at {int(boro[b].idxmax()):02d}:00 ({boro[b].max():.2f} pickups/zone-hour)" for b in boro) + ".",
                       ["I6_boro_zone_hourly.png", "I6_boro_zone_hourly.csv"], "Normalise by zone count and keep sparse areas visible. Boro Zone counts include both yellow and green taxis; supply requirements cannot be inferred from pickups alone.")
    write_section(66, "Fleet planning and demand stability", content)
    print(f"#66: {len(complete)} full-week profiles; {len(union)} hourly top-ten zones; fleet tables and charts saved.", flush=True)


SPECIAL_DAYS = {"2025-10-31": "Halloween", "2025-11-02": "NYC Marathon",
                "2025-11-28": "Black Friday", "2025-12-24": "Christmas Eve"}


def control_calendar(data):
    dates = pd.DatetimeIndex(sorted(data.date.unique()))
    excluded = set(pd.to_datetime(list(HOLIDAYS) + list(SPECIAL_DAYS) + ["2025-11-01"]))
    excluded.update(pd.date_range("2025-11-24", "2025-11-30"))
    excluded.update(pd.date_range("2025-12-20", "2026-01-04"))
    # Per-zone daily totals followed by spatial median avoids counting the same
    # storm once per taxi zone. Exclude days with median >=5 cm from normal days.
    daily_snow = data.groupby(["date", "PULocationID"]).snowfall.sum().groupby("date").median()
    snow_days = set(daily_snow.index[daily_snow.ge(5)])
    excluded.update(snow_days)
    return dates, excluded, snow_days


def select_controls(target, dates, excluded, window=28, max_days=6):
    candidate = [d for d in dates if d.dayofweek == target.dayofweek and d != target
                 and abs((d-target).days) <= window and d not in excluded]
    chosen = sorted(candidate, key=lambda d: (abs((d-target).days), d))[:max_days]
    if len(chosen) < 3:
        raise ValueError(f"Only {len(chosen)} normal same-weekday controls for {target.date()}")
    return pd.DatetimeIndex(sorted(chosen))


def issue65(data, zones):
    dates, excluded, snow_days = control_calendar(data)
    groups = {"All": data, **dict(tuple(data.groupby("Borough")))}
    series = {name: group.groupby("pickup_hour").trip_count.sum() for name, group in groups.items()}
    profiles = {name: s.to_numpy().reshape(len(dates), 24) for name, s in series.items()}
    date_positions = {date: j for j, date in enumerate(dates)}

    def event_rows(event_dates):
        rows = []
        for date_text, name in event_dates.items():
            date = pd.Timestamp(date_text)
            controls = select_controls(date, dates, excluded)
            wider = select_controls(date, dates, excluded, window=42)
            for scope in groups:
                values = profiles[scope]
                actual = float(values[date_positions[date]].sum())
                control_values = np.array([values[date_positions[d]].sum() for d in controls])
                mean = control_values.mean()
                wider_mean = np.mean([values[date_positions[d]].sum() for d in wider])
                rows.append({"date": date.strftime("%Y-%m-%d"), "event": name, "scope": scope,
                    "weekday": DAY_NAMES[date.dayofweek], "actual_trips": actual, "normal_mean_trips": mean,
                    "pct_change": 100*(actual/mean-1), "control_days": len(controls),
                    "control_dates": ";".join(d.strftime("%Y-%m-%d") for d in controls),
                    "normal_min_trips": control_values.min(), "normal_max_trips": control_values.max(),
                    "normal_std_trips": control_values.std(ddof=1), "window_days": 28,
                    "pct_change_window42": 100*(actual/wider_mean-1),
                    "control_dates_window42": ";".join(d.strftime("%Y-%m-%d") for d in wider)})
        return pd.DataFrame(rows)

    present_holidays = {d: n for d, n in HOLIDAYS.items() if pd.Timestamp(d) in date_positions}
    assert len(present_holidays) == 7
    effects = event_rows(present_holidays)
    table(effects, "holiday_effects.csv")
    exclusions = pd.DataFrame({"date": sorted(d for d in excluded if d in date_positions)})
    exclusions["reason"] = exclusions.date.map(lambda d: "heavy snow (median zone-day >=5cm)" if d in snow_days else "holiday, special day, spillover or holiday period")
    table(exclusions, "E1_control_exclusions.csv")
    all_effects = effects.loc[effects.scope.eq("All")].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(range(len(all_effects)), all_effects["pct_change"], color=np.where(all_effects["pct_change"].ge(0), "#27836a", "#c44e52"))
    ax.set_yticks(range(len(all_effects)), all_effects.event); ax.invert_yaxis(); ax.axvline(0, color="black")
    for j, row in enumerate(all_effects.itertuples()):
        ax.annotate(f"{row.pct_change:+.1f}% (n={row.control_days})", (row.pct_change, j), xytext=(4 if row.pct_change>=0 else -4, 0),
                    textcoords="offset points", va="center", ha="left" if row.pct_change>=0 else "right", fontsize=9)
    ax.margins(x=.25)
    ax.set(xlabel="Change from nearby normal same-weekday mean (%)", title="E1 | Seven recurring federal holidays; local controls within +/-28 days")
    save(fig, "E1_holiday_pct_change.png")
    hourly_rows = []
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    for ax, (date_text, name) in zip(axes.flat, present_holidays.items()):
        date = pd.Timestamp(date_text); controls = select_controls(date, dates, excluded)
        normal = profiles["All"][[date_positions[d] for d in controls]]
        actual = profiles["All"][date_positions[date]]
        ax.plot(range(24), actual, label="Holiday", color="#c44e52")
        ax.plot(range(24), normal.mean(axis=0), label="Normal-day mean", color="#2878ad")
        ax.fill_between(range(24), np.quantile(normal, .25, axis=0), np.quantile(normal, .75, axis=0), color="#2878ad", alpha=.15, label="Control-day IQR")
        ax.set(xlabel="Local hour", ylabel="Total pickups / hour", title=f"{name} | {date_text} (n={len(controls)})")
        for h in range(24): hourly_rows.append({"date": date_text, "event": name, "hour": h, "actual_trips": actual[h], "normal_mean_trips": normal[:, h].mean(), "normal_p25": np.quantile(normal[:, h], .25), "normal_p75": np.quantile(normal[:, h], .75), "control_days": len(controls)})
    axes.flat[-1].axis("off"); axes.flat[0].legend(fontsize=8)
    fig.suptitle("E1 | Holiday shapes versus matched normal weekdays; IQR is descriptive")
    save(fig, "E1_holiday_hourly_profiles.png")
    table(pd.DataFrame(hourly_rows), "E1_holiday_hourly_profiles.csv")
    content = """
Normal-day baseline: up to six nearest dates of the **same weekday within +/-28 days**,
with ties favouring the earlier date; at least three are required. Controls exclude
the seven recurring federal holidays, Halloween and its next morning, the marathon/DST
day, Thanksgiving week (Nov 24--30), the Christmas/New Year period (Dec 20--Jan 4),
and heavy-snow days (median across zone daily snowfall totals >=5 cm). The exact control
dates, counts, range and standard deviation are exported. A +/-42-day sensitivity uses
the same exclusion rules and cap of six. IQR shading shows control-day variability,
not a confidence interval. These local comparisons remain affected by weather, trend,
events and supply; they are descriptive changes rather than causal holiday effects.
The calendar is the upstream recurring federal calendar, with special days analysed
separately. [Federal holiday dates](https://www.opm.gov/policy-data-oversight/pay-leave/federal-holidays/).
The marathon date is grounded in [NYRR's 2025 results](https://results.nyrr.org/event/M2025/result/9).
"""
    display_effects = all_effects[["date", "event", "actual_trips", "normal_mean_trips", "pct_change", "control_days"]].copy()
    for col in ["actual_trips", "normal_mean_trips"]: display_effects[col] = display_effects[col].round().astype(int)
    display_effects["pct_change"] = display_effects["pct_change"].map(lambda x: f"{x:+.1f}%")
    content += "\n" + md_table(display_effects) + "\n"
    worst = all_effects.loc[all_effects["pct_change"].idxmin()]
    content += finding("E1", "Seven federal holidays", f"Largest decline: {worst.event}, {worst['pct_change']:+.1f}% relative to its matched normal weekday. All seven holidays have citywide and borough results (21 rows) in holiday_effects.csv. Labor Day cannot be estimated from features.csv because it is in the removed warm-up week. Control counts range {all_effects.control_days.min()}--{all_effects.control_days.max()}.",
                       ["E1_holiday_pct_change.png", "E1_holiday_hourly_profiles.png", "holiday_effects.csv", "E1_holiday_hourly_profiles.csv", "E1_control_exclusions.csv"], "Use holiday identity and interactions, not only one pooled is_holiday coefficient. There is one observation of each holiday, so validate any proposed effect out of time.")
    mlk = all_effects.loc[all_effects.date.eq("2026-01-19")].iloc[0]
    content += (f"\nBaseline sensitivity matters: MLK Day changes from {mlk['pct_change']:+.1f}% "
                f"with +/-28-day controls to {mlk['pct_change_window42']:+.1f}% with +/-42-day controls. "
                "Where the same nearest six dates are selected, the sensitivity result is identical by construction. "
                "Late-period holidays have fewer eligible controls because the dataset ends on Feb 28.\n")
    periods = [("Thanksgiving week", "2025-11-24", "2025-11-30"), ("Christmas / New Year", "2025-12-20", "2026-01-04")]
    period_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    period_findings = []
    for ax, (name, start, end) in zip(axes, periods):
        day_range = pd.date_range(start, end)
        actuals, normals = [], []
        for date in day_range:
            controls = select_controls(date, dates, excluded)
            actual = float(profiles["All"][date_positions[date]].sum())
            normal = float(np.mean([profiles["All"][date_positions[d]].sum() for d in controls]))
            actuals.append(actual); normals.append(normal)
            period_rows.append({"period": name, "date": date.strftime("%Y-%m-%d"), "actual_trips": actual,
                "normal_mean_trips": normal, "pct_change": 100*(actual/normal-1), "control_days": len(controls),
                "control_dates": ";".join(d.strftime("%Y-%m-%d") for d in controls)})
        ax.plot(day_range, actuals, marker="o", label="Actual", color="#c44e52")
        ax.plot(day_range, normals, marker="o", label="Matched normal weekday", color="#2878ad")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1 if len(day_range)<10 else 2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.set(ylabel="Total pickups / day", title=f"E2 | {name}"); ax.legend()
        period_findings.append(f"{name}: {100*(sum(actuals)/sum(normals)-1):+.1f}% across the period")
    save(fig, "E2_holiday_periods.png")
    table(pd.DataFrame(period_rows), "E2_holiday_periods.csv")
    content += finding("E2", "Holiday periods", "; ".join(period_findings) + ". Each day's reference matches its own weekday, rather than comparing the entire period with a single day.",
                       ["E2_holiday_periods.png", "E2_holiday_periods.csv"], "Holiday-period flags and lead/lag days may capture effects missed by exact holiday dates; local seasonal changes still confound the comparisons.")
    zone_hour = data.pivot(index="pickup_hour", columns="PULocationID", values="trip_count")
    event_night = pd.Timestamp("2025-12-31")
    controls = select_controls(event_night, dates, excluded)
    control_nights = [pd.date_range(d+pd.Timedelta(hours=20), periods=9, freq="h") for d in controls]
    assert all(all(h in zone_hour.index for h in hours) for hours in control_nights)
    night_hours = pd.date_range(event_night+pd.Timedelta(hours=20), periods=9, freq="h")
    actual = zone_hour.loc[night_hours].to_numpy()
    baseline = np.mean([zone_hour.loc[hours].to_numpy() for hours in control_nights], axis=0)
    night_by_zone = pd.DataFrame({"PULocationID": zone_hour.columns,
        "actual_night_trips": actual.sum(axis=0), "normal_night_trips": baseline.sum(axis=0)})
    night_by_zone["pct_change"] = 100*(night_by_zone.actual_night_trips/night_by_zone.normal_night_trips.replace(0, np.nan)-1)
    night_by_zone = night_by_zone.merge(zones[["LocationID", "Zone", "Borough"]], left_on="PULocationID", right_on="LocationID").drop(columns="LocationID")
    night_by_zone["control_nights"] = len(controls)
    night_by_zone["control_evening_dates"] = ";".join(d.strftime("%Y-%m-%d") for d in controls)
    table(night_by_zone.sort_values("actual_night_trips", ascending=False), "E3_new_year_night_by_zone.csv")
    night_hourly = []
    for t, timestamp in enumerate(night_hours):
        for j, zone_id in enumerate(zone_hour.columns):
            night_hourly.append({"pickup_hour": timestamp, "PULocationID": zone_id, "actual_trips": actual[t, j],
                                 "normal_mean_trips": baseline[t, j], "control_nights": len(controls)})
    table(pd.DataFrame(night_hourly), "E3_new_year_night_hourly.csv")
    chosen = night_by_zone.nlargest(8, "actual_night_trips")
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for ax, row in zip(axes.flat, chosen.itertuples()):
        j = zone_hour.columns.get_loc(row.PULocationID)
        ax.plot(range(9), actual[:, j], marker="o", color="#c44e52", label="NYE night")
        ax.plot(range(9), baseline[:, j], marker="o", color="#2878ad", label="Normal Wed night")
        ax.set_xticks(range(9), ["20", "21", "22", "23", "00", "01", "02", "03", "04"])
        ax.axvline(3.5, color="black", linestyle="--", alpha=.4)
        ax.set(xlabel="31 Dec evening -> 1 Jan morning (local hour)", ylabel="Pickups / zone-hour", title=label_zone(row.PULocationID, row.Zone, 40))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("E3 | New Year's Eve night: top eight zones by episode pickups")
    save(fig, "E3_new_year_night_profiles.png")
    content += finding("E3", "New Year's Eve night", f"The episode is Dec 31 20:00--Jan 1 04:59 (nine hour labels). Total pickups={actual.sum():,} versus {baseline.sum():,.0f} on {len(controls)} normal Wednesday nights ({100*(actual.sum()/baseline.sum()-1):+.1f}%). All 138 zones and 1,242 zone-hour comparisons are exported; the chart shows the eight largest episode totals.",
                       ["E3_new_year_night_profiles.png", "E3_new_year_night_by_zone.csv", "E3_new_year_night_hourly.csv"], "A calendar-day holiday flag splits this cross-midnight event. Use a night-event flag attached to the full episode; top zones here are retrospective selections.")
    special = event_rows(SPECIAL_DAYS)
    table(special, "E4_special_day_effects.csv")
    special_all = special.loc[special.scope.eq("All")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (date_text, name) in zip(axes.flat, SPECIAL_DAYS.items()):
        date = pd.Timestamp(date_text); controls = select_controls(date, dates, excluded)
        actual_profile = profiles["All"][date_positions[date]]
        baseline_profile = profiles["All"][[date_positions[d] for d in controls]].mean(axis=0)
        ax.plot(range(24), actual_profile, label="Special day", color="#c44e52")
        ax.plot(range(24), baseline_profile, label="Normal same weekday", color="#2878ad")
        ax.set(xlabel="Local hour", ylabel="Total pickups / hour", title=f"E4 | {name} ({date_text})")
        if name == "NYC Marathon": ax.axvspan(.7, 1.3, color="red", alpha=.1, label="Combined DST hour")
        ax.legend(fontsize=8)
    save(fig, "E4_special_day_profiles.png")
    marathon = pd.Timestamp("2025-11-02")
    marathon_controls = select_controls(marathon, dates, excluded)
    keep = np.arange(24) != 1
    without_dst = profiles["All"][date_positions[marathon], keep].sum()
    normal_without_dst = np.mean([profiles["All"][date_positions[d], keep].sum() for d in marathon_controls])
    sensitivity = pd.DataFrame({"comparison": ["All recorded labels", "Omit 01:00 in event and every control"],
        "actual_trips": [profiles["All"][date_positions[marathon]].sum(), without_dst],
        "normal_mean_trips": [np.mean([profiles["All"][date_positions[d]].sum() for d in marathon_controls]), normal_without_dst]})
    sensitivity["pct_change"] = 100*(sensitivity.actual_trips/sensitivity.normal_mean_trips-1)
    table(sensitivity, "E4_marathon_DST_sensitivity.csv")
    content += finding("E4", "Special days", "; ".join(f"{r.event}: {r.pct_change:+.1f}%" for r in special_all.itertuples()) + f". The marathon coincides with DST; omitting 01:00 from both event and controls gives {sensitivity['pct_change'].iloc[1]:+.1f}% versus {sensitivity['pct_change'].iloc[0]:+.1f}% with all labels. Halloween's calendar-day profile excludes the following night's spillover; the model recommendation covers both dates.",
                       ["E4_special_day_profiles.png", "E4_special_day_effects.csv", "E4_marathon_DST_sensitivity.csv"], "Event flags may help, but marathon and DST effects cannot be separated causally from this one day. Black Friday and Christmas Eve are analysed separately from recurring federal flags.")
    city = data.groupby("pickup_hour").agg(trips=("trip_count", "sum"), is_holiday=("is_holiday", "first"), is_weekend=("is_weekend", "first"))
    city["day_class"] = np.where(city.is_holiday, "Holiday", np.where(city.is_weekend, "Weekend (nonholiday)", "Weekday (nonholiday)"))
    city["hour"] = city.index.hour
    shape = city.groupby(["hour", "day_class"]).trips.mean().unstack()
    shape_share = shape.div(shape.sum(axis=0), axis=1) * 100
    shape_days = city.groupby("day_class").size()/24
    table(shape.reset_index(), "D8_day_class_hourly.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    shape.plot(ax=axes[0]); shape_share.plot(ax=axes[1])
    axes[0].set(xlabel="Local hour", ylabel="Mean total pickups / hour", title="D8 | Disjoint weekday / weekend / holiday classes")
    axes[1].set(xlabel="Local hour", ylabel="Share of average daily pickups (%)", title="Shape normalised for different daily volumes")
    save(fig, "D8_day_class_profiles.png")
    content += finding("D8", "Weekday, weekend and holiday shapes", "; ".join(f"{n}: {int(shape_days[n])} days, peak {int(shape[n].idxmax()):02d}:00" for n in shape) + ". Holiday takes priority so classes are disjoint. Weekday/weekend profiles retain special non-federal days and are pooled over months.",
                       ["D8_day_class_profiles.png", "D8_day_class_hourly.csv"], "Volume and shape both differ. Seven holidays are heterogeneous; this pooled comparison should not replace holiday-specific profiles.")
    # G9 is deliberately produced after the data, calendar and fleet analyses.
    required = ["A1_A8_data_checks.csv", "D10_monthly_hourly_profiles.csv", "H3_zone_consistency.csv", "I5_night_hotspots.csv"]
    if not all((TABLES / f).exists() for f in required):
        raise ValueError("Run #63, #64 and #66 before #65/G9")
    recommendations = [
        ("zone_x_hour_x_weekday", "G9; D5/D6; C5; I4", "Interactions of zone/service area with hour and weekday", "Known calendar and location", "Learn coefficients/encodings on training history only"),
        ("zone_x_weekend_night", "G9; C9; I5", "Friday/Saturday evening plus following early morning, by zone", "Calendar known ahead", "Assign 00--04 to the preceding evening date"),
        ("holiday_name_x_hour", "G9; E1; D8", "Separate recurring holiday identity and hourly shape", "Published calendar", "One observed example per holiday; test out of time"),
        ("holiday_period_and_distance", "G9; E2", "Thanksgiving week, Dec20--Jan4, days before/after holiday", "Published calendar", "Fix event definitions before evaluation; avoid target-derived cutoffs"),
        ("special_night_event", "G9; E3/E4", "NYE night, Halloween night/spillover, Black Friday, Christmas Eve, marathon day", "Published event calendar", "Match cross-midnight windows; no post-event counts as inputs"),
        ("DST_and_affected_lags", "G9; A8", "Combined DST hour and lag_24h/lag_168h contamination indicators", "Timezone rules known", "Naive labels cannot recover the two physical hours; holiday flags cannot flag DST"),
        ("seasonal_time_features", "G9; D3/D4; D10; H3", "Cyclic hour/weekday and month or smooth seasonal trend", "Calendar known ahead", "Only six months: annual seasonality not established"),
        ("zone_demand_level_and_sparsity", "G9; B2/B4; H5", "Historical zero share and demand level; optional hierarchical zone grouping", "Past demand only", "Compute on training history or lagged rolling history, never full-data summaries"),
        ("zone_weather_cell_metadata", "G9; A8", "Shared horizontal weather-cell ID for grouped spatial evaluation", "Static source metadata", "Elevation may vary within cell; this is an evaluation/grouping candidate, not proven predictive gain"),
    ]
    features = pd.DataFrame(recommendations, columns=["candidate_feature", "evidence_ids", "definition", "forecast_availability", "validation_note"])
    table(features, "G9_suggested_features.csv")
    content += finding("G9", "Modelling features suggested by all four analyses", "Nine candidate feature families are exported after #63, #64 and #66. They target geographic/calendar interactions, cross-midnight events, holiday periods, DST and seasonal level shifts. These are proposals; no feature engineering, model fitting or association-rule mining is included in this Review 1 work.",
                       ["G9_suggested_features.csv"], "Retain existing lags and weather features for later ablation. Select candidate features using chronological validation; zone-level summaries must be learned on training data only.")
    content += "\n" + md_table(features[["candidate_feature", "evidence_ids", "validation_note"]]) + "\n"
    write_section(65, "Holidays, special days and modelling suggestions", content)
    print(f"#65: 7 holidays, 4 special days and cross-midnight NYE analysed; G9 saved last.", flush=True)


def write_manifest():
    inputs = ["data/processed/features.csv", "data/processed/features_manhattan.csv", "data/processed/features_queens.csv",
              "data/processed/cleaning_report.md", "data/raw/taxi_zones_manhattan_queens.csv",
              "data/raw/weather_hourly_by_zone.csv", "data/raw/taxi_collection_summary.csv"]
    rows = []
    for name in inputs:
        path = ROOT / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                digest.update(chunk)
        rows.append({"input_path": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    table(pd.DataFrame(rows), "input_manifest.csv")
    import sys
    table(pd.DataFrame({"package": ["python", "numpy", "pandas", "matplotlib"],
                        "version": [sys.version.split()[0], np.__version__, pd.__version__, matplotlib.__version__]}), "execution_environment.csv")
    artifacts = []
    for path in sorted(FIGURES.glob("*.png")) + sorted(TABLES.glob("*.csv")):
        if path.name == "artifact_catalog.csv":
            continue
        artifacts.append({"path": path.relative_to(OUT).as_posix(), "type": path.suffix[1:],
                          "ids_in_filename": ";".join(re.findall(r"[A-I]\d+", path.stem))})
    table(pd.DataFrame(artifacts), "artifact_catalog.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues", type=int, nargs="+", default=[63, 64, 66, 65], choices=[63, 64, 66, 65])
    args = parser.parse_args()
    data, zones = load_data()
    for issue in args.issues:
        {63: issue63, 64: issue64, 66: issue66, 65: issue65}[issue](data, zones)
    write_manifest()


if __name__ == "__main__":
    main()
