"""
Build the small data files used by the Streamlit dashboard.

The project already saved every result we need (EDA tables, model predictions,
time-series series, association rules). This script only joins and reshapes a
few of them so the dashboard loads quickly. Nothing is re-trained here.

Run once from the repository root (or from this folder):

    python dashboard/build_data.py

Outputs are written to dashboard/data/.
"""

from pathlib import Path

import pandas as pd

DASHBOARD_DIR = Path(__file__).resolve().parent
REPO = DASHBOARD_DIR.parent
OUT = DASHBOARD_DIR / "data"
OUT.mkdir(exist_ok=True)


def read(relative_path, **kwargs):
    """Read a CSV file from the repository."""
    return pd.read_csv(REPO / relative_path, **kwargs)


# ---------------------------------------------------------------------------
# 1. One row per taxi zone: location, EDA tier, clusters, ARM thresholds,
#    time-series method and model error.
# ---------------------------------------------------------------------------
def build_zones():
    zones = read("data/raw/taxi_zones_manhattan_queens.csv")
    zones = zones.rename(columns={"LocationID": "PULocationID"})

    tiers = read("EDA_doc/tables/zone_tiers.csv")[
        ["PULocationID", "total_trips", "avg_per_hour", "zero_share", "tier", "demand_class"]
    ]
    eda_clusters = read("EDA_doc/tables/zone_clusters.csv")[["PULocationID", "cluster_name"]]

    arm = read("data/processed/association_rules/arm_zone_thresholds.csv")[
        ["PULocationID", "q_low", "q_high", "active", "zone_cluster"]
    ]
    arm = arm.rename(columns={"zone_cluster": "arm_cluster", "active": "arm_active"})

    ts = read("models/time_series/model_1/tables/zone_selection.csv")[
        ["PULocationID", "model", "MAE", "volume_tier", "demand_pattern"]
    ]
    ts = ts.rename(columns={"model": "ts_method", "MAE": "ts_validation_MAE"})

    errors = read("models/modeling/model_3_error_by_zone.csv")[["PULocationID", "MAE", "actual_mean"]]
    errors = errors.rename(columns={"MAE": "hgb_test_MAE", "actual_mean": "test_mean_demand"})

    for table in [tiers, eda_clusters, arm, ts, errors]:
        zones = zones.merge(table, on="PULocationID", how="left")

    zones["label"] = zones["Zone"] + " (" + zones["Borough"] + ")"
    zones.to_csv(OUT / "zones.csv", index=False)
    print("zones.csv", zones.shape)


# ---------------------------------------------------------------------------
# 2. Test-period predictions of the baseline and the three regression models
#    in one table (25 Jan - 28 Feb 2026).
# ---------------------------------------------------------------------------
def build_test_predictions():
    keys = ["PULocationID", "pickup_hour"]
    base = read("models/modeling/baseline_168h_predictions.csv")
    base = base.rename(columns={"actual_demand": "actual", "baseline_prediction": "baseline"})
    table = base[keys + ["actual", "baseline"]]

    for file_name, column in [
        ("model_1_predictions.csv", "linear_regression"),
        ("model_2_predictions.csv", "random_forest"),
        ("model_3_predictions.csv", "hist_gradient_boosting"),
    ]:
        preds = read(f"models/modeling/{file_name}")[keys + ["prediction"]]
        preds = preds.rename(columns={"prediction": column})
        table = table.merge(preds, on=keys, how="left")

    table = table.round(2)
    table.to_csv(OUT / "test_predictions.csv", index=False)
    print("test_predictions.csv", table.shape)


# ---------------------------------------------------------------------------
# 3. Hourly demand in wide form (one column per zone or aggregate series).
#    The time-series forecaster reads these matrices directly.
#    trip_count_adj = pickups per physical hour (the repeated DST hour halved).
# ---------------------------------------------------------------------------
def build_hourly_matrices():
    zone_hourly = read("data/processed/time_series/ts_zone_hourly.csv", usecols=["PULocationID", "ds", "trip_count_adj"])
    zone_wide = zone_hourly.pivot(index="ds", columns="PULocationID", values="trip_count_adj")
    zone_wide.to_csv(OUT / "zone_hourly_wide.csv")
    print("zone_hourly_wide.csv", zone_wide.shape)

    agg_cols = ["series_id", "ds", "trip_count_adj", "apparent_temperature", "rain", "snowfall",
                "hour", "day_of_week", "day_name", "holiday_name", "special_day_name", "is_heavy_snow_day"]
    agg = read("data/processed/time_series/ts_aggregate_hourly.csv", usecols=agg_cols)
    agg.to_csv(OUT / "aggregate_hourly.csv", index=False)
    print("aggregate_hourly.csv", agg.shape)


# ---------------------------------------------------------------------------
# 4. Taxi zone shapes for the maps (official NYC TLC zones from NYC Open Data).
#    Downloaded once, reduced to our 138 zones and saved, so the dashboard maps
#    work offline. Delete data/taxi_zones.geojson to download it again.
# ---------------------------------------------------------------------------
ZONES_URL = "https://data.cityofnewyork.us/resource/8meu-9t5y.geojson?$limit=500"


def round_coordinates(coordinates):
    """Round every longitude/latitude to 4 decimals (about 10 m) to keep the file small."""
    if isinstance(coordinates[0], (int, float)):
        return [round(coordinates[0], 4), round(coordinates[1], 4)]
    return [round_coordinates(part) for part in coordinates]


def build_zone_shapes():
    path = OUT / "taxi_zones.geojson"
    if path.exists():
        print("taxi_zones.geojson already exists")
        return

    import json
    import urllib.request

    request = urllib.request.Request(ZONES_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        all_zones = json.load(response)

    our_ids = set(pd.read_csv(REPO / "data/raw/taxi_zones_manhattan_queens.csv")["LocationID"])
    polygons = {}  # a few zones come as several features; merge them per zone ID
    for feature in all_zones["features"]:
        zone_id = int(feature["properties"]["locationid"])
        if zone_id not in our_ids:
            continue
        geometry = feature["geometry"]
        parts = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        polygons.setdefault(zone_id, []).extend(round_coordinates(parts))

    features = [
        {"type": "Feature", "id": zone_id, "properties": {"id": zone_id},
         "geometry": {"type": "MultiPolygon", "coordinates": parts}}
        for zone_id, parts in sorted(polygons.items())
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print("taxi_zones.geojson", len(features), "zones")


# ---------------------------------------------------------------------------
# 5. "Normal" demand per zone for each weekday and hour, from the training
#    period only (before the test period starts). The demand predictor compares
#    its prediction with this to say "busier / quieter than usual".
# ---------------------------------------------------------------------------
def build_zone_normal():
    features = read("data/processed/features.csv", parse_dates=["pickup_hour"])
    train = features[(features["pickup_hour"] < "2026-01-25 04:00") & (~features["is_holiday"])]
    normal = (train.groupby(["PULocationID", "day_of_week", "hour"], as_index=False)["trip_count"]
              .mean().rename(columns={"trip_count": "normal"}))
    normal["normal"] = normal["normal"].round(2)
    normal.to_csv(OUT / "zone_normal.csv", index=False)
    print("zone_normal.csv", normal.shape)


if __name__ == "__main__":
    build_zones()
    build_zone_normal()
    build_test_predictions()
    build_hourly_matrices()
    build_zone_shapes()
    print("Done. Files saved in", OUT)
