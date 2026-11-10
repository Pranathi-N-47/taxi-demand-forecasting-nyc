"""Prepare frozen, chronological association-rule transactions for issue #30.

Run: python scripts/association_rule_data_preparation.py
This prepares data only. Mining and rule evaluation belong to issues #31--#33.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/association_rules"
DST_HOUR = pd.Timestamp("2025-11-02 01:00:00")
MIN_ACTIVE_MEAN = 1.0
FAMILIES = {
    "time_band": ["late_night", "early_morning", "am_rush", "midday", "pm_rush", "evening"],
    "day_type": ["weekday", "weekend"],
    "holiday": ["regular", "federal"],
    "rain": ["no", "yes"],
    "snow": ["no", "yes"],
    "temperature": ["lower", "middle", "upper"],
    "zone_cluster": ["daytime", "morning_commute", "nightlife", "near_empty"],
    "demand": ["Low", "Medium", "High"],
}
SOURCE_COLUMNS = ["PULocationID", "pickup_hour", "trip_count", "apparent_temperature",
                  "rain", "snowfall", "precipitation_probability", "hour", "day_of_week",
                  "month", "is_weekend", "is_holiday"]


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def load_inputs(root=ROOT):
    root = Path(root)
    frame = pd.read_csv(root / "data/processed/features.csv", parse_dates=["pickup_hour"])
    require(set(SOURCE_COLUMNS) <= set(frame), "Required source columns are missing")
    frame = frame[SOURCE_COLUMNS].copy()
    zones = pd.read_csv(root / "data/raw/taxi_zones_manhattan_queens.csv")
    summary = pd.read_csv(root / "models/modeling/split_summary.csv", parse_dates=["start", "end"])
    require(set(summary.dataset) == {"train", "test"} and len(summary) == 2, "Expected official train/test split")
    require(not frame.isna().any().any(), "Source contains missing values")
    require(not frame.duplicated(["PULocationID", "pickup_hour"]).any(), "Duplicate zone-hours")
    require(not zones.LocationID.duplicated().any(), "Duplicate lookup zone IDs")
    require(set(frame.PULocationID) == set(zones.LocationID), "Source/lookup zones differ")
    require(all(pd.api.types.is_bool_dtype(frame[c]) for c in ["is_weekend", "is_holiday"]), "Invalid flag types")
    require(pd.api.types.is_integer_dtype(frame.trip_count) and frame.trip_count.ge(0).all(), "Invalid pickup counts")
    numeric = frame.select_dtypes("number")
    require(np.isfinite(numeric).all().all(), "Nonfinite source numbers")
    require(frame.rain.ge(0).all() and frame.snowfall.ge(0).all(), "Negative precipitation")
    require(frame.precipitation_probability.between(0, 100).all(), "Probability outside 0--100")
    require(frame.hour.eq(frame.pickup_hour.dt.hour).all()
            and frame.day_of_week.eq(frame.pickup_hour.dt.dayofweek).all()
            and frame.month.eq(frame.pickup_hour.dt.month).all()
            and frame.is_weekend.eq(frame.day_of_week.ge(5)).all(), "Calendar fields disagree with timestamps")
    train = summary.set_index("dataset").loc["train"]
    test = summary.set_index("dataset").loc["test"]
    require(train.end + pd.Timedelta(hours=1) == test.start, "Split must have adjacent disjoint periods")
    hours = pd.date_range(train.start, test.end, freq="h")
    require(set(frame.pickup_hour) == set(hours), "Unexpected or missing hour labels")
    require(frame.groupby("PULocationID").size().eq(len(hours)).all(), "Incomplete per-zone grid")
    frame["split"] = np.where(frame.pickup_hour.le(train.end), "train", "test")
    for row in summary.itertuples():
        part = frame.loc[frame["split"].eq(row.dataset)]
        require(len(part) == row.rows and part.pickup_hour.nunique() == row.timestamps,
                f"{row.dataset} counts disagree with official split")
        require(part.pickup_hour.min() == row.start and part.pickup_hour.max() == row.end,
                f"{row.dataset} dates disagree with official split")
    frame = frame.merge(zones, left_on="PULocationID", right_on="LocationID", validate="many_to_one").drop(columns="LocationID")
    frame["is_dst_merged_hour"] = frame.pickup_hour.eq(DST_HOUR)
    frame = frame.sort_values(["pickup_hour", "PULocationID"]).reset_index(drop=True)
    return frame, summary


def fit_parameters(frame):
    """Learn every data-dependent choice solely from the non-DST train rows."""
    train = frame.loc[frame["split"].eq("train") & ~frame.is_dst_merged_hour].copy()
    require(len(train) > 0, "Empty training partition")
    stats = train.groupby("PULocationID").trip_count.agg(
        train_hours="size", train_mean="mean", train_zero_share=lambda s: s.eq(0).mean(),
        q_low=lambda s: s.quantile(1/3), q_high=lambda s: s.quantile(2/3),
        distinct_counts="nunique")
    stats["active"] = stats.train_mean.ge(MIN_ACTIVE_MEAN)
    stats["tied_thresholds"] = stats.q_low.eq(stats.q_high)
    stats["zone_cluster"] = "near_empty"
    active_ids = stats.index[stats.active]
    require(len(active_ids) >= 3, "At least three active zones required for k=3 clustering")
    normal = train.loc[train.PULocationID.isin(active_ids) & ~train.is_holiday]
    slots = pd.MultiIndex.from_product([range(7), range(24)], names=["day_of_week", "hour"])
    weekly = normal.groupby(["PULocationID", "day_of_week", "hour"]).trip_count.mean().unstack(["day_of_week", "hour"])
    weekly = weekly.reindex(index=active_ids, columns=slots)
    require(not weekly.isna().any().any(), "Training history does not cover every weekly profile slot")
    shape = weekly.div(weekly.sum(axis=1), axis=0)
    model = KMeans(n_clusters=3, n_init=20, random_state=42)
    labels = model.fit_predict(shape.to_numpy())
    centers = model.cluster_centers_
    # Name centroids using training-only timing. No EDA full-period labels reused.
    weekend_night = np.array([(d in [4, 5] and h >= 20) or (d in [5, 6] and h <= 4) for d, h in slots])
    weekday_am = np.array([d < 5 and 7 <= h <= 9 for d, h in slots])
    weekday_midday = np.array([d < 5 and 10 <= h <= 15 for d, h in slots])
    nightlife = int(np.argmax(centers[:, weekend_night].sum(axis=1)))
    remaining = [c for c in range(3) if c != nightlife]
    morning = max(remaining, key=lambda c: centers[c, weekday_am].mean()/centers[c, weekday_midday].mean())
    daytime = next(c for c in remaining if c != morning)
    names = {nightlife: "nightlife", morning: "morning_commute", daytime: "daytime"}
    stats.loc[active_ids, "zone_cluster"] = [names[int(c)] for c in labels]
    temperature = train.loc[train.PULocationID.isin(active_ids), "apparent_temperature"]
    cuts = temperature.quantile([1/3, 2/3]).tolist()
    require(cuts[0] < cuts[1], "Temperature quantiles collapse; review bins before proceeding")
    params = {
        "schema_version": 1,
        "fit_partition": "train, excluding the DST merged hour",
        "fit_start": str(train.pickup_hour.min()), "fit_end": str(train.pickup_hour.max()),
        "fit_rows": len(train), "min_active_mean": MIN_ACTIVE_MEAN,
        "temperature_cuts_c": cuts,
        "temperature_fit_rows": len(temperature),
        "quantiles": [1/3, 2/3], "quantile_interpolation": "linear",
        "demand_intervals": "Low: count <= q_low; Medium: q_low < count <= q_high; High: count > q_high",
        "tie_policy": "Never split identical counts or rank ties into artificial classes; lower interval wins boundaries",
        "cluster_method": {"algorithm": "KMeans", "k": 3, "n_init": 20, "random_state": 42,
                           "fit": "nonholiday active-zone training profiles, 168 slots, each normalised to sum 1",
                           "naming": "largest weekend-night share, then largest weekday AM/midday ratio, then daytime"},
        "cluster_centers": {names[c]: centers[c].tolist() for c in range(3)},
        "zones": stats.reset_index().to_dict(orient="records"),
        "families": FAMILIES,
    }
    return params, shape


def transform(frame, params):
    """Apply saved thresholds and zone assignments without refitting."""
    profiles = pd.DataFrame(params["zones"]).set_index("PULocationID")
    require(set(frame.PULocationID) <= set(profiles.index), "New zone ID absent from fitted parameters")
    rows = frame.copy()
    rows["transaction_id"] = rows.PULocationID.astype(str) + "@" + rows.pickup_hour.dt.strftime("%Y-%m-%dT%H:%M:%S")
    rows["time_band"] = pd.cut(rows.hour, [-1, 4, 6, 9, 15, 19, 23], labels=FAMILIES["time_band"]).astype(str)
    rows["day_type"] = np.where(rows.is_weekend, "weekend", "weekday")
    rows["holiday"] = np.where(rows.is_holiday, "federal", "regular")
    # Raw weather/counts stay in metadata; item strings are categorical.
    rows["rain_mm"] = rows.rain
    rows["snowfall_cm"] = rows.snowfall
    rows["rain"] = np.where(rows.rain_mm.gt(0), "yes", "no")
    rows["snow"] = np.where(rows.snowfall_cm.gt(0), "yes", "no")
    low, high = params["temperature_cuts_c"]
    rows["temperature"] = np.select([rows.apparent_temperature.le(low), rows.apparent_temperature.le(high)],
                                     ["lower", "middle"], default="upper")
    rows["zone_cluster"] = rows.PULocationID.map(profiles.zone_cluster)
    q_low = rows.PULocationID.map(profiles.q_low)
    q_high = rows.PULocationID.map(profiles.q_high)
    rows["demand"] = np.select([rows.trip_count.le(q_low), rows.trip_count.le(q_high)], ["Low", "Medium"], default="High")
    rows["active_zone"] = rows.PULocationID.map(profiles.active).astype(bool)
    rows["mining_eligible"] = rows.active_zone & ~rows.is_dst_merged_hour
    rows["exclusion_reason"] = np.select(
        [rows.is_dst_merged_hour, ~rows.active_zone], ["dst_merged_hour", "train_mean_below_1"], default="none")
    columns = ["transaction_id", "PULocationID", "pickup_hour", "split", "Borough", "Zone", "service_zone",
               "trip_count", "apparent_temperature", "rain_mm", "snowfall_cm", "precipitation_probability",
               "day_of_week", "hour", "active_zone", "is_dst_merged_hour", "mining_eligible", "exclusion_reason", *FAMILIES]
    return rows[columns].reset_index(drop=True)


def item_catalog():
    records = []
    for family, values in FAMILIES.items():
        for value in values:
            records.append({"column_index": len(records), "item": f"{family}={value}", "family": family,
                            "value": value, "role": "consequent_only" if family == "demand" else "antecedent"})
    return pd.DataFrame(records)


def encode(rows, catalog=None):
    """One Boolean item per family per row, with a fixed train/test column order."""
    if catalog is None:
        catalog = item_catalog()
    item_indices = dict(zip(catalog["item"], catalog.column_index))
    encoded = []
    for family in FAMILIES:
        require(rows[family].isin(FAMILIES[family]).all(), f"Unknown {family} category")
        encoded.append((family+"="+rows[family]).map(item_indices).to_numpy(dtype=np.int32))
    indices = np.stack(encoded, axis=1).reshape(-1)
    indptr = np.arange(0, len(indices)+1, len(FAMILIES), dtype=np.int32)
    matrix = sparse.csr_matrix((np.ones(len(indices), dtype=bool), indices, indptr), shape=(len(rows), len(catalog)))
    matrix.sort_indices()
    return matrix


def load_transactions(split="train", output_dir=OUTPUT):
    """Return (Boolean sparse pandas DataFrame, aligned metadata) for mlxtend.

    Supports only the prepared active-zone, non-DST mining population.
    No IDs, timestamps, raw counts or metadata become mining items.
    """
    require(split in ["train", "test"], "Choose train or test")
    output_dir = Path(output_dir)
    catalog = pd.read_csv(output_dir / "arm_item_catalog.csv")
    rows = pd.read_parquet(output_dir / f"arm_{split}_rows.parquet")
    matrix = sparse.load_npz(output_dir / f"arm_{split}_onehot.npz")
    require(matrix.shape == (len(rows), len(catalog)), "Matrix/metadata/catalog sizes disagree")
    require(matrix.dtype == bool, "Expected Boolean matrix")
    frame = pd.DataFrame.sparse.from_spmatrix(matrix, index=pd.Index(rows.transaction_id, name="transaction_id"), columns=catalog["item"])
    return frame, rows.set_index("transaction_id", drop=False)


def validate_artifacts(output_dir=OUTPUT, source_root=ROOT):
    """Check saved files, alignments, categorical semantics and frozen cutoffs."""
    output_dir = Path(output_dir)
    params = json.loads((output_dir / "arm_preparation_parameters.json").read_text())
    all_rows = pd.read_parquet(output_dir / "arm_transactions.parquet")
    catalog = pd.read_csv(output_dir / "arm_item_catalog.csv")
    profiles = pd.DataFrame(params["zones"]).set_index("PULocationID")
    records = []

    def check(name, ok, detail):
        records.append({"check": name, "passed": bool(ok), "detail": str(detail)})

    manifest = json.loads((output_dir / "arm_input_manifest.json").read_text())
    for entry in manifest:
        path = Path(source_root) / entry["path"]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        check("Unchanged input: "+entry["path"], path.stat().st_size == entry["bytes"] and digest == entry["sha256"], "SHA-256")
    original_split = pd.read_csv(output_dir / "arm_source_split.csv")
    check("Complete audit grid", all_rows.PULocationID.nunique() == 138
          and len(all_rows) == original_split.rows.sum()
          and not all_rows.duplicated(["PULocationID", "pickup_hour"]).any(), len(all_rows))
    check("Audit transaction IDs", all_rows.transaction_id.is_unique, len(all_rows))
    check("Audit missing values", not all_rows.isna().any().any(), "all fields")
    check("Frozen parameter scope", pd.Timestamp(params["fit_end"]) < all_rows.loc[all_rows["split"].eq("test"), "pickup_hour"].min(), params["fit_end"])
    check("Catalog roles", catalog.role.eq(np.where(catalog.family.eq("demand"), "consequent_only", "antecedent")).all(), "demand kept out of antecedents")
    check("Unique items", catalog["item"].is_unique and catalog.column_index.tolist() == list(range(len(catalog))), len(catalog))
    check("Metadata excluded from items", set(catalog.family) == set(FAMILIES), "no raw counts, lags, IDs or predictions")
    q_low = all_rows.PULocationID.map(profiles.q_low)
    q_high = all_rows.PULocationID.map(profiles.q_high)
    expected = np.select([all_rows.trip_count.le(q_low), all_rows.trip_count.le(q_high)], ["Low", "Medium"], default="High")
    check("Demand interval boundaries", np.array_equal(expected, all_rows.demand), "including ties and zeros")
    low, high = params["temperature_cuts_c"]
    expected_temp = np.select([all_rows.apparent_temperature.le(low), all_rows.apparent_temperature.le(high)], ["lower", "middle"], default="upper")
    check("Frozen temperature bins", np.array_equal(expected_temp, all_rows.temperature), [low, high])
    check("Weather flags", all_rows.rain.eq(np.where(all_rows.rain_mm.gt(0), "yes", "no")).all()
          and all_rows.snow.eq(np.where(all_rows.snowfall_cm.gt(0), "yes", "no")).all(), "rain mm >0; snow cm >0")
    check("DST exclusion", not all_rows.loc[all_rows.is_dst_merged_hour, "mining_eligible"].any(), "combined hour remains in audit")
    train_ids = set()
    for split in ["train", "test"]:
        rows = pd.read_parquet(output_dir / f"arm_{split}_rows.parquet")
        matrix = sparse.load_npz(output_dir / f"arm_{split}_onehot.npz")
        expected_rows = all_rows.loc[all_rows["split"].eq(split) & all_rows.mining_eligible].reset_index(drop=True)
        check(f"{split} row alignment", rows.equals(expected_rows), len(rows))
        check(f"{split} Boolean matrix", matrix.dtype == bool and matrix.shape == (len(rows), len(catalog)), matrix.shape)
        check(f"{split} exact transaction encoding", (matrix != encode(rows, catalog)).nnz == 0, "saved matrix round trip")
        check(f"{split} one item per family", all(np.all(np.asarray(matrix[:, catalog.loc[catalog.family.eq(f), 'column_index']].sum(axis=1)).ravel() == 1) for f in FAMILIES), len(FAMILIES))
        check(f"{split} valid mining population", rows.mining_eligible.all() and rows.active_zone.all() and not rows.is_dst_merged_hour.any(), rows.PULocationID.nunique())
        check(f"{split} zeros retained", rows.trip_count.eq(0).sum() == expected_rows.trip_count.eq(0).sum(), int(rows.trip_count.eq(0).sum()))
        if split == "train":
            train_ids = set(rows.transaction_id)
            train_end = rows.pickup_hour.max()
        else:
            check("Disjoint chronology", not train_ids & set(rows.transaction_id) and train_end < rows.pickup_hour.min(), "same cutoff across zones")
        frequencies = pd.read_csv(output_dir / "arm_item_frequencies.csv")
        saved_counts = frequencies.loc[frequencies["split"].eq(split)].sort_values("column_index").count_transactions.to_numpy()
        check(f"{split} item frequencies", np.array_equal(saved_counts, np.asarray(matrix.sum(axis=0)).ravel()), len(catalog))
    out = pd.DataFrame(records)
    out.to_csv(output_dir / "arm_preparation_checks.csv", index=False)
    require(out.passed.all(), f"Saved artifact checks failed: {out.loc[~out.passed].to_dict(orient='records')}")
    return out


def prepare(root=ROOT, output_dir=OUTPUT):
    root, output_dir = Path(root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, split_summary = load_inputs(root)
    params, shape = fit_parameters(frame)
    rows = transform(frame, params)
    catalog = item_catalog()
    rows.to_parquet(output_dir / "arm_transactions.parquet", index=False)
    catalog.to_csv(output_dir / "arm_item_catalog.csv", index=False)
    (output_dir / "arm_preparation_parameters.json").write_text(json.dumps(params, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    pd.DataFrame(params["zones"]).merge(frame.groupby("PULocationID")[["Borough", "Zone", "service_zone"]].first(), on="PULocationID").to_csv(output_dir / "arm_zone_thresholds.csv", index=False)
    shape.columns = [f"weekday_{d}_hour_{h:02d}" for d, h in shape.columns]
    shape.to_csv(output_dir / "arm_train_weekly_profiles.csv", index_label="PULocationID")
    pd.DataFrame(params["cluster_centers"]).rename_axis("hour_of_week").reset_index().to_csv(output_dir / "arm_cluster_centers.csv", index=False)
    split_summary.to_csv(output_dir / "arm_source_split.csv", index=False)
    audit = rows.groupby(["split", "exclusion_reason"]).agg(rows=("transaction_id", "size"), zones=("PULocationID", "nunique"), pickups=("trip_count", "sum")).reset_index()
    audit.to_csv(output_dir / "arm_row_audit.csv", index=False)
    frequencies, balances = [], []
    for split in ["train", "test"]:
        part = rows.loc[rows["split"].eq(split) & rows.mining_eligible].reset_index(drop=True)
        matrix = encode(part, catalog)
        part.to_parquet(output_dir / f"arm_{split}_rows.parquet", index=False)
        sparse.save_npz(output_dir / f"arm_{split}_onehot.npz", matrix, compressed=True)
        freq = catalog.copy()
        freq["split"] = split
        freq["count_transactions"] = np.asarray(matrix.sum(axis=0)).ravel()
        freq["transactions"] = len(part)
        freq["support"] = freq.count_transactions / len(part)
        frequencies.append(freq)
        balance = part.groupby(["zone_cluster", "demand"]).size().rename("transactions").reset_index()
        balance["split"] = split
        balance["share_within_cluster"] = balance.transactions / balance.groupby("zone_cluster").transactions.transform("sum")
        balances.append(balance)
        print(f"{split}: {len(part):,} transactions, {matrix.shape[1]} items, {part.PULocationID.nunique()} active zones", flush=True)
    pd.concat(frequencies, ignore_index=True).to_csv(output_dir / "arm_item_frequencies.csv", index=False)
    pd.concat(balances, ignore_index=True).to_csv(output_dir / "arm_demand_class_balance.csv", index=False)
    inputs = ["data/processed/features.csv", "data/raw/taxi_zones_manhattan_queens.csv", "models/modeling/split_summary.csv"]
    manifest = []
    for p in inputs:
        with (root/p).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        manifest.append({"path": p, "bytes": (root/p).stat().st_size, "sha256": digest})
    (output_dir / "arm_input_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    (output_dir / "arm_environment.json").write_text(json.dumps(versions, indent=2)+"\n", encoding="utf-8")
    checks = validate_artifacts(output_dir, source_root=root)
    print(f"Audit: {len(rows):,} original rows retained. {len(checks)} saved-file checks pass.", flush=True)
    return rows, params, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_artifacts(args.output_dir)
        print(f"{len(result)} checks passed")
    else:
        prepare(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
