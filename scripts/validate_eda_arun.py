"""Check EDA artifact coverage and exported statistics against source rows.

Run after scripts/eda_arun.py. Raises on stale inputs or inconsistent results.
"""
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "EDA_doc"
TABLES = OUT / "tables"


def validate():
    source = pd.read_csv(ROOT / "data/processed/features.csv", parse_dates=["pickup_hour"])
    zones = pd.read_csv(ROOT / "data/raw/taxi_zones_manhattan_queens.csv")
    data = source.merge(zones, left_on="PULocationID", right_on="LocationID", validate="many_to_one")
    data["date"] = data.pickup_hour.dt.normalize()
    total = int(data.trip_count.sum())
    checks = pd.read_csv(TABLES / "A1_A8_data_checks.csv")
    assert len(checks) == 15 and checks.passed.eq(True).all(), "Input checks failed"
    assert len(source) == 576288 and source.PULocationID.nunique() == 138
    assert not source.duplicated(["PULocationID", "pickup_hour"]).any()
    manifest = pd.read_csv(TABLES / "input_manifest.csv")
    assert len(manifest) == 7
    for row in manifest.itertuples():
        path = ROOT / row.input_path
        assert path.stat().st_size == row.bytes
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == row.sha256, f"Input changed: {row.input_path}"
    findings = (OUT / "EDA_findings.md").read_text(encoding="utf-8")
    expected = {"A1", "A2", "A3", "A5", "A6", "A7", "A8", "B1", "B2", "B3", "B4", "B5",
                "C3", "C4", "C5", "C9", "D3", "D4", "D5", "D6", "D8", "D9", "D10",
                "E1", "E2", "E3", "E4", "G9", "H3", "H4", "H5", "I1", "I2", "I3", "I4", "I5", "I6"}
    titles = re.findall(r"\*\*\[([^\]]+)\]", findings)
    found = {c for title in titles for c in re.findall(r"[A-I]\d+", title)}
    assert expected.issubset(found), f"Missing findings: {expected-found}"
    for document in ["EDA_findings.md", "README.md"]:
        text = (OUT / document).read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not target.startswith("http"):
                assert (OUT / target).exists(), f"Broken link in {document}: {target}"
    figures = list((OUT / "figures").glob("*.png"))
    assert len(figures) >= 35
    for path in figures:
        with Image.open(path) as image:
            assert min(image.size) >= 500, f"Undersized figure: {path.name}"
            image.verify()
    catalog = pd.read_csv(TABLES / "artifact_catalog.csv")
    actual_paths = {p.relative_to(OUT).as_posix() for p in figures}
    actual_paths.update(p.relative_to(OUT).as_posix() for p in TABLES.glob("*.csv") if p.name != "artifact_catalog.csv")
    assert set(catalog.path) == actual_paths, "Artifact catalog is incomplete"

    weekly = pd.read_csv(TABLES / "D3_weekly_totals.csv")
    monthly = pd.read_csv(TABLES / "D4_monthly_trips_per_day.csv")
    assert weekly.total.sum() == monthly.total.sum() == total
    assert weekly.observed_days.sum() == monthly.observed_days.sum() == 174
    assert weekly.complete_week.sum() == 24 and weekly.observed_days.iloc[-1] == 6
    assert monthly.observed_days.iloc[0] == 23
    np.testing.assert_allclose(monthly.mean_per_day, monthly.total/monthly.observed_days, rtol=1e-7)
    shares = pd.read_csv(TABLES / "D9_time_band_share.csv").drop(columns="year_month")
    np.testing.assert_allclose(shares.sum(axis=1), 100, atol=1e-5)
    ratio = pd.read_csv(TABLES / "C9_weekend_weekday_by_zone.csv").set_index("PULocationID").sort_index()
    by_type = data.groupby(["PULocationID", "is_weekend"]).trip_count.mean().unstack()
    np.testing.assert_allclose(ratio.weekend_weekday_ratio, by_type[True]/by_type[False].replace(0, np.nan), rtol=1e-7, equal_nan=True)
    assert ratio.weekday_hours.eq(3000).all() and ratio.weekend_hours.eq(1176).all()
    coverage = pd.read_csv(TABLES / "I2_topN_hourly_coverage.csv").set_index("hour")
    hourly_zone_total = data.groupby(["hour", "PULocationID"]).trip_count.sum()
    top10 = pd.read_csv(TABLES / "I1_top10_zones_each_hour.csv")
    assert len(top10) == 240 and top10.groupby("hour").size().eq(10).all()
    for h in range(24):
        hour = hourly_zone_total.loc[h].sort_index().sort_values(ascending=False, kind="stable")
        assert top10.loc[top10.hour.eq(h), "PULocationID"].tolist() == hour.head(10).index.tolist()
        for n in [5, 10, 20]:
            np.testing.assert_allclose(coverage.loc[h, f"top_{n}_share_pct"], 100*hour.head(n).sum()/hour.sum(), rtol=1e-7)
    assert (coverage.top_5_share_pct <= coverage.top_10_share_pct).all()
    assert (coverage.top_10_share_pct <= coverage.top_20_share_pct).all()
    peaks = pd.read_csv(TABLES / "H5_I3_zone_peak_demand.csv").set_index("PULocationID").sort_index()
    actual_peaks = data.groupby("PULocationID").trip_count
    np.testing.assert_allclose(peaks.p95, actual_peaks.quantile(.95), rtol=1e-7)
    np.testing.assert_allclose(peaks.p99, actual_peaks.quantile(.99), rtol=1e-7)
    assert (peaks.p99 >= peaks.p95).all() and (peaks.maximum >= peaks.p99).all()
    consistency = pd.read_csv(TABLES / "H3_zone_consistency.csv")
    correlations = pd.read_csv(TABLES / "H3_weekly_profile_correlations.csv")
    assert len(consistency) == 138 and len(correlations) == 138*24
    assert correlations.profile_correlation.dropna().between(-1.000001, 1.000001).all()
    change = pd.read_csv(TABLES / "H4_hourly_change_summary.csv")
    assert change.rows.sum() == 138*(4176-1-2), "H4 must omit first rows and both DST transitions"

    exclusions = set(pd.to_datetime(pd.read_csv(TABLES / "E1_control_exclusions.csv").date))
    daily = {"All": data.groupby("date").trip_count.sum(),
             **{b: g.groupby("date").trip_count.sum() for b, g in data.groupby("Borough")}}
    for filename, expected_rows in [("holiday_effects.csv", 21), ("E4_special_day_effects.csv", 12)]:
        effects = pd.read_csv(TABLES / filename)
        assert len(effects) == expected_rows
        for row in effects.itertuples():
            event = pd.Timestamp(row.date)
            assert daily[row.scope].loc[event] == row.actual_trips
            controls = pd.to_datetime(row.control_dates.split(";"))
            assert len(controls) == row.control_days and 3 <= len(controls) <= 6
            assert event not in controls and not set(controls) & exclusions
            assert all(d.dayofweek == event.dayofweek and abs((d-event).days) <= 28 for d in controls)
            normal = daily[row.scope].loc[controls].mean()
            np.testing.assert_allclose(row.normal_mean_trips, normal, rtol=1e-7)
            np.testing.assert_allclose(row.pct_change, 100*(row.actual_trips/normal-1), atol=1e-5)
            wider = pd.to_datetime(row.control_dates_window42.split(";"))
            assert not set(wider) & exclusions and all(d.dayofweek == event.dayofweek and abs((d-event).days) <= 42 for d in wider)
            np.testing.assert_allclose(row.pct_change_window42, 100*(row.actual_trips/daily[row.scope].loc[wider].mean()-1), atol=1e-5)
    night = pd.read_csv(TABLES / "E3_new_year_night_by_zone.csv").set_index("PULocationID").sort_index()
    actual_night = data.loc[data.pickup_hour.between("2025-12-31 20:00", "2026-01-01 04:00")].groupby("PULocationID").trip_count.sum()
    assert len(night) == 138
    np.testing.assert_array_equal(night.actual_night_trips, actual_night)
    night_hour = pd.read_csv(TABLES / "E3_new_year_night_hourly.csv")
    assert len(night_hour) == 138*9 and night_hour.actual_trips.sum() == actual_night.sum()
    np.testing.assert_allclose(night.normal_night_trips.sum(), night_hour.normal_mean_trips.sum(), rtol=1e-7)
    night_hotspots = pd.read_csv(TABLES / "I5_night_hotspots.csv")
    weekend = data.loc[((data.day_of_week.isin([4, 5])) & data.hour.ge(20)) |
                       ((data.day_of_week.isin([5, 6])) & data.hour.le(4))]
    weekend_stats = weekend.groupby("PULocationID").trip_count.mean()
    saved_weekend = night_hotspots.loc[night_hotspots.period.eq("weekend_night_Fri_Sat_20_04")].set_index("PULocationID").sort_index()
    np.testing.assert_allclose(saved_weekend.mean_per_hour, weekend_stats, rtol=1e-7)
    print(f"Validated 37 assigned IDs, {len(figures)} figures, {len(actual_paths)-len(figures)} tables, 7 input hashes, and source-derived calendar/ranking/holiday/night statistics.")


if __name__ == "__main__":
    validate()
