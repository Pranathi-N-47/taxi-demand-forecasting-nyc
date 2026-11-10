# Data Processing & Feature Engineering

This document describes how the raw trip dataset was cleaned and transformed into the analysis-ready feature tables used by all downstream modelling and visualisation work. It covers every decision made during processing: which records were removed, how the hourly demand grid was built, how weather and calendar features were added, and how lag and rolling features were computed. Field-level descriptions of the final output columns are in [data_dictionary.md](data_dictionary.md). How the raw data was collected is in [data_collection.md](data_collection.md).

## 1. Scope

| Item | Value |
|---|---|
| Input | `data/raw/NYC-taxi-demand-Manhattan-Queens.csv` — 23,318,979 trip-level rows, uncleaned |
| Supporting inputs | `data/raw/weather_hourly_by_zone.csv`, `data/raw/taxi_zones_manhattan_queens.csv`, `data/raw/taxi_collection_summary.csv` |
| Code | [`scripts/data_processing_engineering.ipynb`](../../scripts/data_processing_engineering.ipynb) |
| Grain of final output | **One row per zone per hour** (not one row per trip) |
| Spatial scope | 138 Manhattan and Queens TLC taxi zones (unchanged from collection) |
| Temporal scope | 2025-09-01 00:00 to 2026-02-28 23:59 (4,344 hours) |
| Target variable | `trip_count` — number of clean pickups in that zone during that hour |
| Final outputs | Three feature CSVs in `data/processed/` — one per borough and one combined |

Processing does not add new data sources. It only reorganises and cleans what the collection stage produced. `data/raw/` is never modified.

## 2. Processing steps

All code is in one notebook, [`scripts/data_processing_engineering.ipynb`](../../scripts/data_processing_engineering.ipynb). Run all cells from top to bottom to reproduce every output in `data/processed/`. The notebook is divided into sections that correspond to the steps below.

```
NYC-taxi-demand-Manhattan-Queens.csv (23,318,979 trip rows, uncleaned)
              |
              v
    Step 1: Data Cleaning
    Apply four removal rules -> trips_clean.csv (< 23 M clean trip rows)
    + cleaning_report.md
              |
              v
    Step 2: Hourly Aggregation & Zero-Fill
    Count pickups per zone per hour; join onto the full 138-zone x 4,344-hour
    grid from weather_hourly_by_zone.csv; fill zone-hours with no trips -> 0
    -> hourly_demand.csv (599,472 rows)
              |
              v
    Step 3: Weather Merge
    Join 4 weather columns from weather_hourly_by_zone.csv on (zone, hour)
    -> hourly_demand.csv + weather columns
              |
              v
    Step 4: Time Features
    Derive hour, day_of_week, month, is_weekend, is_holiday from pickup_hour
    -> hourly_demand.csv + calendar columns
              |
              v
    Step 5: Lag & Rolling Features
    Compute lagged and rolling trip counts within each zone; drop warm-up rows
    -> features.csv / features_manhattan.csv / features_queens.csv
```

### Step 1: Data Cleaning

The raw trip file was collected without any cleaning. Four removal rules are applied to the trip-level data before it is aggregated:

| Rule | What is removed | Why |
|---|---|---|
| 1. Exact duplicates | Rows where all five trip fields (`taxi_type`, `pickup_datetime`, `dropoff_datetime`, `PULocationID`, `DOLocationID`) are identical to another row in the dataset | Likely double-recorded trips. The first occurrence is kept; the rest are dropped |
| 2. Dropoff at or before pickup | Trips where `dropoff_datetime <= pickup_datetime` | Meter clock errors. These trips have a zero or negative duration and cannot be real |
| 3. Duration over 6 hours | Trips where `dropoff_datetime - pickup_datetime > 6 h` | Implausibly long trips. The most likely cause is a driver forgetting to end the trip in the meter system |
| 4. Unknown dropoff zone | Trips where `DOLocationID` is 264 (Unknown) or 265 (Outside of NYC) | The drop-off zone is missing or unresolvable. These trips cannot be used for any analysis that involves the destination |

Rules are applied in the order listed. Each count is computed on the rows remaining after all prior rules have been applied, so the counts are not overlapping. The expected counts for each rule were cross-checked against `taxi_collection_summary.csv`, which recorded them at collection time.

**Daylight saving time (DST) on 2025-11-02.** When the clock fell back at 02:00, the hour 01:00-01:59 occurred twice. TLC timestamps cannot distinguish the two occurrences, so both map to `pickup_hour` 2025-11-02 01:00. This hour has roughly double the usual trip count (~10,850 trips instead of ~6,000). These trips are **not removed** because they represent real demand; the aggregated count for that hour is inflated by about one hour's worth of trips. This is noted in the cleaning report and in section 5 below.

After cleaning, the results and per-rule counts are written to `data/processed/cleaning_report.md`. The cleaned trip-level file is saved to `data/processed/trips_clean.csv`.

### Step 2: Hourly Aggregation & Zero-Fill

After cleaning, the trip-level data is aggregated to one row per zone per hour.

**Deriving the pickup hour.** `pickup_datetime` is truncated to the hour: a trip starting at 07:42 belongs to hour 07:00. This is the same rounding used at collection time to join weather to trips, so the two match exactly.

**Counting trips.** Trips are grouped by `PULocationID` and `pickup_hour` and counted. This gives a sparse table: only zone-hour pairs that had at least one clean pickup appear.

**Zero-fill with the canonical grid.** A plain groupby produces no row for zone-hours with zero pickups. These silent hours are real observations (zero demand), and leaving them out would bias all downstream statistics — averages, lags and rolling windows would be computed only over non-zero hours, overstating typical demand. To fix this, the full **138 zones x 4,344 hours = 599,472 row grid** is taken from `weather_hourly_by_zone.csv`, which already has one row for every zone and every hour in the study period. The sparse trip counts are left-joined onto this grid, and any zone-hour with no trips is filled with `trip_count = 0`.

After the join, the row count is asserted to equal exactly 599,472. The share of zero rows is printed and recorded.

### Step 3: Weather Merge

Four weather variables are joined from `weather_hourly_by_zone.csv` onto the hourly demand grid:

| Column | What it records | Unit |
|---|---|---|
| `apparent_temperature` | "Feels-like" temperature at the start of the pickup hour | C |
| `rain` | Rain from large-scale weather systems during the pickup hour | mm |
| `snowfall` | Snowfall during the pickup hour | cm |
| `precipitation_probability` | Model probability of more than 0.1 mm of precipitation during the pickup hour | % (0-100) |

The join key is `(PULocationID, pickup_hour)` = `(LocationID, weather_hour)` in the weather table. Because the demand grid was built directly from the weather table, every row in the grid is guaranteed to have exactly one matching weather row. After the join, a null count assertion confirms that no weather values are missing.

Weather values for the DST double-hour (2025-11-02 01:00) come from the single Open-Meteo row for that timestamp, so all ~10,850 trips from both clock-cycles share the same weather record. This is a consequence of the DST issue noted in Step 1 and is unchanged from the raw dataset.

### Step 4: Time Features

Calendar and period features are derived from `pickup_hour`:

| Feature | Values | How it is computed |
|---|---|---|
| `hour` | 0-23 | `pickup_hour.dt.hour` |
| `day_of_week` | 0 (Mon) - 6 (Sun) | `pickup_hour.dt.dayofweek` |
| `month` | 9-12, 1-2 | `pickup_hour.dt.month` |
| `is_weekend` | True / False | True when `day_of_week >= 5` (Saturday or Sunday) |
| `is_holiday` | True / False | True when the date is a US federal holiday (using the `holidays` Python package) |

**Why not cyclical encodings.** Sine/cosine transforms of `hour` and `day_of_week` are sometimes used so that tree-based models see hour 23 and hour 0 as adjacent. This project keeps the raw integer features because gradient-boosted tree models (XGBoost, LightGBM) handle ordinal integers correctly, and the raw values are more interpretable in feature importance plots. Cyclical encodings can be added later if neural-network models are introduced.

**US federal holidays in the study period.** The following dates are marked `is_holiday = True`:

| Date | Holiday |
|---|---|
| 2025-09-01 | Labor Day |
| 2025-10-13 | Columbus Day |
| 2025-11-11 | Veterans Day |
| 2025-11-27 | Thanksgiving Day |
| 2025-12-25 | Christmas Day |
| 2026-01-01 | New Year's Day |
| 2026-01-19 | Martin Luther King Jr. Day |
| 2026-02-16 | Presidents' Day |

Each holiday marks all 24 hours of that date as `is_holiday = True`. State and city holidays (for example, the NYC Marathon) are not included.

After adding time features, three assertions confirm the ranges are correct:
- `hour` is in 0-23 for all rows
- `day_of_week` is in 0-6 for all rows
- `month` is in the set {9, 10, 11, 12, 1, 2} for all rows

### Step 5: Lag & Rolling Features

Lag and rolling features carry past demand into each row as model inputs. All features are computed **within each zone separately**, after sorting each zone's rows by `pickup_hour` ascending. Sorting before computing is required — without it, lags would be taken from the wrong rows and the model would see future demand (data leakage).

**Lag features.** A lag of *k* hours takes the `trip_count` value from *k* rows earlier in the same zone's sorted sequence.

| Feature | Lag | What it captures |
|---|---|---|
| `trip_count_lag_1h` | 1 hour | Immediate prior demand; the strongest short-term predictor |
| `trip_count_lag_24h` | 24 hours | Demand at the same hour yesterday; captures the daily cycle |
| `trip_count_lag_168h` | 168 hours (7 days) | Demand at the same hour last week; captures the weekly pattern |

**Rolling features.** A rolling mean of *w* hours is computed over the *w* hours immediately before the current row (the current row is excluded to prevent leakage). This is implemented by shifting the series one step before applying the rolling window.

| Feature | Window | What it captures |
|---|---|---|
| `trip_count_roll_3h` | 3 hours | Short-term trend; smooths the 1-hour lag |
| `trip_count_roll_24h` | 24 hours | Broader daily pattern; less sensitive to individual spikes |

**Warm-up period.** The 168-hour lag requires at least 168 prior rows per zone before it can produce a value. The first 168 rows of each zone (the first 7 days, 2025-09-01 to 2025-09-07) have NaN for `trip_count_lag_168h` and are dropped. This removes 168 x 138 = **23,184 rows** (3.9% of the grid). The dropped rows correspond to the first week of the study period for each zone. The underlying trip data from that week is not lost — it is embedded in the lag values of later rows.

**Export.** After dropping warm-up rows, the lag columns are cast to `int32`. A final null assertion confirms no missing values remain. Borough membership is joined from `taxi_zones_manhattan_queens.csv` and the table is split into three files (see section 4).

## 3. Design decisions

**Why aggregate to zone-hour instead of keeping trip-level data.** Modelling at the trip level would require predicting each individual trip, which is not the goal. The forecast target is *how many taxis are needed in zone X at hour H* — a count. Aggregating first also drastically reduces the data volume (from ~23 million rows to ~576,000), making training and feature computation much faster.

**Why use the weather table as the aggregation grid, not a generated Cartesian product.** Generating a zone x hour grid independently would require knowing the exact set of zone IDs and the exact set of hours. Using the weather table as the grid guarantees consistency: the same 138 zones and the same 4,344 hours appear in both the demand table and the weather table, and the join is guaranteed to succeed with no unmatched rows.

**Why keep zero-trip hours.** Many zone-hours have zero pickups — late-night hours in quiet residential zones, for example. If these rows are dropped, the model never sees "this zone is normally quiet at this time", and averages and lags computed without them overstate demand. Zero-fill is the correct treatment for count data when no event is itself a meaningful observation.

**Why drop warm-up rows instead of filling NaN lags.** Three alternatives exist: fill with 0, fill with the zone median, or use only shorter lags. Filling with 0 is misleading (0 looks like zero demand, not "unknown"). Filling with the median introduces a small bias for the first week. Dropping is the cleanest option: no assumption is made about the unobserved prior week, and the cost (3.9% of rows) is small given the 6-month study window.

## 4. Output files

The notebook writes directly to the final outputs with no intermediate files saved to disk. All outputs land in `data/processed/`:

| File | Rows | Zones | Description |
|---|---|---|---|
| `cleaning_report.md` | — | — | Human-readable table of rows removed per rule, with percentages and the DST note |
| `features.csv` | ~576,288 | 138 | **Final combined feature table** (all 138 zones). Primary model input |
| `features_manhattan.csv` | ~288,144 | 69 | Final feature table for Manhattan zones only |
| `features_queens.csv` | ~288,144 | 69 | Final feature table for Queens zones only |

> Exact row counts depend on the cleaning results and are printed by the notebook when it runs. Feature file rows = 599,472 − 23,184 warm-up rows, split by borough.

**Kaggle.** The three feature files and `cleaning_report.md` will be uploaded to the same private Kaggle dataset used for the raw data: [adithyaajay1210/nyc-taxi-demand-manhattan-queens](https://www.kaggle.com/datasets/adithyaajay1210/nyc-taxi-demand-manhattan-queens). How to download and use the dataset: [data/DATA_GUIDE.md](../DATA_GUIDE.md).

**Column list for the three feature files (`features*.csv`):**

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `PULocationID` | int | TLC zone ID (pickup zone) |
| 2 | `pickup_hour` | datetime | Start of the hour, local NYC time |
| 3 | `trip_count` | int | Number of clean pickups in this zone during this hour |
| 4 | `apparent_temperature` | float | Feels-like temperature in C at the start of the hour |
| 5 | `rain` | float | Rain in mm during the hour |
| 6 | `snowfall` | float | Snowfall in cm during the hour |
| 7 | `precipitation_probability` | int | Probability of precipitation during the hour (%) |
| 8 | `hour` | int | Hour of day (0-23) |
| 9 | `day_of_week` | int | Day of week (0 = Monday, 6 = Sunday) |
| 10 | `month` | int | Month (1-12) |
| 11 | `is_weekend` | bool | True if Saturday or Sunday |
| 12 | `is_holiday` | bool | True if a US federal holiday |
| 13 | `trip_count_lag_1h` | int | `trip_count` 1 hour earlier in the same zone |
| 14 | `trip_count_lag_24h` | int | `trip_count` 24 hours earlier in the same zone |
| 15 | `trip_count_lag_168h` | int | `trip_count` 168 hours earlier in the same zone |
| 16 | `trip_count_roll_3h` | float | Mean of `trip_count` over the 3 hours before this row in the same zone |
| 17 | `trip_count_roll_24h` | float | Mean of `trip_count` over the 24 hours before this row in the same zone |

## 5. Things to know before using the processed data

**The DST double-hour is not corrected.** The row for 2025-11-02 01:00 in every zone has a `trip_count` that is roughly twice as large as the same hour in surrounding weeks. Lag features that reach back to this hour (`trip_count_lag_24h` on 2025-11-03 01:00, and `trip_count_lag_168h` on 2025-11-09 01:00) will also be inflated. Models should be made aware of this, for example by including `is_holiday` or an explicit DST flag as a feature.

**Zero-trip rows are real observations, not missing data.** A `trip_count` of 0 means no clean pickups were recorded for that zone and hour. It is not a gap in the data. The lag and rolling features for these rows are computed from the same zero and non-zero history as every other row.

**The warm-up period covers 2025-09-01 to 2025-09-07 (inclusive).** These rows have been dropped from the feature files because the 168-hour lag cannot be computed without a full prior week of data. The underlying trip data from that week is still used — it is embedded in the lag values of all later rows.

**The three feature files have identical schemas.** `features_manhattan.csv` and `features_queens.csv` are strict subsets of `features.csv` by `PULocationID`. No columns differ between them. They are provided as a convenience so that borough-specific models can be trained without filtering in code.

## 6. Notebook

All the processing code is in [`scripts/data_processing_engineering.ipynb`](../../scripts/data_processing_engineering.ipynb):

| Notebook section | What it does |
|---|---|
| Setup | Imports (`pandas`, `numpy`, `holidays`), path constants for input files and the three output CSVs, directory creation |
| Data Cleaning | Loads raw trips, cross-checks counts against `taxi_collection_summary.csv`, applies four removal rules, flags DST hour, writes `cleaning_report.md`. Trips stay in memory; no intermediate CSV is written |
| Hourly Aggregation & Zero-Fill | Derives `pickup_hour`, counts trips per zone per hour, builds the 599,472-row grid from the weather table, left-joins and zero-fills. Demand table stays in memory |
| Weather Merge | Joins four weather columns onto the in-memory demand table, asserts zero nulls |
| Time Features | Adds `hour`, `day_of_week`, `month`, `is_weekend`, `is_holiday` to the in-memory demand table, validates ranges |
| Lag & Rolling Features | Sorts within each zone, computes five lag/rolling columns, drops warm-up rows, joins borough info, saves `features.csv`, `features_manhattan.csv`, `features_queens.csv` |
