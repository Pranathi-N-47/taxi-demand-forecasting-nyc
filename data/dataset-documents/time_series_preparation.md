# Time Series Data Preparation

This document covers the data preparation for the time series analysis stage (Review 2). The dataset is **built on `data/processed/features.csv`**, the cleaned zone-hour table from the processing stage. The document explains what was added to it and why, every step and decision, every output file and field, the checks that were run, and what the readiness diagnostics suggest for the analysis.

- Code: [`scripts/time_series_data_preparation.ipynb`](../../scripts/time_series_data_preparation.ipynb). Run all cells, about 2 minutes. Everything it needs is in the repository (no Kaggle download).
- Outputs: [`data/processed/time_series/`](../processed/time_series/)
- Related: [data_collection.md](data_collection.md) (how the raw data was collected), [data_processing.md](data_processing.md) (cleaning and the ML feature table), [EDA findings](../../EDA_doc/EDA_findings.md)

## 1. Purpose

The next stage forecasts and analyses taxi demand as **time series**: decomposition, stationarity and autocorrelation analysis, and models such as seasonal ARIMA, dynamic harmonic regression, exponential smoothing and Prophet. Those methods have stricter requirements than the machine-learning models:

1. **An unbroken, regular time index.** Every hour present exactly once, no gaps, no repeats.
2. **The full history.** Every hour of data available for fitting.
3. **Series at the right level.** City and borough series are smooth and easy to model. Individual zones range from very busy to almost empty.
4. **Known-in-advance regressors.** Calendar, holidays, events and seasonal terms that are available at forecast time.
5. **A time-based evaluation design.** Fixed train/validation/test periods and rolling-origin folds.

`data/processed/features.csv` is the cleaned zone-hour demand table and the base of this dataset. It was built for the regression models, so on its own it does not meet these needs:

| Requirement | `features.csv` | Time series dataset |
|---|---|---|
| Full history | Starts 8 Sep: the first week was dropped as warm-up for the 168 h lag | Starts 1 Sep, all 4,344 hours. The first week is recovered exactly from `features.csv`'s own `trip_count_lag_168h` |
| Levels | Zone-hour only | City, borough, service area and zone; hourly and daily |
| Daylight-saving double hour | Described in docs only | Flagged, plus an adjusted target and flags for the affected lags |
| Calendar effects | One `is_holiday` flag | Holiday names, holiday periods, days to/from holidays, special days, cross-midnight night events, weekend nights, heavy-snow days |
| Seasonality | Raw `hour`, `day_of_week` | Daily and weekly Fourier terms, time bands, trend index |
| Evaluation design | Split later in the modelling notebook | Train/validation/test built in, with the **same test period as the ML models**, plus 6 rolling-origin folds |
| Per-zone behaviour | None | Volume tier and demand pattern per zone, computed on training data only |
| Safe use of columns | Not specified | Every column has a role (target, known-in-advance regressor, needs-forecast regressor, trend, descriptive), checked for leakage, extrapolation and collinearity (section 6) |

The time series dataset doesn't replace `features.csv`; it extends it. From 8 Sep onwards the demand values are taken from `features.csv` unchanged, so both files describe **exactly the same demand**. The ML models and the time series models therefore work from one source.

## 2. Inputs

| File | Used for |
|---|---|
| `data/processed/features.csv` | **The base:** cleaned zone-hour demand (8 Sep onwards), its 168 h lag (to recover 1-7 Sep), and the holiday flag (cross-check) |
| `data/raw/weather_hourly_by_zone.csv` | The 138 × 4,344 hour grid, zone weather for all hours, and weather grid cells |
| `data/raw/taxi_zones_manhattan_queens.csv` | Borough, service area and coordinates of each zone |
| `data/processed/cleaning_report.md` | Cross-check: total number of clean trips |
| `models/modeling/split_summary.csv` | The ML models' test period, reused so results can be compared |

All inputs are in the repository. The 3.2 GB raw trip file is **not** needed, because the cleaning was already done by the processing stage.

## 3. Preparation steps

```
features.csv (138 zones × 4,176 hours, from 8 Sep)
   │  Step 1  recover 1-7 Sep exactly from trip_count_lag_168h  → full history from 1 Sep
   ▼
   │  Step 2  full 138 × 4,344 grid, DST adjustment, zone weather
   ▼
zone hourly panel (599,472 rows) ──────────────┐
   │  Step 3  calendar, events, Fourier terms,   │  Step 5  zone daily series,
   │          DST flags, splits, CV folds        │          zone profiles (train only)
   ▼                                             ▼
   Step 4  aggregate series: city, borough, service area (hourly + daily)
   ▼
   Step 6  readiness checks and diagnostics → data/processed/time_series/
```

### Step 1: Load `features.csv` and recover the first week

`features.csv` already contains the cleaned demand: the four cleaning rules (duplicates, dropoff at or before pickup, trips over 6 hours, unknown dropoff zone) were applied by the processing stage, leaving 22,579,793 clean trips, and zero-demand hours were filled in. This dataset uses it as its base and does **not** clean the data again.

The processing stage dropped **1-7 Sep** (168 hours × 138 zones = 23,184 zone-hours) as warm-up for its 168-hour lag. Time series models don't need that warm-up, and the week contains Labor Day, so it is recovered.

The recovery is exact. The processing stage computed `trip_count_lag_168h` on the complete zero-filled grid, sorted by time within each zone, *before* it dropped the warm-up rows. For every row from 8-14 Sep, `trip_count_lag_168h` is therefore the zone's count exactly one week earlier, on 1-7 Sep. The EDA used the same fact in its trip reconciliation (A7). Three checks prove it holds:

| Check | Result |
|---|---|
| On every row where both are in the file (15 Sep onwards), `trip_count_lag_168h` equals `trip_count` 168 hours earlier | 553,104 zone-hours, all equal |
| The recovered 7 Sep counts agree with two other lag columns: `trip_count_lag_24h` of each 8 Sep hour, and `trip_count_lag_1h` of 8 Sep 00:00 | 3,312 + 138 values, all equal |
| The total of the full panel equals the clean trips in `cleaning_report.md` | 22,579,793, equal |

### Step 2: Zone-hour panel

- The recovered week and `features.csv` are stacked into one grid of **138 zones × 4,344 hours = 599,472 rows**, checked against the weather table. Zero-demand hours stay **0**, as in `features.csv`: they are real observations, and a time series can't have holes.
- Weather comes from `weather_hourly_by_zone.csv` for all hours, because `features.csv` has no rows for the first week. From 8 Sep it is identical to the weather columns of `features.csv` (checked on all 576,288 zone-hours).

**Daylight saving time.** TLC timestamps are local wall-clock times with no offset. On 2 Nov 2025 the clocks went back at 02:00, so 01:00-01:59 happened twice, and both physical hours are recorded under one `01:00` label. Two options were considered:

- **Switch to UTC.** This would give 4,345 physical hours, but the trips in the merged label can't be split between the two physical hours, so the extra hour could only be filled with a guess.
- **Keep local wall-clock time and adjust that one value (chosen).** The index stays perfectly regular (4,344 consecutive labels), "hour of day" keeps its everyday meaning, and the dataset carries:
  - `trip_count`: the observed count, unchanged
  - `trip_count_adj`: the same, except the merged hour is divided by 2, the number of physical hours it covers, so every value means **pickups per physical hour**
  - flags for the merged hour and for the hours whose 24 h and 168 h lags read it (step 3)

City-wide, the merged hour has 9,166 trips against 8,735 at 00:00 and 3,423 at 02:00. The adjusted value of 4,583 fits smoothly between them. **Fit models on `trip_count_adj`.** The test period has no DST hour, so `trip_count_adj` and `trip_count` are identical there, and evaluation is unaffected.

### Step 3: Calendar and event table

One row per hour holds every time-related regressor (`ts_calendar_hourly.csv`). Everything except `is_heavy_snow_day` is **known in advance** (calendar, published holidays, scheduled events), so it can be used when forecasting without leakage. The event definitions are the same ones the EDA used (`scripts/eda_arun.py`), so both stages talk about the same events.

- **Trend and position.** `t` (hours since start), `day_index`, `week_index` (1 Sep 2025 is a Monday, so weeks run Monday to Sunday), and `hour_of_week` (0-167), a grouping key for weekly profiles.
- **Calendar.** Hour, weekday, weekend, month, day of month, and the EDA's six time bands.
- **Weekend nights.** Friday and Saturday 20:00-04:59. The hours after midnight belong to the evening before, as in the EDA.
- **Federal holidays with names.** The EDA found holidays differ a lot: Christmas −58%, Thanksgiving −45%, Columbus Day −8% and MLK Day about 0%, each against comparable normal days. One shared flag can't capture that. All 8 holidays in the window are kept, including Labor Day, which `features.csv` lost with its warm-up week. Days to the next and since the previous holiday are counted across holidays outside the window too (4 July 2025, Memorial Day 2026), so the edge values are correct. Both are capped at 14 days (see section 6).
- **Holiday periods.** Thanksgiving week (24-30 Nov) and Christmas-New Year (20 Dec - 4 Jan). The EDA measured −24% and −27%.
- **Special days.** Halloween, NYC Marathon (2 Nov, also the DST day), Black Friday, Christmas Eve, New Year's Eve.
- **Night events.** Halloween night and New Year's Eve night, 20:00-04:59 across midnight. A calendar-day flag would cut these in two. The EDA measured +62% for New Year's Eve night.
- **DST flags.** The merged hour, plus the two hours whose 24 h and 168 h lags point at it.
- **Heavy-snow days.** The median zone snowfall for the day is at least 5 cm, the EDA's rule. Five days qualify: 14 Dec, 26 Dec, **25 Jan** (about 19 cm, the biggest storm in the data), **22 Feb** and **23 Feb**. This flag is derived from the weather data, so at forecast time it is only known through a weather forecast.
- **Fourier terms.** $\sin(2\pi k h/P)$ and $\cos(2\pi k h/P)$ for the daily cycle ($P = 24$) and the weekly cycle ($P = 168$), with $k = 1, 2, 3$: 12 columns. These are the standard way to give ARIMA (dynamic harmonic regression), regression and Prophet-style models a long seasonal period. A seasonal ARIMA with a 168-hour season is impractical to fit. They also work as cyclical encodings, so 23:00 sits next to 00:00. With only six months of data there is no yearly seasonality to model.
- **Splits.** See section 5.

### Step 4: Aggregate series

Six series at three levels are built by summing the zone panel:

| Level | `series_id` | Zones | Clean trips | Mean pickups/hour | Max |
|---|---|---:|---:|---:|---:|
| city | All zones | 138 | 22,579,793 | 5,196.9 | 12,648 |
| borough | Manhattan | 69 | 20,431,943 | 4,702.5 | 11,843 |
| borough | Queens | 69 | 2,147,850 | 494.4 | 1,449 |
| service_area | Yellow Zone | 55 | 19,472,673 | 4,481.7 | 11,277 |
| service_area | Boro Zone | 81 | 1,639,075 | 377.3 | 1,172 |
| service_area | Airports | 2 | 1,468,045 | 337.9 | 1,058 |

Each series and hour has the observed and adjusted counts, $\log(1+y)$ of the adjusted count, the number of zones, the **mean weather across its zones**, and all calendar columns.

- **Why a plain mean for weather.** Weighting by each zone's pickups in that hour would put the target inside a regressor.
- **Why $\log(1+y)$.** The EDA found demand strongly right-skewed and overdispersed. A log evens out the variance (ARIMA and ETS assume roughly constant variance), and $\log(1+y)$ keeps zero hours valid. Forecasts made on the log scale need back-transforming with $e^{\hat{y}}-1$, which estimates the median rather than the mean. Check this on the validation period.

Daily versions of the six series have daily weather summaries (temperature mean/min/max, rain and snow totals, precipitation probability mean/max, hours with rain or snow), the day-level calendar fields and 7-day Fourier terms.

### Step 5: Zone-level series and zone profiles

- **Zone hourly panel.** 599,472 rows, kept narrow (counts plus the zone's own weather). Calendar fields are the same for every zone, so they stay in the calendar table and are joined on `ds`. This keeps the file at about 31 MB, within GitHub's limits.
- **Zone daily series.** 138 zones × 181 days.
- **Zone profiles.** The 138 series behave very differently, so each zone gets a profile. **It uses the training period only**, so the test period can't influence any modelling choice:
  - **Volume tier** by mean pickups per hour: High ≥ 50, Medium 10-50, Low 1-10, Sparse > 0 and < 1, Zero.
  - **Demand pattern** (Syntetos-Boylan classification):
    - **ADI** = hours ÷ hours with at least one pickup
    - **CV²** = squared coefficient of variation of the non-zero counts
    - *smooth*: ADI < 1.32 and CV² < 0.49
    - *erratic*: ADI < 1.32 and CV² ≥ 0.49
    - *intermittent*: ADI ≥ 1.32 and CV² < 0.49
    - *lumpy*: ADI ≥ 1.32 and CV² ≥ 0.49
  - **Suggested approach** derived from the pattern.
  - **Weather cell.** The 138 zones get their weather from 50 Open-Meteo grid cells, so zones in one cell are not independent weather observations.

| Volume tier | smooth | erratic | intermittent | lumpy | no demand | Total |
|---|---:|---:|---:|---:|---:|---:|
| High | 20 | 19 | 0 | 0 | 0 | 39 |
| Medium | 17 | 7 | 0 | 0 | 0 | 24 |
| Low | 11 | 9 | 5 | 9 | 0 | 34 |
| Sparse | 0 | 0 | 38 | 1 | 0 | 39 |
| Zero | 0 | 0 | 0 | 0 | 2 | 2 |

Result: **83 zones** can be modelled as individual series, **53** need an intermittent-demand method (Croston/TSB) or pooling into their borough or service-area series, and **2** (zones 103 and 104, the Governor's/Ellis/Liberty Island zones) have no demand and only appear at aggregate level.

For hourly data, a high CV² in busy zones mostly reflects the strong daily cycle, not randomness. For those zones "erratic" means "needs seasonal terms". The pattern matters most for its ADI part, which separates regular demand from intermittent demand.

## 4. Why no lag features in this dataset?

`features.csv` already has lag and rolling features for the ML models. Time series models build their own dependence on the past: ARIMA through its AR/MA terms and differencing, ETS through its smoothing states. Fixed lag columns would duplicate that, and dropping the warm-up rows they need would break the continuous history. Anyone who needs lags for a regression-style forecaster can compute them from the regular series with `groupby(...).shift(k)`. The `is_dst_lag24_affected` and `is_dst_lag168_affected` flags show where such lags read the DST hour.

## 5. Evaluation design

| Split | Hourly period | Hours | Daily period | Days |
|---|---|---:|---|---:|
| train | 2025-09-01 00:00 - 2026-01-11 03:00 | 3,172 | 2025-09-01 - 2026-01-10 | 132 |
| validation | 2026-01-11 04:00 - 2026-01-25 03:00 | 336 | 2026-01-11 - 2026-01-24 | 14 |
| **test** | **2026-01-25 04:00 - 2026-02-28 23:00** | **836** | 2026-01-25 - 2026-02-28 | 35 |

- **Test** is read from `models/modeling/split_summary.csv`. It is exactly the window the naive 168 h baseline, Linear Regression, Random Forest and HistGradientBoosting were scored on. At zone level, the test rows of `ts_zone_hourly.csv` are the same 138 × 836 = 115,368 zone-hours as the ML test set, so MAE, RMSE and R² can be compared directly.
- **Validation** is the two weeks just before test, for choosing orders and settings without touching test.
- **Train** starts on 1 Sep. The ML training set started on 8 Sep, but time series models have no lag warm-up to drop.
- **Rolling-origin folds** (`ts_cv_folds.csv`): six folds that each forecast 7 days (168 hours), with weekly cut-offs from 14 Dec to 24 Jan. Each fold trains on everything before its window. Use them to check that a model's accuracy holds up across several weeks, including the holiday period, rather than on one window.

**Watch out:** the test period starts with the biggest snowstorm in the data (25 Jan) and contains two more heavy-snow days (22-23 Feb). Demand on those days drops far below normal (see figure 1). Report test errors for the whole period and also without those days, so a single storm doesn't dominate the comparison.

## 6. Checks and feature review

### 6.1 Data checks

Every check is recorded in [`ts_readiness_checks.csv`](../processed/time_series/ts_readiness_checks.csv). **All 41 passed.** The main ones:

| Check | Result |
|---|---|
| `trip_count_lag_168h` is exact wherever testable | 553,104 zone-hours, all equal |
| First week recovered and consistent with the 24 h and 1 h lags | 23,184 zone-hours recovered; 3,450 cross-checked values equal |
| Total equals `cleaning_report.md` | 22,579,793 clean trips |
| Weather equals `features.csv`'s weather columns | 576,288 zone-hours, identical |
| Holiday flag matches `features.csv` | 4,176 hours compared, identical |
| Complete, regular grid | 138 × 4,344 rows; every step within a zone is exactly 1 hour |
| Every clean trip counted once | 22,579,793 in the zone panel, in every level's total, and in the daily totals |
| Same values as an earlier build from the raw trip file | All counts identical (the series were first built from the raw trips with the same cleaning, then rebuilt on `features.csv`) |
| No missing values | In all five data tables, checked again after re-reading every saved CSV the way a user would |
| Zone weather equals the source table | 599,472 rows identical |
| Test period equals the ML models' | 836 hours, 25 Jan 04:00 - 28 Feb 23:00 |
| Zone profiles use training data only | Last hour used: 2026-01-11 03:00 |

Separately from the notebook, an independent script recomputed the tables from the saved files: the city, borough and service-area series rebuilt from the zone panel, the daily sums, the Fourier terms, the holidays against the official calendar, the splits and the zone profiles. All 37 of its checks passed.

### 6.2 Is every feature safe and suitable for time series analysis?

A feature can be correct and still be wrong to use in a forecast. Every column was reviewed against four questions:
1. Does it hide the answer?
2. Is it known at forecast time?
3. Does the test period take values that training never saw?
4. Is it an exact copy of other columns?

The review found five problems, all fixed before release:

| Feature | Problem | Fix |
|---|---|---|
| `active_zones` (zones with a pickup in the hour) | Built from the **same hour's demand** (correlation 0.47 with the target). As an input it would leak the answer | Removed |
| `days_to_next_holiday` | Reached **97 days** in the test period but never above 41 in training, so models would extrapolate | Capped at 14 days, along with `days_since_prev_holiday` |
| `iso_week` | Jumps from 52 back to 1 at New Year (misleading as a number) and duplicates `week_index` | Removed |
| `hour_of_week` | Exactly `24 × day_of_week + hour`, so the three together are **perfectly collinear** and break regression and ARIMA-with-regressors fits | Kept as a grouping key, role `descriptive` |
| `month`, `holiday_name` | February, MLK Day and Washington's Birthday appear **only in validation/test**, so a model can never learn them | Kept for analysis, role `descriptive`; model holidays with `is_holiday` |

To make correct use easy, every column has a **role** in [`ts_feature_catalog.csv`](../processed/time_series/ts_feature_catalog.csv), and the roles are repeated in the field tables in section 9:

| Role | Columns | Use |
|---|---:|---|
| `target` | 1 | `trip_count_adj`, the value to fit and forecast |
| `target_observed` | 1 | `trip_count`, the observed count, for evaluation. Never an input |
| `target_transform` | 1 | `log1p_trip_count_adj`. Never an input |
| `regressor_known_in_advance` | 29 | Calendar, holiday, event, DST and Fourier inputs. Safe to use when forecasting |
| `regressor_needs_forecast` | 14 | Weather inputs. Using observed weather assumes a perfect weather forecast |
| `trend` | 3 | `t`, `day_index`, `week_index`. Extrapolate by design; use with care over six months |
| `descriptive` | 9 | Grouping and plots only, not model inputs |
| `identifier`, `split` | 7 | Keys and split labels |

The notebook enforces these rules with five checks:

| Rule | Result |
|---|---|
| Every column has exactly one role | 65 columns, none missing |
| Known-in-advance regressors carry no demand information: identical across all 6 series at every timestamp | 28 hourly and 16 daily regressors pass |
| Known-in-advance regressors stay within their training range in validation and test; no unseen categories | None outside |
| Known-in-advance regressors are not exactly collinear | 27 numeric regressors + intercept: rank 28 of 28 |
| Fourier terms are not collinear | Rank 13 of 13, condition number 1.4 |

**Model test.** To confirm the files work with standard tools, the following were fitted on the training data and forecast over validation and test. All loaded with a regular frequency (`asfreq`) and ran without errors, and every forecast was finite:
- a daily SARIMA(1,0,1)(0,1,1)₇ with holiday regressors
- an hourly ARIMA with Fourier and calendar regressors
- an MSTL decomposition with periods 24 and 168
- the zone panel joined to the calendar (599,472 rows, no missing values)

This only checks that the data is usable. Choosing and comparing models is the analysis stage's work.

**Weather is the one input that goes beyond its training range.** The test period is colder than anything in training: the lowest feels-like temperature is −23.8 °C, against −14.2 °C in training. It also contains the biggest snowstorm in the data. That is real weather, not a data error. Models that use weather will be extrapolating there, which is another reason to report results with and without the storm days.

## 7. Readiness diagnostics and starting points for the analysis

These diagnostics use the training period only.

**Stationarity** (`ts_stationarity_tests.csv`). ADF (null: non-stationary) and KPSS (null: stationary) were run on the `All zones` log series:

| Series | Transformation | ADF p | KPSS p | Stationary by both |
|---|---|---:|---:|---|
| hourly | log level | 0.000 | 0.018 | No |
| hourly | seasonal difference, 168 h | 0.001 | ≥ 0.10 | **Yes** |
| hourly | seasonal 168 h + first difference | 0.000 | ≥ 0.10 | Yes |
| daily | log level | 0.016 | 0.081 | Yes (borderline) |
| daily | seasonal difference, 7 days | 0.004 | ≥ 0.10 | **Yes** |
| daily | seasonal 7 days + first difference | 0.000 | ≥ 0.10 | Yes |

**Seasonal periods** (`ts_acf_key_lags.csv`). The autocorrelation of the hourly log series is 0.93 at lag 1, 0.88 at lag 24 and **0.89 at lag 168**, the highest of the seasonal lags. It stays at 0.83 at lag 336 (figure 3). The weekly cycle is the main one, with the daily cycle inside it (figure 2).

**Suggested starting points:**
- **Hourly:** one seasonal difference at 168 hours (D = 1) is enough, and an extra first difference is not needed (d = 0). A seasonal ARIMA with a 168-hour season is impractical, so these work better:
  - ARIMA errors with the daily and weekly Fourier terms plus calendar and event dummies (dynamic harmonic regression)
  - MSTL decomposition with periods 24 and 168
  - TBATS or Prophet, with the holiday and event columns as regressors
- **Daily:** seasonal ARIMA with a weekly season (m = 7, D = 1) or ETS with weekly seasonality, with holiday and snow dummies.
- **Baseline:** the seasonal naive forecast (same hour last week) is the natural baseline, and it is the ML stage's 168-hour baseline (test MAE 9.31). Compare against it on the same test rows.
- **Zones:** model the 83 regular zones individually. Use Croston/TSB for intermittent zones, or forecast them through their aggregate series.

These are starting points from the diagnostics, not results. Order selection and model comparison belong to the analysis stage.

**Weather as a regressor.** The weather columns are the observed weather for the hour (from the Open-Meteo historical forecast archive). Using them to forecast assumes the weather is known in advance, the "perfect forecast" case. In the ML stage, removing weather barely changed HistGradientBoosting's test error (`models/modeling/weather_ablation_results.csv`). Test weather as an optional regressor and report both versions.

## 8. Figures

**Figure 1.** City-wide hourly pickups, daily mean, validation and test periods, holidays and heavy-snow days. The Christmas-New Year dip and the storm days in the test period stand out.

![All zones series](figures/time_series/ts_all_zones_series.png)

**Figure 2.** Average weekly profile per borough (training period). Five weekday cycles with an evening peak, different weekend shapes, and Queens highest on Sunday evening, probably because of the two airports, which are in Queens.

![Weekly profile](figures/time_series/ts_weekly_profile.png)

**Figure 3.** Autocorrelation of the hourly log series up to two weeks. Peaks every 24 hours, highest at 168 and 336.

![ACF](figures/time_series/ts_acf.png)

## 9. Output files and fields

All files are in `data/processed/time_series/`, as CSV, with timestamps as `YYYY-MM-DD HH:MM:SS` in naive local New York time. Booleans are `True`/`False`. On ordinary days the event-name columns (`holiday_name`, `holiday_period`, `special_day_name`, `night_event_name`) hold the value `none`, not a blank, so they don't load as missing values. The only empty cells in the whole dataset are `adi` and `cv2` for the two zones with no demand (103, 104), where they are undefined.

| File | Rows × columns | Size | Contents |
|---|---|---:|---|
| `ts_aggregate_hourly.csv` | 26,064 × 54 | 9.3 MB | 6 series × 4,344 hours, with all calendar fields |
| `ts_aggregate_daily.csv` | 1,086 × 42 | 0.3 MB | 6 series × 181 days |
| `ts_zone_hourly.csv` | 599,472 × 8 | 28.0 MB | 138 zones × 4,344 hours (join the calendar on `ds`) |
| `ts_zone_daily.csv` | 24,978 × 14 | 1.9 MB | 138 zones × 181 days |
| `ts_calendar_hourly.csv` | 4,344 × 44 | 1.3 MB | Calendar, events, Fourier terms, DST flags, splits |
| `ts_zone_profile.csv` | 138 × 20 | < 0.1 MB | Zone profiles (training period) |
| `ts_cv_folds.csv` | 6 × 7 | < 0.1 MB | Rolling-origin folds |
| `ts_stationarity_tests.csv` | 6 × 6 | < 0.1 MB | ADF/KPSS results |
| `ts_acf_key_lags.csv` | 11 × 2 | < 0.1 MB | Autocorrelation at key lags |
| `ts_feature_catalog.csv` | 65 × 4 | < 0.1 MB | Role and description of every column |
| `ts_readiness_checks.csv` | 41 × 3 | < 0.1 MB | Every check and its result |

### 9.1 Count and weather fields (aggregate, zone and daily tables)

| Field | Type | Role | Description | Purpose |
|---|---|---|---|---|
| `series_level` | text | identifier | `city`, `borough` or `service_area` (aggregate tables) | Pick a level |
| `series_id` | text | identifier | `All zones`, `Manhattan`, `Queens`, `Yellow Zone`, `Boro Zone`, `Airports` | Identifies one series |
| `PULocationID` | int | identifier | TLC pickup zone (zone tables) | Identifies one zone series |
| `ds` | datetime | identifier | Start of the hour, local time (hourly tables) | Time index, the usual name in forecasting libraries |
| `date` | date | identifier | Calendar day (daily tables, calendar) | Daily time index |
| `trip_count_adj` | float | **target** | `trip_count` with the DST merged hour halved | **Fit and forecast this**: pickups per physical hour |
| `trip_count` | int | target_observed | Clean pickups in the hour or day (from `features.csv`; 1-7 Sep recovered from its 168 h lag) | Observed demand; for evaluation, never an input |
| `log1p_trip_count_adj` | float | target_transform | $\log(1 + \text{trip\_count\_adj})$ (aggregate tables) | Variance-stabilised target; never an input |
| `n_zones` | int | descriptive | Zones in the series (constant per series) | Scale and normalising |
| `apparent_temperature` | float | needs forecast | Feels-like temperature at the start of the hour, °C (series: mean over zones) | Weather regressor |
| `rain` | float | needs forecast | Rain during the hour, mm (series: mean over zones) | Weather regressor |
| `snowfall` | float | needs forecast | Snowfall during the hour, cm (series: mean over zones) | Weather regressor |
| `precipitation_probability` | float/int | needs forecast | Chance of precipitation in the hour, % (series: mean over zones) | Weather regressor |
| `apparent_temperature_mean`, `_min`, `_max` | float | needs forecast | Daily mean, minimum and maximum of hourly apparent temperature, °C | Daily weather regressors |
| `rain_total`, `snowfall_total` | float | needs forecast | Daily rain (mm) and snow (cm) totals | Daily weather regressors |
| `precipitation_probability_mean`, `_max` | float | needs forecast | Daily mean and maximum precipitation probability, % | Daily weather regressors |
| `rain_hours`, `snow_hours` | int | needs forecast | Hours in the day with rain or snow above 0 | How long the bad weather lasted |
| `split` | text | split | `train`, `validation` or `test` | Evaluation design (section 5) |

### 9.2 Calendar and event fields (`ts_calendar_hourly.csv`; also in the aggregate tables)

"Known" means the role `regressor_known_in_advance`.

| Field | Type | Role | Description | Purpose |
|---|---|---|---|---|
| `t` | int | trend | Hours since 1 Sep 2025 00:00 (0-4,343) | Trend term; extrapolates by design |
| `day_index`, `week_index` | int | trend | Days and Monday-start weeks since 1 Sep 2025 | Trend and grouping |
| `hour`, `day_of_week` | int | known | Hour (0-23), weekday (0 = Monday) | Daily and weekly cycles; best used as categories or through the Fourier terms |
| `day_name` | text | descriptive | Weekday name | Readable label for `day_of_week` |
| `is_weekend` | bool | known | Saturday or Sunday | Weekday vs weekend demand |
| `month`, `day_of_month` | int | descriptive | Calendar position | Grouping; February occurs only in the test period, so `month` is not an input |
| `hour_of_week` | int | descriptive | `day_of_week × 24 + hour` (0-167) | Key for weekly profiles and seasonal-naive forecasts; exactly collinear with `hour` and `day_of_week` |
| `time_band` | text | known | EDA bands: Late night 00-04, Early morning 05-06, AM 07-09, Midday 10-15, PM 16-19, Evening 20-23 | Grouped time-of-day effects (as categories) |
| `is_weekend_night` | bool | known | Friday or Saturday 20:00-04:59 | Nightlife demand (EDA I5) |
| `is_holiday` | bool | known | US federal holiday | Holiday effect (EDA E1) |
| `holiday_name` | text | descriptive | Holiday name, `none` otherwise | Analyse individual holidays; MLK Day and Washington's Birthday occur only in validation/test |
| `is_day_before_holiday`, `is_day_after_holiday` | bool | known | Day before or after a federal holiday | Lead and lag effects |
| `days_to_next_holiday`, `days_since_prev_holiday` | int | known | Days to the next and since the last federal holiday (0 on the holiday), **capped at 14** | Distance-to-holiday effects without extrapolation |
| `is_holiday_period` | bool | known | Thanksgiving week or Christmas-New Year | Period effects (EDA E2) |
| `holiday_period` | text | descriptive | Period name, `none` otherwise | Label for `is_holiday_period` |
| `is_special_day` | bool | known | Halloween, NYC Marathon, Black Friday, Christmas Eve, New Year's Eve | Event effects (EDA E4) |
| `special_day_name` | text | descriptive | Special day name, `none` otherwise | Label for `is_special_day` |
| `is_night_event` | bool | known | Halloween night, New Year's Eve night (20:00-04:59) | Cross-midnight events (EDA E3) |
| `night_event_name` | text | descriptive | Night event name, `none` otherwise | Label for `is_night_event` |
| `is_dst_merged_hour` | bool | known | 2025-11-02 01:00, which holds two physical hours | Mark the DST artefact |
| `is_dst_lag24_affected`, `is_dst_lag168_affected` | bool | known | Hours whose 24 h and 168 h lags read the merged hour | Mark affected lags (EDA A8) |
| `is_heavy_snow_day` | bool | needs forecast | Zone-median daily snowfall ≥ 5 cm (derived from weather) | Storm effects |
| `fourier_daily_sin_k`, `fourier_daily_cos_k` | float | known | $\sin/\cos(2\pi k\,\text{hour}/24)$, k = 1-3 | Daily seasonality regressors |
| `fourier_weekly_sin_k`, `fourier_weekly_cos_k` | float | known | Hourly tables: $\sin/\cos(2\pi k\,\text{hour\_of\_week}/168)$. Daily tables: $\sin/\cos(2\pi k\,\text{day\_of\_week}/7)$. k = 1-3 | Weekly seasonality regressors |
| `split`, `split_daily` | text | split | Hourly split, and the split of the calendar day (daily tables have this as `split`) | Evaluation design |
| `contains_dst_merged_hour` | bool | known | 2 Nov 2025, whose `trip_count` includes the extra hour (daily tables) | Mark the DST day |

### 9.3 Zone profile fields (`ts_zone_profile.csv`)

| Field | Description |
|---|---|
| `PULocationID`, `Borough`, `Zone`, `service_zone`, `latitude`, `longitude` | Zone identity and centre point |
| `train_hours`, `train_total_trips` | Hours and pickups in the training period |
| `train_mean`, `train_median`, `train_std`, `train_p95`, `train_max` | Hourly demand statistics, training period |
| `train_zero_share` | Share of training hours with no pickups |
| `adi`, `cv2` | Average demand interval and squared coefficient of variation of non-zero demand. Empty (undefined) for zones 103 and 104, which have no pickups |
| `volume_tier` | High, Medium, Low, Sparse or Zero |
| `demand_pattern` | smooth, erratic, intermittent, lumpy or no demand |
| `suggested_approach` | individual series model, intermittent-demand method or pool to aggregate, or aggregate level only |
| `weather_cell_id` | Open-Meteo grid cell supplying the zone's weather (50 cells) |

### 9.4 Fold fields (`ts_cv_folds.csv`)

`fold`, `train_start`, `train_end`, `forecast_start`, `forecast_end`, `forecast_hours` (168), `forecast_days` (7).

## 10. How to use it

```python
import pandas as pd

TS = "data/processed/time_series/"

# one city or borough series, hourly, with all regressors
agg = pd.read_csv(TS + "ts_aggregate_hourly.csv", parse_dates=["ds", "date"])
manhattan = agg[agg["series_id"] == "Manhattan"].set_index("ds").asfreq("h")
train = manhattan[manhattan["split"] == "train"]

# zone panel with the calendar joined
zone = pd.read_csv(TS + "ts_zone_hourly.csv", parse_dates=["ds"])
cal = pd.read_csv(TS + "ts_calendar_hourly.csv", parse_dates=["ds", "date"])
zone = zone.merge(cal, on="ds", how="left")

# the safe model inputs, straight from the catalogue
catalog = pd.read_csv(TS + "ts_feature_catalog.csv")
known_inputs = catalog.loc[catalog["role"] == "regressor_known_in_advance", "column"]
X_train = train[[c for c in known_inputs if c in train.columns and train[c].dtype != object]]

# zones that can be modelled individually
profile = pd.read_csv(TS + "ts_zone_profile.csv")
regular_zones = profile.loc[profile["suggested_approach"] == "individual series model", "PULocationID"]
```

`.asfreq("h")` works directly because the index has no gaps or repeats.

## 11. Things to know

- **Fit on `trip_count_adj`, evaluate on the test split.** There is no DST hour in the test period, so evaluation is unaffected.
- **The test period contains three heavy-snow days**, including the biggest storm in the data on its first day. Report errors with and without them.
- **Holidays are single events.** Each federal holiday appears once in the data, so holiday-specific effects can't be validated on another year. Treat them as dummies, not as learned patterns.
- **Six months of data.** Daily and weekly seasonality are well supported. Yearly seasonality and long-term trends can't be estimated.
- **Zone weather comes from 50 grid cells.** Zones sharing a cell have the same or very similar weather.
- **Observed pickups measure completed trips, not unmet demand.**
- **Reproduce:** install `requirements.txt` and run the notebook. All its inputs are in the repository, so no Kaggle download is needed. All outputs are overwritten and every check is re-run.
- **Yellow and green are combined**, as in `features.csv`. A per-fleet split would need the raw trip file.
