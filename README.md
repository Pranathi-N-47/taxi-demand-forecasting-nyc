# Weather-Aware Taxi Demand Prediction and Forecasting in New York City

How many taxi pickups will each zone see in the next hour, day or week, and what
should a fleet operator do about it? This project answers that for **138 pickup
zones in Manhattan and Queens**, using six months of Yellow and Green Taxi trips,
local weather and calendar data.

| Stage | What we built | Headline result |
|---|---|---|
| Data | Clean zone-hour dataset with weather and calendar features | 23.3 M trips → 576,288 zone-hours |
| EDA | Where, when and why demand changes | 16 of 138 zones carry half of all trips |
| Review 1 · Regression | Next-hour demand prediction | HistGradientBoosting, **R² = 0.963**, 38% less error than baseline |
| Review 2 · Time series | Day-ahead and week-ahead forecasts | **16.5% less error** than repeating last week |
| Review 2 · Association rules | If-then rules linking conditions to demand levels | **152 rules** that hold up on unseen data |
| Dashboard | Interactive Streamlit app with the models running live | Story of the project + three planning tools |

## Business impact

What a taxi operator can do differently because of this work:

| Action | Evidence |
|---|---|
| **Concentrate the fleet** in core Manhattan / Yellow-Zone areas | Manhattan produces 90.6% of pickups; 55 Yellow-Zone areas capture 86.3% of volume |
| **Stage cabs before airport peaks** (LaGuardia 14:00, JFK 16:00) | Two airport zones produce 6.5% of all pickups |
| **React to the weather** | Morning rain adds 15–22% demand; heavy snow cuts it by 40–70% |
| **Plan shifts and maintenance 1–7 days ahead** | Strong daily and weekly cycles; Model 1 forecasts beat "same as last week" by 16.5% |
| **Pre-position for the weekday morning rush** in commuter zones | Rule confidence 70–73%, 2.6–2.7× more likely than usual |
| **Follow the nightlife** from 20:00, longer on Friday and Saturday | Rule confidence 58–63%, 2.2–2.3× more likely than usual |

In short: less empty cruising, shorter passenger waits, and planning ahead
instead of reacting.

## Project scope

- **Geography:** Manhattan and Queens, 69 zones each
- **Taxi types:** Yellow and Green Taxi
- **Study period:** September 2025 to February 2026
- **Target:** hourly pickup count per zone (`trip_count`)
- **Split:** chronological, so the models never see the future
  - Training: `2025-09-08 00:00` to `2026-01-25 03:00`
  - Testing: `2026-01-25 04:00` to `2026-02-28 23:00`

The raw trip data (23,318,979 trips, about 3.2 GB) is not stored in GitHub.
Get it from the private
[Kaggle dataset](https://www.kaggle.com/datasets/adithyaajay1210/nyc-taxi-demand-manhattan-queens)
or rebuild it with the data-collection notebook.

## Data sources

| Source | Used for |
|---|---|
| [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Pickup times and taxi zones |
| [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) | Hourly feels-like temperature, rain, snowfall, precipitation probability |

See [data/DATA_GUIDE.md](data/DATA_GUIDE.md) for file layouts, field
descriptions and known data issues.

## Setup

```bash
pip install -r requirements.txt                                   # data, EDA, Review 1 models
pip install -r requirements.txt -r requirements-time-series.txt   # adds Review 2 time series
```

Open the notebooks in VS Code with the Jupyter extension and pick the same
Python environment as the kernel. Run everything from the repository root.

## Reproduce the work

### Review 1: data, EDA and regression models

1. **Collect data** – [scripts/data_collection.ipynb](scripts/data_collection.ipynb)
   downloads the TLC files and zone-level weather (about 425 MB, needs internet).
2. **Clean and build features** – [scripts/data_processing_engineering.ipynb](scripts/data_processing_engineering.ipynb)
   writes `data/processed/features.csv` (plus Manhattan and Queens versions and a cleaning report).
3. **Explore** – `python scripts/eda_arun.py` then `python scripts/validate_eda_arun.py`
   (or [scripts/EDA_Arun.ipynb](scripts/EDA_Arun.ipynb)). Findings are in
   [EDA_doc/EDA_findings.md](EDA_doc/EDA_findings.md).
4. **Split and baseline** – [models/modeling_preparation_baseline.ipynb](models/modeling_preparation_baseline.ipynb)
   creates the 80/20 time split and the "same hour last week" baseline.
5. **Train the models** –
   [Linear Regression](models/model_1_linear_regression.ipynb),
   [Random Forest](models/model_2_random_forest.ipynb),
   [HistGradientBoosting](models/model_3_hist_gradient_boosting.ipynb).
   All use zone, hour, weekday, month, holiday, weather, demand lags (1h, 24h, 168h)
   and rolling averages (3h, 24h).
6. **Compare** – [models/model_comparison_error_analysis.ipynb](models/model_comparison_error_analysis.ipynb)
   for errors by hour, zone and residuals. A weather ablation is in
   [models/weather_ablation_histgradientboosting.ipynb](models/weather_ablation_histgradientboosting.ipynb).

### Review 2: time series

7. **Prepare the series** – [scripts/time_series_data_preparation.ipynb](scripts/time_series_data_preparation.ipynb)
   builds hourly and daily series for every zone, borough and service area, plus a
   calendar/event table and rolling-origin folds, in `data/processed/time_series/`.
   Details: [time_series_preparation.md](data/dataset-documents/time_series_preparation.md).
8. **Model 1: seasonal and intermittent-demand forecasting** – in
   [models/time_series/model_1/](models/time_series/model_1/README.md).
   Each zone gets the best of four simple methods (adaptive seasonal, four-week
   mean, weekly naive, TSB for sparse zones), chosen on past data only. Zone
   forecasts are added up for borough and city totals, with calibrated 80% and
   95% ranges.

   ```bash
   python models/time_series/model_1/test_time_series.py
   python models/time_series/model_1/run_time_series.py
   python models/time_series/model_1/validate_time_series.py
   ```

9. **Models 2 and 3: Dynamic Harmonic Regression and MSTL + ETS** – in
   [timeSeries/](timeSeries/README.md). Run `python timeSeries/create_time_series_split.py`
   first (70/15/15 split), then the model notebooks.

### Review 2: association rules

10. **Prepare transactions** – each transaction is one zone in one hour, described
    by time band, day type, holiday, rain, snow, temperature band, zone cluster and
    demand level (Low / Medium / High for that zone).

    ```bash
    python scripts/association_rule_data_preparation.py
    python -m unittest discover -s tests -v
    ```

11. **Mine and validate rules** – [scripts/association_rule_mining.ipynb](scripts/association_rule_mining.ipynb)
    compares Apriori with FP-Growth, re-checks every rule on the test period and
    removes redundant rules. See [ARM_doc/association_rule_mining.md](ARM_doc/association_rule_mining.md).

## Results

### Review 1: next-hour prediction (test period)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline (same hour last week) | 9.31 | 24.06 | 0.878 |
| Linear Regression | 7.49 | 15.70 | 0.948 |
| Random Forest | 6.80 | 15.26 | 0.951 |
| **HistGradientBoosting** | **5.78** | **13.16** | **0.963** |

- The strongest predictors are recent demand (1h, 24h and 168h lags, 24h
  average), location and hour of day.
- Removing weather barely changes the error (MAE 5.67 without vs 5.78 with):
  recent demand already reflects most weather effects. Weather matters mainly
  in extremes such as heavy snow.

### Review 2: day-ahead and week-ahead forecasts

| Series | Baseline MAE (last week) | Model 1 MAE | Improvement |
|---|---:|---:|---:|
| Per zone, day-ahead | 9.31 | 7.82 | 16.0% |
| All zones, day-ahead | 777.7 | 649.4 | 16.5% |
| All zones, week-ahead | 777.7 | 675.6 | 13.1% |
| All zones, daily totals | 14,270 | 12,774 | 10.5% |

- The 80% forecast ranges contain the actual value 83% of the time day-ahead and
  80% week-ahead.
- The heavier models (Dynamic Harmonic Regression, MSTL + ETS) did not beat the
  weekly naive baseline on their own split (test MAE 11.98 and 11.35 vs 8.85).
  Simple seasonal methods won on these noisy zone series.
- These forecasts hide every outcome inside the forecast day or week, while the
  Review 1 models see the previous hour. The two scores answer different
  questions and should not be ranked against each other.

### Review 2: association rules

| Step | Rules |
|---|---:|
| Mined on training data (FP-Growth, support 0.005, confidence ≥ 0.5, lift ≥ 1.2) | 897 |
| Still hold on the unseen test period | 632 |
| After removing redundant rules | **152** |

FP-Growth found exactly the same rules as Apriori, over 20× faster (1.3 s vs
28.5 s). Time of day appears in every final rule. Extra weather or calendar
conditions usually added nothing, so they were pruned.

## Dashboard

[dashboard/](dashboard/README.md) is an interactive Streamlit app that tells the
project as a story (data → EDA → models → time series → rules → business impact)
and lets users try the work live:

| Tool | What the user does | What runs |
|---|---|---|
| Demand predictor | Pick a zone and hour, change weather or recent demand | The saved HistGradientBoosting model predicts pickups and compares them with normal demand |
| Forecast planner | Pick an area or zone and a start date | Time-series Model 1 forecasts the next 24 hours or 7 days |
| Dispatch advisor | Describe a situation (time, day, weather, zone type) | Matches it against the 152 validated rules |

Nothing is retrained: the app uses the saved model, the team's forecasting code
and the saved rules. Quick start:

```bash
pip install -r dashboard/requirements.txt
python dashboard/build_data.py    # one time only
cd dashboard && streamlit run app.py
```

Step-by-step virtual-environment setup for macOS and Windows is in
[dashboard/README.md](dashboard/README.md).

## Repository structure

```text
data/
  DATA_GUIDE.md
  dataset-documents/      how each dataset was collected and prepared
  raw/                    zones, weather, collection summary (large trip file is local only)
  processed/              features.csv, time_series/, association_rules/, arm_findings/
EDA_doc/                  EDA findings, figures and tables
models/
  *.ipynb                 Review 1 baseline, regression models, comparison, weather ablation
  modeling/               Review 1 predictions, metrics and saved models
  time_series/model_1/    Review 2 time-series Model 1 (code, notebook, figures, tables)
timeSeries/               Review 2 time-series Models 2 (DHR) and 3 (MSTL + ETS)
ARM_doc/                  association-rule preparation and mining documentation
scripts/                  data collection, processing, EDA, time-series and ARM notebooks
tests/                    tests for the association-rule preparation
dashboard/                Streamlit dashboard
```

The large Random Forest `.joblib` file is not in GitHub (it exceeds the 100 MB
limit); rerun its notebook to recreate it.

## Limitations

- Only six months of data (autumn and winter), so there is no yearly seasonality.
- Pickups measure served demand, not people who could not find a cab.
- Sudden events such as the February 2026 blizzard cause large forecast errors.
- The Review 2 test period had been looked at during earlier work; a fresh period
  is needed for a fully blind check.
- [scripts/EDA_visuallization.ipynb](scripts/EDA_visuallization.ipynb) is an older
  notebook with outdated paths; use the Arun EDA scripts instead.
