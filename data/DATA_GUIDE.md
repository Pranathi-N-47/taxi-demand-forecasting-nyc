# Data Guide

This guide explains what is in the `data/` folder, how to get the full dataset, and how to load and use it. How the data was collected is described in [dataset-documents/data_collection.md](dataset-documents/data_collection.md). Every column is described in [dataset-documents/data_dictionary.md](dataset-documents/data_dictionary.md).

## 1. Folder layout

```
data/
├── DATA_GUIDE.md                             this file
├── dataset-documents/                        documentation of the dataset
│   ├── data_collection.md                      how the data was collected (sources, APIs, steps)
│   ├── data_dictionary.md                      every field, its purpose and why it was chosen
│   ├── data_processing.md                      cleaning and the ML feature table
│   └── time_series_preparation.md              the time-series dataset: steps, files and fields
├── raw/                                      everything collected in the data collection stage
│   ├── NYC-taxi-demand-Manhattan-Queens.csv    full dataset (3.2 GB, NOT in git, on Kaggle)
│   ├── taxi_zones_manhattan_queens.csv         the 138 taxi zones with latitude/longitude
│   ├── weather_hourly_by_zone.csv              hourly weather for every zone
│   ├── taxi_collection_summary.csv             audit of the 12 downloaded TLC files
│   └── tlc_downloads/                          cache of the original TLC files (NOT in git)
└── processed/                                cleaned and prepared data
    ├── features.csv (+ _manhattan, _queens)    zone-hour feature table for the ML models
    ├── cleaning_report.md                      rows removed by each cleaning rule
    └── time_series/                            analysis-ready series for the time-series stage
```

**Rule of thumb:** `raw/` holds data exactly as collected. Don't edit these files. Anything cleaned, aggregated or engineered goes in `processed/`.

## 2. The files in `data/raw/`

### 2.1 `NYC-taxi-demand-Manhattan-Queens.csv`: the full dataset

| | |
|---|---|
| What | Every yellow and green taxi trip **picked up** in Manhattan or Queens from 1 Sep 2025 to 28 Feb 2026, with the weather at its pickup zone during its pickup hour |
| Rows | 23,318,979 (one row per trip) |
| Columns | 14 |
| Size | about 3.2 GB |
| In git | **No**, it is too large for GitHub. Get it from Kaggle (section 3) |
| Order | By month, sorted by pickup time within each month |

Columns (the example is a real row, the first Midtown Center pickup in the file):

| Column | Example | Meaning |
|---|---|---|
| `taxi_type` | `yellow` | `yellow` or `green` |
| `pickup_datetime` | `2025-09-01 00:00:00` | When the trip started (local NYC time) |
| `dropoff_datetime` | `2025-09-01 00:24:22` | When the trip ended (local NYC time) |
| `PULocationID` | `161` | Pickup taxi zone ID |
| `DOLocationID` | `56` | Dropoff taxi zone ID (can be in any borough) |
| `pickup_borough` | `Manhattan` | Borough of the pickup zone |
| `pickup_zone` | `Midtown Center` | Name of the pickup zone |
| `pickup_zone_latitude` | `40.75803` | Centre of the pickup zone (where the weather was taken) |
| `pickup_zone_longitude` | `-73.9777` | Centre of the pickup zone |
| `pickup_hour` | `2025-09-01 00:00:00` | Pickup time rounded down to the hour (the key used to match the weather) |
| `apparent_temperature` | `17.8` | "Feels-like" temperature in °C at the start of the pickup hour |
| `rain` | `0.0` | Rain in mm during the pickup hour |
| `snowfall` | `0.0` | Snow in cm during the pickup hour |
| `precipitation_probability` | `0` | Chance of rain or snow in the pickup hour, in % |

### 2.2 `taxi_zones_manhattan_queens.csv`: zone list

138 rows, one per taxi zone (69 Manhattan + 69 Queens). Columns: `LocationID`, `Borough`, `Zone`, `service_zone`, `latitude`, `longitude`.

Use it to turn a zone ID into a name or borough, or to put zones on a map. Join it on `LocationID` = `PULocationID` (or `DOLocationID`).

### 2.3 `weather_hourly_by_zone.csv`: weather table

599,472 rows = 138 zones × 4,344 hours. It has weather for **every** zone and **every** hour, including hours with no taxi pickups. Columns: `LocationID`, `weather_hour`, the four weather fields, and `grid_latitude`, `grid_longitude`, `grid_elevation_m`, which describe the weather-model grid cell Open-Meteo used.

The weather in the full dataset was copied from this table, matched on pickup zone and pickup hour. Use this table when you build anything per zone per hour. The trip file only has weather for hours in which a trip happened.

### 2.4 `taxi_collection_summary.csv`: collection audit

12 rows, one per downloaded TLC file (6 yellow + 6 green months). For each file it records: the source URL, the file size, the number of trips in the file (whole city), how many were picked up in Manhattan/Queens, how many were dropped for dates outside the study period, and how many were kept. It also has four data-quality counts (possible duplicates, dropoff not after pickup, trips over 6 hours, unknown dropoff zone).

The data-quality counts are for the **data preparation** step. Those trips are still in the dataset.

### 2.5 `tlc_downloads/`

The original monthly Parquet files from TLC (~425 MB), plus the TLC zone lookup and zone boundary files. The collection notebook keeps them so that a rerun does not download everything again. They are not in git, and you don't need them to use the dataset.

### 2.6 `data/processed/time_series/`: the time-series dataset

Prepared for the time-series analysis stage by `scripts/time_series_data_preparation.ipynb`, built on `features.csv`. It holds regular hourly and daily demand series from 1 Sep 2025 at four levels (all zones, borough, service area, zone), a calendar and event table, splits that reuse the ML test period, rolling-origin folds, zone profiles and readiness checks. The week that `features.csv` dropped (1-7 Sep) is recovered exactly from its 168-hour lag. All files are in git (the largest, `ts_zone_hourly.csv`, is about 28 MB).

Start with `ts_aggregate_hourly.csv` or `ts_aggregate_daily.csv`: each row already carries every calendar regressor. `ts_feature_catalog.csv` says which columns are safe model inputs. Files, fields and decisions: [dataset-documents/time_series_preparation.md](dataset-documents/time_series_preparation.md).

## 3. How to get the data

### 3.1 Small files (zones, weather, summary)

They are in the repository. Clone or pull and they are in `data/raw/`:

```bash
git clone https://github.com/BA-CapstoneProject-12/nyc-taxi-demand-forecasting.git
```

### 3.2 Full dataset: private Kaggle dataset

The 3.2 GB CSV is shared as a **private Kaggle dataset**. Only teammates who have been added can see it.

**Dataset:** [adithyaajay1210/nyc-taxi-demand-manhattan-queens](https://www.kaggle.com/datasets/adithyaajay1210/nyc-taxi-demand-manhattan-queens)

The Kaggle dataset has the same files as `data/raw/` (except `tlc_downloads/`), plus this guide and the `dataset-documents/` folder.

**Getting access:** send your Kaggle username to the dataset owner (Kaggle user `adithyaajay1210`). Once you have been added, the link above opens for you, and the dataset also shows up under *Your Work → Datasets → Shared with you* on Kaggle.

**Option A: download from the website**
1. Open the [dataset page](https://www.kaggle.com/datasets/adithyaajay1210/nyc-taxi-demand-manhattan-queens) and click **Download**.
2. Unzip it and move `NYC-taxi-demand-Manhattan-Queens.csv` into `data/raw/` of your local copy of the repo. The other files are already in the repo.

**Option B: download with the Kaggle command line**
1. `pip install kaggle`
2. On kaggle.com go to *Settings → API → Create New Token*. This downloads `kaggle.json`. Put it in `C:\Users\<you>\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Mac/Linux).
3. Download and unzip the dataset into a folder of its own, then move the CSV into `data/raw/`:
   ```bash
   kaggle datasets download -d adithyaajay1210/nyc-taxi-demand-manhattan-queens -p kaggle_download --unzip
   ```
   Windows: `move kaggle_download\NYC-taxi-demand-Manhattan-Queens.csv data\raw\`. Mac/Linux: `mv kaggle_download/NYC-taxi-demand-Manhattan-Queens.csv data/raw/`. Then delete the `kaggle_download` folder, so the extra copies don't end up in git.

**Option C: work directly on Kaggle.** In a Kaggle notebook, click *Add Input*, search for the dataset under *Your datasets / Shared with you*, and read it from `/kaggle/input/nyc-taxi-demand-manhattan-queens/`. Nothing needs to be downloaded, and Kaggle notebooks have enough memory for the full file.

### 3.3 Rebuild it yourself

Instead of downloading, you can recreate everything in `data/raw/`:

```bash
pip install -r requirements.txt
```

Then open `scripts/data_collection.ipynb` and run all cells. It takes about 15 minutes and downloads about 425 MB from TLC. The result is the same data, unless TLC republishes its files.

### 3.4 For the owner: creating the Kaggle dataset

The dataset has been created at [adithyaajay1210/nyc-taxi-demand-manhattan-queens](https://www.kaggle.com/datasets/adithyaajay1210/nyc-taxi-demand-manhattan-queens) (private, shared with the team). These are the steps that were used. Follow them again only if it ever has to be recreated.

**On the website**
1. Zip the four files in `data/raw/` (not `tlc_downloads/`), together with `data/DATA_GUIDE.md` and the `data/dataset-documents/` folder, keeping the same layout. This makes the upload much smaller, and Kaggle unzips archives automatically.
2. On kaggle.com: *Create → New Dataset*. Upload the zip, set the title to *NYC Taxi Demand Manhattan Queens*, and keep the visibility **Private**.
3. On the dataset page, open **Settings → Sharing** and add each teammate by Kaggle username with view access.
4. Write the dataset description (About Dataset), with a link to the GitHub repo.

**With the Kaggle command line** (after the API token setup in 3.2 Option B)
1. Put the same files in an empty folder, for example `kaggle_upload/`.
2. `kaggle datasets init -p kaggle_upload` creates `dataset-metadata.json`. Edit it: set `"title": "NYC Taxi Demand Manhattan Queens"` and `"id": "adithyaajay1210/nyc-taxi-demand-manhattan-queens"`.
3. `kaggle datasets create -p kaggle_upload`. New datasets are **private by default**. Don't add `--public`.
4. Add teammates in the dataset's Settings on the website, as in step 3 above.
5. If the data or these documents change later, upload a new version so Kaggle and GitHub stay the same: `kaggle datasets version -p kaggle_upload -m "what changed"`, or on the website use *New Version* on the dataset page.

## 4. How to use the dataset

### 4.1 Loading it

The full CSV needs several GB of memory if you load every column. Load only the columns you need, and tell pandas which columns are dates:

```python
import pandas as pd

trips = pd.read_csv(
    "data/raw/NYC-taxi-demand-Manhattan-Queens.csv",
    usecols=["taxi_type", "pickup_datetime", "PULocationID", "pickup_borough",
             "apparent_temperature", "rain", "snowfall"],
    parse_dates=["pickup_datetime"],
    dtype={"taxi_type": "category", "pickup_borough": "category"},
)
```

If memory is still a problem, read it in chunks and keep only running totals:

```python
counts = []
for chunk in pd.read_csv("data/raw/NYC-taxi-demand-Manhattan-Queens.csv",
                         usecols=["PULocationID", "pickup_hour"],
                         parse_dates=["pickup_hour"], chunksize=2_000_000):
    counts.append(chunk.groupby(["PULocationID", "pickup_hour"]).size())
pickups_per_zone_hour = pd.concat(counts).groupby(level=[0, 1]).sum()
```

Tip: after the first load, save your subset as Parquet in `data/processed/` (for example `trips.to_parquet(...)`). It reloads in seconds and takes far less disk space.

The CSV **cannot be opened in Excel**, because Excel stops at about 1 million rows. The three small files open fine.

### 4.2 Common tasks

**Look up zone names**
```python
zones = pd.read_csv("data/raw/taxi_zones_manhattan_queens.csv")
trips = trips.merge(zones[["LocationID", "Zone"]], left_on="PULocationID", right_on="LocationID")
```

**Weather for hours with no trips.** Use the weather table, not the trip file:
```python
weather = pd.read_csv("data/raw/weather_hourly_by_zone.csv", parse_dates=["weather_hour"])
```

**Only one month or one borough**
```python
nov = trips[trips["pickup_datetime"].dt.strftime("%Y-%m") == "2025-11"]
queens = trips[trips["pickup_borough"] == "Queens"]
```

## Association-rule preparation outputs

The cleaned `data/processed/features.csv` also feeds the Review 2 association-rule
preparation. Its output folder is `data/processed/association_rules/`, containing a
complete categorical audit, eligible train/test metadata, compressed Boolean
one-hot matrices, frozen bin/cluster parameters and validation tables.

Read [dataset-documents/association_rule_preparation.md](dataset-documents/association_rule_preparation.md)
for the exact item definitions, chronology, sparse-zone/DST exclusions and loader
instructions. No raw-trip or Kaggle download is required for this stage.

## 5. Things to keep in mind

- **The data is not cleaned.** Collection only filtered by borough and date. About 1.4% of trips have a dropoff at or before the pickup, about 1.2% look like duplicates, and a few last over 6 hours. The counts per file are in `taxi_collection_summary.csv`. Decide on cleaning rules in the data preparation step.
- **Times are local New York time** with no timezone attached, in both the trips and the weather.
- **Daylight saving time:** on 2 Nov 2025 the hour 01:00–01:59 happened twice. Trips from both are labelled 01:00, so that hour has about double the usual pickups.
- **The weather refers to the pickup hour:** temperature at the start of the hour, and rain, snow and probability for the whole hour.
- **Neighbouring zones can have identical weather.** The 138 zones fall into 50 weather-model grid cells.
- **Pickups define the dataset.** A trip is included if it started in Manhattan or Queens, wherever it ended. Trips from other boroughs into Manhattan/Queens are not included.
