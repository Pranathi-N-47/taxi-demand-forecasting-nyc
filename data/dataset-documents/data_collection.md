# Data Collection

This document describes how the data for the project was collected: which sources were used, how each API was called, and how every taxi trip was matched with the weather at its pickup zone and hour. Field-level descriptions and the reasons for choosing each field are in [data_dictionary.md](data_dictionary.md).

## 1. Scope

| Item | Value |
|---|---|
| Area | **Manhattan** and **Queens** (69 + 69 = 138 TLC taxi zones) |
| Taxi types | **Yellow** and **Green** taxis |
| Period | **1 Sep 2025 00:00 to 28 Feb 2026 23:59** (6 months, 181 days, 4,344 hours), local NYC time |
| Taxi fields | Pickup time, dropoff time, `PULocationID`, `DOLocationID` |
| Weather fields | Hourly apparent temperature, rain, snowfall, precipitation probability |
| Weather location | One point per taxi zone (the zone's centroid latitude/longitude) |
| Final output | **One row per trip**: the four taxi fields plus the weather at the trip's pickup zone and hour |

A trip belongs to the study when it is **picked up** in a Manhattan or Queens zone (`PULocationID` in the 138 zones). The drop-off can be anywhere. We filter on the pickup because the project forecasts where passengers need a cab, and the weather joined to a trip is the weather the passenger faced when they got in.

## 2. Data sources

| # | Data | Provider | Access method | Endpoint |
|---|---|---|---|---|
| 1 | Yellow taxi trip records | NYC Taxi & Limousine Commission (TLC) | Programmatic HTTP requests to the official monthly Parquet files | `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet` |
| 2 | Green taxi trip records | NYC TLC | Same as above | `https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_YYYY-MM.parquet` |
| 3 | Taxi zone lookup (ID → zone name → borough) | NYC TLC | Programmatic HTTP request (CSV) | `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv` |
| 4 | Taxi zone boundaries | NYC TLC | Programmatic HTTP request (zipped shapefile) | `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip` |
| 5 | Hourly weather | Open-Meteo | REST API (JSON) | `https://historical-forecast-api.open-meteo.com/v1/forecast` |

- TLC trip record page: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
- Open-Meteo Historical Forecast API docs: <https://open-meteo.com/en/docs/historical-forecast-api>

Both sources are free and need no API key.

### How the data was accessed

Taxi trip data was collected programmatically through HTTP requests to the NYC TLC's official trip-data endpoint. The NYC Open Data API was not used because it does not yet publish yellow taxi data for 2025 or any taxi data for 2026. Weather data was collected through the Open-Meteo Historical Forecast REST API.

All requests are made from Python code in the collection notebook, with no manual downloads. The TLC endpoint serves one complete file per taxi type per month, at the same URLs that the TLC Trip Record Data page links to. The Open-Meteo API answers queries: we send the coordinates, dates, variables and timezone, and it returns JSON.

### Why these sources

- **TLC monthly Parquet files.** These are the official trip records linked from the TLC Trip Record Data page. Each month has a fixed URL, so the collection can be scripted and repeated, and every team member gets the same input. The endpoint returns whole monthly files, not query results, so the notebook fetches each file once and does the filtering (borough, date, columns) itself.
- **Why not the NYC Open Data (Socrata) API.** NYC Open Data publishes TLC trip data as one dataset per year and taxi type, with a query API. When we checked its catalog on 30 Sep 2026, it could not cover our study period:
  - the newest yellow dataset is *2024 Yellow Taxi Trip Data*. There is no 2025 yellow dataset yet.
  - *2025 Green Taxi Trip Data* exists, but its trips end around 1 Jan 2026.
  - there are no 2026 taxi trip datasets at all.

  The API would also return rows in pages of up to 50,000. Pulling our 23 million trips would take hundreds of paged requests, with rate limits and a risk of gaps.
- **Open-Meteo Historical Forecast API.** It gives hourly weather for any latitude/longitude, which lets us get weather **per taxi zone** instead of one weather station for the whole city. It is built from archived runs of high-resolution weather models, and it has `precipitation_probability`. The Historical Weather (reanalysis archive) API does not have that variable.

## 3. Collection steps

All the code is in one notebook, [`scripts/data_collection.ipynb`](../../scripts/data_collection.ipynb), which explains every step in text cells next to the code. Run all cells from top to bottom to rebuild everything (about 15 minutes, mostly downloads, API pauses and writing the 3.2 GB CSV):

```bash
pip install -r requirements.txt
# then open scripts/data_collection.ipynb and run all cells
```

```
 TLC zone lookup ─┐
                  ├─► Step 1: 138 Manhattan + Queens zones with centroid lat/lon
 TLC zone shapes ─┘            (data/raw/taxi_zones_manhattan_queens.csv)
                                  │                                  │
                                  ▼                                  ▼
                   Step 2: Open-Meteo hourly weather      Step 3: TLC monthly files (yellow + green,
                   for every zone centroid                Sep 2025 – Feb 2026) → trips picked up
                   (data/raw/weather_hourly_by_zone.csv)  in these zones
                                  │                                  │
                                  └──────────────┬───────────────────┘
                                                 ▼
                              Step 4: every trip + the weather at its pickup zone and hour
                              (data/raw/NYC-taxi-demand-Manhattan-Queens.csv)  ◄── FULL DATASET
```

### Step 1: Zones and their coordinates (notebook Step 1)

1. Download `taxi_zone_lookup.csv` (265 zones) and keep the rows where `Borough` is `Manhattan` or `Queens`. This gives **138 zones** (69 each).
2. Download `taxi_zones.zip`, the TLC zone boundary shapefile. The coordinates are in the NY State Plane Long Island system (**EPSG:2263**, US feet).
3. For each zone polygon, compute the **area-weighted centroid** in State Plane coordinates. Holes and multi-part zones (for example islands) are handled by summing the signed areas of all rings.
4. Convert the centroid to **WGS84 latitude/longitude** (EPSG:4326) with `pyproj`, since that is what Open-Meteo expects.
5. Save `data/raw/taxi_zones_manhattan_queens.csv` with `LocationID, Borough, Zone, service_zone, latitude, longitude`.

The notebook stops with an error if any zone has no shape. Spot checks: Central Park (43) → 40.7825, −73.9656; JFK Airport (132) → 40.6470, −73.7865; LaGuardia Airport (138) → 40.7744, −73.8736.

### Step 2: Hourly weather per zone (notebook Step 2)

1. Read the 138 zone centroids from Step 1.
2. Call the Open-Meteo Historical Forecast API. The API accepts several comma-separated coordinates in one request, so the zones are sent in **14 requests of up to 10 zones**. Parameters:

   | Parameter | Value | Why |
   |---|---|---|
   | `latitude`, `longitude` | zone centroids | weather at each zone |
   | `start_date`, `end_date` | `2025-09-01`, `2026-03-01` | study period + 1 day (see point 4) |
   | `hourly` | `apparent_temperature,rain,snowfall,precipitation_probability` | the four required variables |
   | `timezone` | `America/New_York` | returns local NYC hours, the same clock as the TLC timestamps |

   Example request (one zone, Midtown Center):
   ```
   https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=40.75803&longitude=-73.9777&start_date=2025-09-01&end_date=2026-03-01&hourly=apparent_temperature,rain,snowfall,precipitation_probability&timezone=America/New_York
   ```
   **Rate limits.** The free API allows 600 "calls" per minute. Open-Meteo counts one location over six months as about 13 calls, so a 10-zone request is about 130 calls. The notebook waits 15 seconds between requests. If it still gets HTTP 429 (limit reached), it waits a full minute and retries. Network errors are retried as well.
3. Turn each zone's response into rows of `LocationID, weather_hour, <4 variables>`. We also keep the grid-cell latitude, longitude and elevation that Open-Meteo actually used.
4. **Align every value to the hour that starts at its timestamp.** Open-Meteo reports `apparent_temperature` as an instant value at the timestamp, but `rain`, `snowfall` and `precipitation_probability` as totals or probabilities for the **preceding** hour (the 07:00 value covers 06:00–07:00). These three are shifted back one hour, so the 07:00 row holds the temperature at 07:00 and the rain, snow and probability for **07:00–08:00**. To fill the last hour of the period, the request asks for one extra day (up to 1 Mar 2026), which is then trimmed off.
5. **Check completeness**: every zone must have exactly one row for each of the **4,344 hours** in the period. The notebook stops with an error if not. No values were missing.
6. Save to `data/raw/weather_hourly_by_zone.csv` (138 × 4,344 = **599,472 rows**).

### Step 3: Taxi trips (notebook Step 3)

For each month (2025-09 to 2026-02) and each taxi type (yellow, green), that is 12 files, one month at a time to keep memory low:

1. **Download** the monthly Parquet file into `data/raw/tlc_downloads/` (a cache, not in git). Downloads are streamed to disk and retried on errors. A file that is already on disk is not downloaded again.
2. **Read only the four needed columns**, pushing the filter `PULocationID IN (138 Manhattan/Queens zone IDs)` into the Parquet reader.
3. **Standardise the column names.** Yellow files use `tpep_pickup_datetime` / `tpep_dropoff_datetime` and green files use `lpep_…`. Both are renamed to `pickup_datetime` / `dropoff_datetime`, and a `taxi_type` column is added so the two fleets can be combined.
4. **Keep only trips whose pickup falls in the study period.** TLC files contain a few trips with meter clock errors (dates from other months or years). We have no weather for those. **33 trips** were dropped this way.
5. Pass the month's trips on to Step 4. Steps 3 and 4 run together in one loop, month by month, so no intermediate trip files are saved.
6. Write a per-file audit table to `data/raw/taxi_collection_summary.csv`: URL, file size, rows in the file, Manhattan/Queens pickups, rows dropped, rows kept, and the data-quality counts in section 6.

| Month | Yellow rows in file | Yellow kept | Green rows in file | Green kept |
|---|---:|---:|---:|---:|
| 2025-09 | 4,251,015 | 4,048,166 | 48,893 | 40,059 |
| 2025-10 | 4,428,699 | 4,234,338 | 49,416 | 40,573 |
| 2025-11 | 4,181,444 | 3,982,593 | 46,912 | 38,373 |
| 2025-12 | 4,305,006 | 4,096,636 | 48,236 | 39,806 |
| 2026-01 | 3,724,889 | 3,525,433 | 40,272 | 32,792 |
| 2026-02 | 3,399,866 | 3,209,710 | 37,373 | 30,500 |
| **Total** | **24,290,919** | **23,096,876** | **271,102** | **222,103** |

"Kept" is counted by the file a trip came from. A file can hold a few trips from the last minutes of the previous month, so monthly totals by pickup date (section 5) differ very slightly.

In total **23,318,979 of the 24,562,021 trips (94.9%)** in the 12 files were picked up in Manhattan or Queens. The rest started in Brooklyn, the Bronx, Staten Island, Newark Airport or an unknown zone.

### Step 4: Add weather to every trip (notebook Step 4)

For every trip, one month (yellow + green together) at a time:

1. Take the pickup zone `PULocationID`.
2. Look up the zone's borough, name and centroid latitude/longitude (Step 1). This is the location used for the weather request.
3. Round the pickup time **down to the hour**: `pickup_hour = floor(pickup_datetime, 1h)`. For example, a pickup at 07:42 gets the 07:00 weather row: the apparent temperature at 07:00 and the rain, snowfall and precipitation probability for 07:00–08:00.
4. Left-join the weather table on **(`PULocationID`, `pickup_hour`)**.
5. Check that the row count did not change (each join is validated as many-to-one) and that every trip has weather. **All 23,318,979 trips got weather.**
6. **Append** the month to `data/raw/NYC-taxi-demand-Manhattan-Queens.csv`. The CSV is written to a temporary `.part` file and renamed only after every month has gone through, and the notebook checks that the number of rows written equals the number of trips kept in the audit table.

This is the **full dataset** of the collection: one CSV with 23,318,979 rows and 14 columns, about **3.2 GB**. Rows are in month order and sorted by pickup time within each month. The few trips a TLC file carries from the last minutes of the previous month appear at the start of the next month's block.

The file is too large for GitHub (100 MB limit per file), so it is **not committed**. It is shared with the team as a private Kaggle dataset, [adithyaajay1210/nyc-taxi-demand-manhattan-queens](https://www.kaggle.com/datasets/adithyaajay1210/nyc-taxi-demand-manhattan-queens), and anyone can also rebuild it by running the notebook. How to get it and load it: [data/DATA_GUIDE.md](../DATA_GUIDE.md).

## 4. Output files

All outputs of the collection are in `data/raw/`:

| File | In git | Rows | Size | Description |
|---|---|---:|---:|---|
| `NYC-taxi-demand-Manhattan-Queens.csv` | no ([private Kaggle dataset](https://www.kaggle.com/datasets/adithyaajay1210/nyc-taxi-demand-manhattan-queens)) | 23,318,979 | 3.2 GB | **Full dataset**: every trip with the weather at its pickup zone and hour |
| `taxi_zones_manhattan_queens.csv` | yes | 138 | 8 KB | Zones with centroid latitude/longitude |
| `weather_hourly_by_zone.csv` | yes | 599,472 | 39 MB | Hourly weather for every zone and hour (138 × 4,344), with the grid cell Open-Meteo used |
| `taxi_collection_summary.csv` | yes | 12 | 2 KB | Audit of each downloaded TLC file |
| `tlc_downloads/` | no | n/a | ~425 MB | Cache of the original TLC files, so reruns skip the downloads |

`data/processed/` is empty. It is reserved for the next stages of the project. The dataset CSV and `tlc_downloads/` are in `.gitignore` because of their size. Where each file is and how to use it: [data/DATA_GUIDE.md](../DATA_GUIDE.md).

The weather table is also kept on its own. It has weather for **every** zone and hour, including hours with no trips, which later steps can use.

## 5. Dataset summary

These numbers come from Step 5 (Dataset profile) of the notebook.

**Full dataset** (`NYC-taxi-demand-Manhattan-Queens.csv`): 23,318,979 rows, 14 columns, pickups from 2025-09-01 00:00:00 to 2026-02-28 23:59:59. There are no missing values.

| Pickup borough | Zones | Yellow trips | Green trips | Total trips |
|---|---:|---:|---:|---:|
| Manhattan | 69 | 20,856,906 | 159,610 | 21,016,516 |
| Queens | 69 | 2,239,970 | 62,493 | 2,302,463 |
| **Total** | **138** | **23,096,876** | **222,103** | **23,318,979** |

| Month (by pickup time) | Yellow | Green | Total |
|---|---:|---:|---:|
| 2025-09 | 4,048,175 | 40,052 | 4,088,227 |
| 2025-10 | 4,234,347 | 40,575 | 4,274,922 |
| 2025-11 | 3,982,584 | 38,375 | 4,020,959 |
| 2025-12 | 4,096,633 | 39,799 | 4,136,432 |
| 2026-01 | 3,525,437 | 32,792 | 3,558,229 |
| 2026-02 | 3,209,700 | 30,510 | 3,240,210 |

- **Busiest zones**: Upper East Side South (1,090,154 pickups), JFK Airport (988,812), Midtown Center (988,458), Upper East Side North (984,375), Penn Station/Madison Sq West (748,457).
- **Quietest zones**: the three Governor's Island/Ellis Island/Liberty Island zones (103, 104, 105) had 0, 0 and 7 pickups. Jamaica Bay (2) had 38 and Breezy Point/Fort Tilden/Riis Beach (27) had 98.
- **Time of day**: pickups are lowest at 04:00 (161,898 over the whole period) and highest at 18:00 (1,676,922).
- **Destinations of the collected trips**: Manhattan 20.6 M, Queens 1.31 M, Brooklyn 977 K, Bronx 231 K, Newark Airport 42 K, Staten Island 5 K, Unknown / outside NYC 133 K.
- **Weather**: apparent temperature from −25.3 °C to 35.4 °C. Rain fell in 5.3% of zone-hours and snow in 2.4%. 1,090,584 trips started in an hour with rain and 312,615 in an hour with snowfall.

## 6. Things to know before using the data

**Trips are not cleaned yet.** Collection filters only by borough and period, so the dataset includes every collected trip. The TLC files contain records the data preparation step may want to remove. They are counted per file in `taxi_collection_summary.csv`:

| Issue | Trips | Share |
|---|---:|---:|
| Same values in all five collected fields as another trip in the same file (possible duplicate) | 276,749 | 1.2% |
| Dropoff at or before pickup | 329,681 | 1.4% |
| Trip longer than 6 hours | 6,877 | 0.03% |
| Dropoff zone Unknown / Outside NYC (264, 265) | 133,038 | 0.6% |

**Timezone and daylight saving.** TLC timestamps are naive local NYC time, and weather was requested in `America/New_York`, so both use the same clock. When daylight saving ended on 2 Nov 2025, 01:00–01:59 happened twice. TLC timestamps cannot tell the two apart, so trips from both hours get `pickup_hour` **2025-11-02 01:00** (10,850 trips, against 6,684 at the same hour the following Sunday). Open-Meteo returns one 01:00 row for that day, so all of these trips get the same weather. The spring-forward change (8 Mar 2026) is outside the study period.

**Weather resolution.** Weather is requested at each zone's centroid, but Open-Meteo returns the value of the nearest model grid cell. The 138 zones map to **50 distinct grid cells**, so some neighbouring zones share identical weather. The grid cell used for each zone is kept in `weather_hourly_by_zone.csv` (`grid_latitude`, `grid_longitude`, `grid_elevation_m`).

**Centroid as the zone location.** A zone centroid stands for the whole zone. Manhattan zones are small, so the centroid is close to anywhere in the zone. Some Queens zones (Jamaica Bay, the airports, parks) are large, so their centroid is a rougher stand-in for the whole zone.

**Rain vs showers.** In Open-Meteo, `rain` is rain from large-scale weather systems and `showers` is convective rain, reported separately. `showers` is not one of the four required variables, so short convective downpours are not counted in `rain`. If this matters for modelling, add `showers` to `WEATHER_VARIABLES` in the notebook's settings cell and rerun.

**Data versions.** Data was collected on **30 Sep 2026**. The TLC 2026-02 files were last modified on 25 Mar 2026. TLC occasionally republishes files, so rerunning the pipeline later could give slightly different counts. `taxi_collection_summary.csv` records the file sizes that were used.

## 7. Notebook

All the collection code is in [`scripts/data_collection.ipynb`](../../scripts/data_collection.ipynb):

| Notebook section | What it does |
|---|---|
| 0. Setup | Imports, settings (study period, boroughs, taxi types, URLs, weather variables, output paths), and helpers for logging, HTTP requests with retries (including API rate limits) and cached file downloads |
| Step 1 | Manhattan and Queens zones and their centroid coordinates |
| Step 2 | Hourly weather per zone from Open-Meteo |
| Step 3 | Download the TLC files and keep Manhattan/Queens pickups |
| Step 4 | Add the pickup zone's weather to every trip |
| Step 5 | Dataset profile: the summary numbers used in this document |
