# Data Dictionary

This file describes the datasets produced by the collection pipeline and every field in them, and explains why these fields were chosen out of everything the TLC and Open-Meteo sources provide. How the data was collected is in [data_collection.md](data_collection.md).

## 1. Main dataset: trips with weather

**File:** `data/raw/NYC-taxi-demand-Manhattan-Queens.csv`

| Property | Value |
|---|---|
| Grain | **One row per taxi trip** |
| Rows | 23,318,979 (23,096,876 yellow + 222,103 green) |
| Columns | 14 |
| Area | Trips **picked up** in the 69 Manhattan + 69 Queens TLC taxi zones |
| Period | Pickups from 2025-09-01 00:00:00 to 2026-02-28 23:59:59, local NYC time |
| Sources | NYC TLC Trip Record Data (yellow + green), TLC Taxi Zones, Open-Meteo Historical Forecast API |
| Join | Weather is matched to each trip on **pickup zone + pickup hour** |
| Missing values | None |
| Size | about 3.2 GB (CSV). Not in git because of its size. Shared as a private Kaggle dataset ([adithyaajay1210/nyc-taxi-demand-manhattan-queens](https://www.kaggle.com/datasets/adithyaajay1210/nyc-taxi-demand-manhattan-queens)), or rebuilt by running `scripts/data_collection.ipynb` (see `data/DATA_GUIDE.md`) |
| Row order | By month, sorted by pickup time within each month |

### Field descriptions

| # | Field | Type | Unit / format | Source | Description | Purpose in the project |
|---|---|---|---|---|---|---|
| 1 | `taxi_type` | text | `yellow` / `green` | from the source file name | Which TLC fleet the trip came from | Lets yellow and green trips share one schema and be analysed together or separately |
| 2 | `pickup_datetime` | datetime | `YYYY-MM-DD HH:MM:SS`, local NYC time | TLC `tpep_pickup_datetime` (yellow) / `lpep_pickup_datetime` (green) | When the meter was engaged, i.e. when the passenger was picked up | **When** demand happened. The basis for hourly demand and time features (hour, weekday, month, weekend), and used to match weather |
| 3 | `dropoff_datetime` | datetime | `YYYY-MM-DD HH:MM:SS`, local NYC time | TLC `tpep_dropoff_datetime` / `lpep_dropoff_datetime` | When the meter was disengaged | Gives trip duration, which is needed to spot bad records (dropoff before pickup, trips of many hours), and shows when the cab becomes free again |
| 4 | `PULocationID` | integer | TLC zone ID (1–265) | TLC | Zone where the trip started. Always one of the 138 Manhattan/Queens zones | **Where** demand happened. Used for the borough filter and the weather lookup, and it is the spatial unit of the forecast |
| 5 | `DOLocationID` | integer | TLC zone ID (1–265) | TLC | Zone where the trip ended. Can be in any borough | Where riders go and where cabs end up after a trip, which matters for fleet positioning |
| 6 | `pickup_borough` | text | `Manhattan` / `Queens` | TLC zone lookup | Borough of the pickup zone | Group and compare the two boroughs |
| 7 | `pickup_zone` | text | zone name | TLC zone lookup | Name of the pickup zone, e.g. *Midtown Center*, *JFK Airport* | Readable label for EDA, charts and the dashboard |
| 8 | `pickup_zone_latitude` | float | decimal degrees (WGS84) | computed from the TLC zone shapefile | Latitude of the pickup zone's centroid | The point sent to Open-Meteo for this zone's weather. Also usable for maps |
| 9 | `pickup_zone_longitude` | float | decimal degrees (WGS84) | computed from the TLC zone shapefile | Longitude of the pickup zone's centroid | Same as above |
| 10 | `pickup_hour` | datetime | `YYYY-MM-DD HH:00:00`, local NYC time | `pickup_datetime` rounded down to the hour | The clock hour the trip started in | **Join key with the weather table**, and the time key for hourly analysis |
| 11 | `apparent_temperature` | float | °C, at the start of `pickup_hour` | Open-Meteo | "Feels-like" temperature at 2 m for the pickup zone. Combines air temperature, humidity, wind (wind chill) and sun | Weather feature for comfort. Range −25.3 to 35.4 °C |
| 12 | `rain` | float | mm, total for `pickup_hour` → +1 h | Open-Meteo | Rain from large-scale weather systems during the pickup hour. Does not include convective showers or snow | Weather feature for wet conditions. 1,090,584 trips started in an hour with rain |
| 13 | `snowfall` | float | cm, total for `pickup_hour` → +1 h | Open-Meteo | Snowfall during the pickup hour | Weather feature for snow. 312,615 trips started in an hour with snowfall |
| 14 | `precipitation_probability` | integer | %, 0–100, for `pickup_hour` → +1 h | Open-Meteo | Model probability of more than 0.1 mm of precipitation during the pickup hour | How likely wet weather looked in the forecast, which people use to plan trips in advance |

Notes:
- All timestamps are **naive local New York time** (no UTC offset). TLC publishes them this way, and the weather was requested with `timezone=America/New_York`, so both use the same clock.
- Open-Meteo labels rain, snowfall and precipitation probability with the **end** of the hour they cover. The pipeline shifts them back one hour, so all four weather fields refer to the pickup hour (see Step 2 in `data_collection.md`).
- The zone coordinates describe the **zone** (its centroid), not the exact pickup point. TLC stopped publishing exact coordinates in 2016 for privacy, and only zone IDs are available.
- Weather values come from the Open-Meteo grid cell nearest to the zone centroid. The 138 zones fall into 50 grid cells, so some neighbouring zones share the same weather values.
- The trips are **not cleaned** yet (see section 6 of `data_collection.md`).
- Trips from both 01:00 hours on **2025-11-02** (end of daylight saving time) share the same `pickup_hour` and weather.

## 2. Supporting files

### `data/raw/weather_hourly_by_zone.csv`

Hourly weather for every zone and every hour: 599,472 rows = 138 zones × 4,344 hours.

| Field | Type | Unit | Description |
|---|---|---|---|
| `LocationID` | integer | | TLC zone ID |
| `weather_hour` | datetime | local NYC time | Start of the hour |
| `apparent_temperature` | float | °C | As in section 1 |
| `rain` | float | mm | As in section 1 |
| `snowfall` | float | cm | As in section 1 |
| `precipitation_probability` | integer | % | As in section 1 |
| `grid_latitude` | float | degrees | Latitude of the Open-Meteo grid cell actually used |
| `grid_longitude` | float | degrees | Longitude of the grid cell actually used |
| `grid_elevation_m` | float | metres | Elevation of the grid cell used by the API |

### `data/raw/taxi_zones_manhattan_queens.csv`

| Field | Type | Description |
|---|---|---|
| `LocationID` | integer | TLC zone ID |
| `Borough` | text | `Manhattan` or `Queens` |
| `Zone` | text | Zone name |
| `service_zone` | text | TLC service area: `Yellow Zone` (Manhattan south of 96th/110th St, where green cabs cannot pick up street hails), `Boro Zone`, or `Airports` (JFK, LaGuardia) |
| `latitude` | float | Zone centroid latitude (WGS84) |
| `longitude` | float | Zone centroid longitude (WGS84) |

### `data/raw/taxi_collection_summary.csv`

One row per downloaded TLC file, as an audit trail.

| Field | Description |
|---|---|
| `taxi_type`, `month` | Which file |
| `source_url`, `file_size_mb` | Where it was downloaded from and its size |
| `rows_in_file` | All trips in the file (whole city) |
| `borough_pickups` | Trips picked up in Manhattan or Queens |
| `outside_period_dropped` | Trips dropped because the pickup time was outside the study period |
| `rows_kept` | Trips kept (= `borough_pickups − outside_period_dropped`) |
| `manhattan_pickups`, `queens_pickups` | `rows_kept` split by borough |
| `duplicate_rows` | Trips with the same values in all five collected fields as another trip in the same file |
| `dropoff_not_after_pickup` | Trips whose dropoff time is at or before the pickup time |
| `longer_than_6h` | Trips lasting more than 6 hours |
| `dropoff_zone_unknown` | Trips whose dropoff zone is 264 (Unknown) or 265 (Outside of NYC) |

## 3. Why these fields were chosen

The project predicts **how many taxis are needed, where, and when**, and asks how weather changes that. The target is the number of pickups per zone per hour. A field was kept only if it helps measure that target or explain it. The field list was set in the project instructions. The reasoning behind each choice is below.

### 3.1 Taxi fields

The TLC files have 20 columns (yellow) and 21 columns (green). We kept 4.

| TLC field | Kept? | Reason |
|---|---|---|
| `tpep/lpep_pickup_datetime` | ✅ | The **when** of demand. Without it there is no hourly demand and no way to match weather |
| `tpep/lpep_dropoff_datetime` | ✅ | Gives trip duration, used to check records and to see when a cab is free again |
| `PULocationID` | ✅ | The **where** of demand. Needed for the borough filter, the zone-level forecast and the weather lookup |
| `DOLocationID` | ✅ | Where trips end, i.e. where cabs end up. Supports fleet repositioning analysis |
| `VendorID` | ❌ | Which technology provider recorded the trip. Administrative only, no effect on demand |
| `passenger_count` | ❌ | We forecast **trips (cabs needed)**, not people. The field is also driver-entered and often 0 or missing |
| `trip_distance` | ❌ | Describes the trip once it happens. Does not measure or explain how many trips are requested |
| `RatecodeID` | ❌ | Tariff type (standard, JFK, Newark…). A pricing detail, not demand |
| `store_and_fwd_flag` | ❌ | Whether the record was buffered in the vehicle before upload. Technical only |
| `payment_type` | ❌ | Cash/card. Known only after the trip, not related to how much demand there is |
| `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount` | ❌ | Revenue fields. The project is about demand volume, not revenue |
| `congestion_surcharge`, `cbd_congestion_fee` | ❌ | Fixed charges added to trips in Manhattan. The pricing rules did not change during the study period, so these fields cannot explain changes in demand from hour to hour. They are fare components, not demand |
| `Airport_fee` (yellow) | ❌ | Charged on pickups at JFK and LaGuardia. Airport pickups are already identified by their zone (`PULocationID` 132 and 138) |
| `ehail_fee` (green) | ❌ | Unused by TLC (empty) |
| `trip_type` (green) | ❌ | Street-hail vs dispatch. Only exists for green taxis, so it cannot be compared across both fleets |

With only these four fields, yellow and green files have the **same schema** after renaming the timestamps, so they can be combined without handling fleet-specific columns.

### 3.2 Why both yellow and green taxis

Yellow cabs can pick up anywhere in NYC. Green (Boro) cabs can only pick up street hails outside the Manhattan "Yellow Zone" (south of E 96th St / W 110th St) and outside the airports, so they serve upper Manhattan and Queens. Together they make up all TLC street-hail taxi supply in both boroughs. Green is small overall (222,103 pickups, about 1% of trips), but it is concentrated in exactly the areas yellow cabs serve less. In the Boro Zones (upper Manhattan and non-airport Queens), green cabs made 203,142 of the 1,672,412 pickups (12%). Leaving them out would understate demand there. The `taxi_type` column keeps the fleets apart if needed.

### 3.3 Weather fields

The Open-Meteo Historical Forecast API offers over 50 hourly variables. We kept 4.

| Open-Meteo variable | Kept? | Reason |
|---|---|---|
| `apparent_temperature` | ✅ | How cold or hot it **feels**, which is what makes people choose a cab over walking or waiting for a bus. It already includes the effect of wind and humidity, so one field replaces three |
| `rain` | ✅ | Rain is the classic driver of taxi demand: people avoid walking and waiting at stops in the rain. Kept separate from snow because the effect is different |
| `snowfall` | ✅ | Snow raises demand but also slows traffic and reduces supply, which is a different effect from rain. The study period covers winter 2025–26, so snow events are included |
| `precipitation_probability` | ✅ | The chance of rain or snow **in the forecast**. People plan around the forecast, and a dispatcher planning a day ahead only has a forecast. This variable exists in the Historical Forecast API but not in the reanalysis archive, which is one reason that API was chosen |
| `temperature_2m` | ❌ | Covered by `apparent_temperature`, which is closer to how people feel. Using both would give two highly correlated features |
| `relative_humidity_2m`, `dew_point_2m` | ❌ | Their effect on comfort is already included in `apparent_temperature` |
| `wind_speed_10m`, `wind_gusts_10m`, `wind_direction_10m` | ❌ | Wind chill is already included in `apparent_temperature`. Direction has no clear link to demand |
| `precipitation` | ❌ | Equals rain + showers + snowfall. It would repeat `rain` and `snowfall` and mix two effects we want to keep apart |
| `showers` | ❌ (not in the instructions) | Convective rain. Could be added later if short downpours matter (see `data_collection.md`) |
| `snow_depth` | ❌ | Snow already on the ground. Affects roads more than the decision to hail a cab, and is strongly tied to recent `snowfall` |
| `weather_code` | ❌ | A category code (clear, fog, drizzle, thunderstorm…) that is a summary of the numeric variables we already have |
| `cloud_cover*`, `visibility`, radiation, `sunshine_duration`, `uv_index` | ❌ | Little expected effect on taxi use once rain, snow and temperature are known |
| `pressure_msl`, `surface_pressure`, `cape`, soil variables, upper-level variables | ❌ | Meteorological inputs, not conditions people react to |

Together the four chosen variables describe the weather conditions riders react to (how it feels, whether it is raining or snowing) and what the forecast said. They also avoid the redundancy of the full variable list.

### 3.4 Fields added by the pipeline

| Field | Why it was added |
|---|---|
| `taxi_type` | Needed once yellow and green are combined |
| `pickup_borough`, `pickup_zone` | Human-readable labels for analysis and the dashboard |
| `pickup_zone_latitude`, `pickup_zone_longitude` | Record which point the weather was taken for. The instructions ask for weather by zone latitude/longitude |
| `pickup_hour` | The exact key used to match weather to each trip |
