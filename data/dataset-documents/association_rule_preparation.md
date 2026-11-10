# Data Preparation for Association Rule Mining

Owner: Arun N (`ArunN2005`). Review 2, issue [#30](https://github.com/BA-CapstoneProject-12/nyc-taxi-demand-forecasting/issues/30).

This stage converts the existing cleaned zone-hour data into categorical transactions
and Boolean one-hot matrices. It prepares the inputs for the member responsible for
mining frequent itemsets and demand rules in #31, interpreting/exporting rules in #32,
and checking stability in #33. It does not generate association rules.

## 1. Current project context and inputs

The base is `data/processed/features.csv` on `main`: 576,288 rows, 138 zones,
8 Sep 2025 to 28 Feb 2026, with 21,728,117 recorded pickups. Every row is one
pickup zone during one recorded local hour. Yellow and green taxi pickups are pooled.
The first week was removed during lag preparation and remains outside this stage.

Inputs, all already in the repository:

| File | Purpose |
|---|---|
| `data/processed/features.csv` | Clean hourly counts, calendar flags and weather |
| `data/raw/taxi_zones_manhattan_queens.csv` | Zone names, boroughs and service areas |
| `models/modeling/split_summary.csv` | Reuse the initial models' exact chronological split |

The initial modelling and both EDA contributions are complete on `main`. The
`time-series-dataset` branch was reviewed for context. This preparation is independent
of its unmerged outputs and does not change the time-series or modelling data.
`EDA_doc/tables/zone_clusters.csv` describes the full observed period, so it is not
used as a training/test cluster assignment. This stage fits fresh training-only clusters.

## 2. Reproduce and validate

```bash
pip install -r requirements.txt
python scripts/association_rule_data_preparation.py
python scripts/association_rule_data_preparation.py --validate-only
python -m unittest discover -s tests -v
```

Alternatively run all cells in
[`scripts/association_rule_data_preparation.ipynb`](../../scripts/association_rule_data_preparation.ipynb).
The script resolves paths relative to its location, so it also works from another
working directory. `--output-dir PATH` selects a separate output directory.
Existing outputs in the selected directory are regenerated. Source files stay read only.

## 3. Transaction population and split

One transaction represents **one zone in one hour**, with exactly one categorical
item from each of eight families. Repeated baskets across different hours are valid
observations and are retained. Only duplicate zone-hour keys are rejected.

| Partition | Source period | Source rows | Default mining rows |
|---|---|---:|---:|
| Train | 8 Sep 2025 00:00 to 25 Jan 2026 03:00 | 460,920 | 320,544 |
| Test | 25 Jan 2026 04:00 to 28 Feb 2026 23:00 | 115,368 | 80,256 |
| Total | Retained feature-table period | 576,288 | 400,800 |

All zones at a timestamp belong to the same partition. Test data never contributes
to fitted demand cutoffs, temperature cutoffs, activity selection or cluster profiles.
There is no random split and no bin refitting on the test period.

**Active-zone population.** A zone is active when its mean training pickup count,
excluding the DST merged hour, is at least 1 per hour. This selects 96 zones and
excludes 42 near-empty zones from the default mining population. The active zones
contain 99.66% of retained pickups outside the DST merged hour. Eligibility uses
training counts only, so the count differs from the 40 near-empty zones in full-period EDA.
All normal zero-count hours in the active zones remain in the mining matrices.

**DST.** The 138 rows at 2 Nov 2025 01:00 combine two physical hours. They remain
unchanged in the audit and are excluded from fitting and mining. The row audit gives
DST precedence when a row is also in a sparse zone, so exclusions never double-count.
This stage does not invent separate counts for the two physical hours or halve the count.

The complete `arm_transactions.parquet` retains all 576,288 source rows with
`active_zone`, `is_dst_merged_hour`, `mining_eligible` and `exclusion_reason`.
Sparse-zone alternatives can be studied explicitly from this audit with `encode()`;
their support denominator and tied demand bins must be reported separately.

## 4. Item definitions and boundaries

Item names use `family=value`, such as `time_band=pm_rush` or `demand=High`.

| Family | Values and definition | Role |
|---|---|---|
| `time_band` | `late_night` 00-04, `early_morning` 05-06, `am_rush` 07-09, `midday` 10-15, `pm_rush` 16-19, `evening` 20-23 | Antecedent |
| `day_type` | `weekday` Monday-Friday, `weekend` Saturday-Sunday | Antecedent |
| `holiday` | `federal` when upstream `is_holiday=True`, otherwise `regular` | Antecedent |
| `rain` | `yes` when rain >0 mm, otherwise `no` | Antecedent |
| `snow` | `yes` when snowfall >0 cm, otherwise `no` | Antecedent |
| `temperature` | `lower` <= -0.9 C, `middle` > -0.9 and <=9.9 C, `upper` >9.9 C | Antecedent |
| `zone_cluster` | Training-only `daytime`, `morning_commute`, `nightlife`; `near_empty` for audited excluded zones | Antecedent |
| `demand` | Per-zone training `Low`, `Medium`, `High` as defined below | Consequent only |

Rain and snow can both be `yes` in mixed-precipitation hours. No separate weather-condition
item duplicates these flags. Temperature labels are relative training terciles, not
universal definitions of hot/cold weather. The cutoffs come from active training
zone-hours; unseen colder/warmer test values fall into the outer intervals without clipping.

### Demand classes relative to each zone

For every zone, calculate the 1/3 and 2/3 quantiles of its **training** counts, including
zeros and excluding the DST merged hour. Use pandas' linear quantile interpolation.

- `Low`: `trip_count <= q_low`
- `Medium`: `q_low < trip_count <= q_high`
- `High`: `trip_count > q_high`

The cutoffs are absolute counts for that particular zone, not shared thresholds across
Manhattan and Queens. For example, a count can be High in a quiet zone and Low in a busy
one. The saved `arm_zone_thresholds.csv` makes the exact zone-specific boundaries visible.

Ties stay together. If both quantiles equal zero, zero is Low and positive counts are
High, with no Medium interval. No noise, rank-based assignment or test-period thresholds
force balanced classes. Such sparse/degenerate profiles are audited and excluded by
the default active-zone filter. Fractional cutoffs are valid boundaries for integer counts.

### Training-only zone clusters

Fit the EDA's fixed choice of k=3 using **nonholiday active-zone training rows**:

1. Compute mean pickups for each weekday/hour slot, producing a 168-slot profile per zone.
2. Normalise each profile to sum to 1, so clustering captures timing rather than volume.
3. Fit `KMeans(n_clusters=3, n_init=20, random_state=42)`.
4. Name the centroid with the largest Friday/Saturday-night share `nightlife`.
   Of the other two, name the largest weekday AM-to-midday ratio `morning_commute`;
   the remaining centroid is `daytime`.

The fit gives 57 daytime, 29 morning-commute and 10 nightlife zones. Assignments stay
fixed in test. Cluster names describe profile shape and do not assert a zone's land use.
Profiles, centroids and assignments are exported for inspection and reproduction.

## 5. Outputs and schemas

Outputs are in [`data/processed/association_rules/`](../processed/association_rules/).

| File | Contents |
|---|---|
| `arm_transactions.parquet` | Complete 576,288-row categorical audit with raw descriptive values and eligibility flags |
| `arm_train_rows.parquet`, `arm_test_rows.parquet` | Eligible metadata/category rows, in exact matrix row order |
| `arm_train_onehot.npz`, `arm_test_onehot.npz` | Compressed SciPy CSR Boolean matrices, 320,544 x 24 and 80,256 x 24 |
| `arm_item_catalog.csv` | Fixed column order, item family/value and antecedent/consequent role |
| `arm_preparation_parameters.json` | Frozen temperature cuts, all zone demand cuts/assignments, fit scope and centroids |
| `arm_zone_thresholds.csv` | Training counts/mean/zero share, demand quantiles, tied-bin/activity flags and cluster, plus lookup labels |
| `arm_train_weekly_profiles.csv`, `arm_cluster_centers.csv` | 168-slot training profile evidence |
| `arm_source_split.csv`, `arm_row_audit.csv` | Official split and accounting of eligible/excluded rows and pickups |
| `arm_item_frequencies.csv` | Item counts and support by split, using eligible transaction counts as denominator |
| `arm_demand_class_balance.csv` | Demand-class counts and shares within each cluster and split |
| `arm_preparation_checks.csv` | Saved-file and source-fingerprint validation results |
| `arm_input_manifest.json`, `arm_environment.json` | Input SHA-256 hashes and execution package versions |

Metadata/category row fields:

- **Identity:** `transaction_id` (`zone@YYYY-MM-DDTHH:MM:SS`), `PULocationID`,
  `pickup_hour` (naive New York local time), `split`.
- **Geography:** `Borough`, `Zone`, `service_zone` (area category, not taxi fleet).
- **Observed values:** `trip_count` (pickups), `apparent_temperature` (C), `rain_mm`,
  `snowfall_cm`, `precipitation_probability` (%), `day_of_week` (Mon=0), `hour` (0-23).
- **Eligibility:** Boolean `active_zone`, `is_dst_merged_hour`, `mining_eligible`,
  and `exclusion_reason` (`none`, `dst_merged_hour`, `train_mean_below_1`).
- **Items:** The eight categorical family columns in section 4.

These descriptive fields do not enter the mining matrix. Each matrix row contains
exactly eight True values. All other values are False. The `near_empty` item has zero
support in the default population but remains in the fixed 24-column vocabulary.
The train/test vocabulary and column order are identical.

## 6. Item frequency and class balance

| Item | Train support | Test support |
|---|---:|---:|
| `rain=yes` | 4.57% | 8.10% |
| `snow=yes` | 1.41% | 6.70% |
| `demand=Low` | 37.57% | 41.60% |
| `demand=Medium` | 31.90% | 31.39% |
| `demand=High` | 30.53% | 27.01% |

The imbalance reflects count ties and later-period changes; it is not repaired using
test data. Snow is rare in training: a minimum support above 1.41% would eliminate
every itemset containing `snow=yes`. Combinations involving snow can have still lower
support. The mining member should document the support/confidence thresholds tried
and retain the frozen test partition for #33's stability evaluation.

Readiness charts are in
[`ARM_doc/figures/`](../../ARM_doc/figures/), with a compact
[`ARM_doc/README.md`](../../ARM_doc/README.md) entry point.

## 7. Mining handoff

The loader creates the sparse Boolean pandas format supported by
[mlxtend FP-Growth](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
and [Apriori](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/).
The preparation itself does not require mlxtend. The mining member can install it
in their environment when implementing #31.

```python
import sys
sys.path.insert(0, "scripts")  # run from the repository root
from association_rule_data_preparation import load_transactions

X_train, train_metadata = load_transactions("train")
X_test, test_metadata = load_transactions("test")
assert X_train.index.equals(train_metadata.index)
assert X_train.columns.equals(X_test.columns)

# For a later cluster-specific analysis, keep metadata and baskets aligned:
night_train = X_train.loc[train_metadata.zone_cluster.eq("nightlife")]
```

Mine on `X_train` only. Retain rules with exactly one consequent from
`demand=Low`, `demand=Medium` or `demand=High`, and no `demand=` item in the
antecedent. `arm_item_catalog.csv` makes these roles explicit. Freezing encoding
does not itself enforce that filter in Apriori/FP-Growth: the mining member must apply it.
Discover rules and choose thresholds on training data before recomputing metrics on
`X_test`. Never concatenate the test rows into the discovery population.

Do not add raw `trip_count`, lag/rolling demand, model predictions or residuals to
the antecedents: they would answer a different question or create circular demand rules.
Geographic and optional temporal subsets must report their own support denominator.

## 8. Checks, limitations and interpretation

The saved-file validator checks source fingerprints, full audit coverage, unique IDs,
frozen fit scope, bin boundaries, flags, matrix/row/column alignment, exact encoding,
one item per family, eligibility, zero retention, item frequencies and disjoint chronology.
Unit tests verify holdout independence, DST exclusion, ties, extreme temperatures,
mixed rain/snow, calendar boundaries, unknown-category rejection and Boolean decoding.

- Counts measure completed pickups, not unmet rider demand or the required taxi fleet.
- Rule support counts **zone-hours**, not individual trips, independent people or
  independent weather events. Zones can share weather cells and hourly observations
  are dependent. Association rules describe co-occurrence, not causes.
- `rain=no` is specifically zero in the retained rain field. The collection omitted
  the separate convective-showers variable, so it is not proof of completely dry weather.
- The weather is matched to the pickup hour. This is retrospective condition analysis;
  using it for forecasting would require weather available at the decision time.
- Training/test covers one autumn-winter window. Individual holidays occur once and
  test contains major snow events. High temperature means the upper training tercile,
  not a universal hot-weather category. Test stability may differ from training.
- Full-period EDA cluster/tier files are context only. All learned choices in this
  preparation use training rows. See the official [KMeans reference](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  for the explicit reproducible clustering settings.
- Existing processing/model outputs and other members' branches remain unchanged.
