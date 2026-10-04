# Technical Specification: Flights Analytics Project

**Version:** Phase 1 draft
**Authors:** Salma Lahbibi & Yaseen Khalil
**Language:** Python 3.10+
**Dataset:** `flights.csv` (~337,000 rows, 21 columns, 16 carriers)

---

## 1. Goals

1. Build a reproducible Python pipeline using the **medallion architecture** (Bronze / Silver / Gold).
2. Perform statistical analysis (descriptive stats, normality testing).
3. Perform feature selection and dimensionality reduction (RFE, PCA, LDA).
4. Train and evaluate models: **MLP classification**, **DBSCAN clustering + outlier detection**, and optionally **Elastic Net regression**.
5. Present everything in an interactive **Dash** dashboard.

---

## 2. Dataset

### 2.1 Features (21)

**Categorical (6):** `carrier`, `flight`, `tailnum`, `origin`, `dest`, `name`

**Numerical (15):** `id`, `year`, `month`, `day`, `hour`, `minute`, `dep_time`, `sched_dep_time`, `dep_delay`, `arr_time`, `sched_arr_time`, `arr_delay`, `air_time`, `distance`, `time_hour`

> `time_hour` is a timestamp (`yyyy-mm-dd hh:mm:ss`, local scheduled departure). It is listed as numerical in the proposal but is parsed as `datetime64` in code.

### 2.2 Key variables

- `dep_delay`, `arr_delay`: central to analysis and the classification target.
- `distance`, `air_time`, `time_hour`: main numeric predictors.
- `flight`, `tailnum`: **high cardinality**; evaluated before inclusion (see 5.3).

---

## 3. Architecture

```
            ┌────────────┐     ┌────────────┐     ┌──────────────────────┐
flights.csv │   BRONZE   │ ──▶ │   SILVER   │ ──▶ │         GOLD         │
            │ raw, as-is │     │  cleaned   │     │ star schema │  OBT   │
            └────────────┘     └────────────┘     └──────────────────────┘
                                                         │          │
                                                  analytics/BI   ML models
                                                         └────┬─────┘
                                                              ▼
                                                       Dash dashboard
```

| Layer | Contents | Format | Produced by |
|---|---|---|---|
| Bronze | Original `flights.csv`, unmodified | CSV | `scripts/download_data.py`, `pipeline/bronze.py` |
| Silver | Cleaned data, correct dtypes, no duplicates, redundant time columns removed | Parquet | `pipeline/silver.py` |
| Gold (star) | Fact + dimension tables for querying and the dashboard | Parquet | `pipeline/gold_star.py` |
| Gold (OBT) | One flat, scaled/encoded table for ML | Parquet | `pipeline/gold_obt.py` |

**Rule:** each layer is built only from the layer before it. Bronze is never edited.

---

## 4. Module specifications

### 4.1 `config.py` and `config/config.yaml`

Single source of truth for paths, random seed, and hyperparameters.

```yaml
paths:
  bronze: data/bronze/flights.csv
  silver: data/silver/flights_clean.parquet
  gold_star: data/gold/star/
  gold_obt: data/gold/obt/flights_obt.parquet
  models: models/
  reports: reports/

random_seed: 42

target:
  column: arr_delay        # see 5.4
  threshold_minutes: 0     # delay > threshold => "late"

features:
  drop_after_time_hour: [year, month, day, hour, minute]
  max_categorical_cardinality: 50   # above this, flight/tailnum are excluded or encoded differently

mlp:
  hidden_layer_sizes: [64, 32]
  activation: relu
  max_iter: 200
  early_stopping: true

dbscan:
  eps: 0.5
  min_samples: 10

cv:
  n_splits: 5

dashboard:
  host: 127.0.0.1
  port: 8050
```

### 4.2 `pipeline/bronze.py`

- `load_raw(path) -> DataFrame`: read with pandas, no modification.
- `profile_raw(df) -> dict`: row count, dtypes, null counts, duplicate count. Written to `reports/data_quality_report.json`.

### 4.3 `pipeline/silver.py`

- `clean(df) -> DataFrame`
  1. Drop exact duplicate rows (and duplicate `id`s).
  2. Enforce dtypes (categoricals as `category`, times as numeric, `time_hour` via `pd.to_datetime`).
  3. Handle missing values. Cancelled flights typically have null `dep_time`/`arr_delay`/`air_time`; the policy (drop vs. impute) must be documented in the data quality report.
  4. Sanity checks: `distance > 0`, `air_time > 0`, times within valid HHMM range.
  5. Drop `year, month, day, hour, minute` (redundant with `time_hour`).
- `write_silver(df, path)`

### 4.4 `features/transforms.py`

| Function | Purpose |
|---|---|
| `encode_time(df)` | Derives hour-of-day, day-of-week, month from `time_hour`; applies cyclical sin/cos encoding |
| `log_transform(df, cols)` | `log1p` on right-skewed columns (e.g., `distance`, `air_time`). Delay columns can be negative, so use a signed log or shift first. |
| `standardize(df, cols)` | `StandardScaler`; scaler is **fit on training data only** and saved to `models/scaler.joblib` |
| `encode_categoricals(df, cols)` | One-hot for low-cardinality columns (`carrier`, `origin`); other strategy for high-cardinality columns |

### 4.5 `pipeline/gold_star.py`: star schema

```
                 dim_date
                    │
 dim_carrier ── fact_flights ── dim_aircraft
                    │
              dim_airport (joined as origin and dest)
```

| Table | Grain / key | Key columns |
|---|---|---|
| `fact_flights` | one row per flight (`id`) | FKs to all dims, `dep_delay`, `arr_delay`, `air_time`, `distance`, scheduled/actual times |
| `dim_carrier` | `carrier` | `carrier`, `name` |
| `dim_airport` | airport code | `airport_code` (used as origin and dest) |
| `dim_aircraft` | `tailnum` | `tailnum` |
| `dim_date` | `time_hour` | date, hour, day-of-week, month, etc. |

Purpose: keep dimensions separate for fast filtering and grouping in the dashboard.

### 4.6 `pipeline/gold_obt.py`: one big table

Flattens everything into a single ML-ready table:
1. Start from Silver.
2. Apply `transforms.py` (time encoding, log, standardize, encode).
3. Add the binary target (`features/target.py`).
4. Apply feature selection (RFE) and optionally reduction (PCA/LDA) outputs.
5. Write to `data/gold/obt/flights_obt.parquet`.

### 4.7 `stats/normality.py`

Three complementary checks, per numeric variable:

1. **Q-Q plot**: `scipy.stats.probplot` returns theoretical vs. observed quantiles for plotting.
2. **Kolmogorov-Smirnov**: `scipy.stats.kstest(standardized_x, "norm")`. Also return the empirical CDF and reference CDF for the line plot.
3. **D'Agostino's K²**: `scipy.stats.normaltest`, which tests skewness and kurtosis.

> With ~337k rows, even trivial departures from normality give p ≈ 0. Interpretation must combine **test results and the Q-Q plot**; do not rely on p-values alone. Output goes to `reports/normality_results.csv`.

### 4.8 `features/selection.py` and `features/reduction.py`

- **RFE**: `sklearn.feature_selection.RFE` with a simple estimator (e.g., logistic regression or random forest) to rank features against the delay target. Returns selected feature names + ranking.
- **PCA**: fit on the standardized numeric features; return components, explained variance ratio, and a 2D projection for the scatter plot. Saved to `models/pca.joblib`.
- **LDA**: supervised reduction against the delay label; used for the OBT.

### 4.9 `models/classifier.py`: MLP

- `sklearn.neural_network.MLPClassifier`.
- Input: standardized numeric features + encoded categoricals.
- Output: binary label (delayed / not delayed), plus predicted probability.
- Rationale: delay behavior is driven by nonlinear interactions between time, carrier, and distance, so an MLP is used over logistic regression.
- Exposes `predict_from_user_input(dict) -> (label, probability)` for the dashboard's Prediction tab. This must apply the **same saved scaler/encoder** as training.

### 4.10 `models/clustering.py`: DBSCAN

- Runs on the engineered Gold feature set (typically the PCA-reduced space, for speed and for meaningful distances).
- Parameters: `eps`, `min_samples`.
- Output: cluster label per row; label `-1` = noise = **outlier**.
- Tuning aid: k-distance plot to choose `eps`.
- Performance note: DBSCAN on 337k rows can be slow and memory-heavy. Prefer reduced dimensions and/or a sample for tuning.

### 4.11 `models/regression.py` (optional)

- `sklearn.linear_model.ElasticNetCV` (L1 + L2) to predict a continuous delay (e.g., `arr_delay`).
- Report coefficients, which are useful for showing which features drive delay.

### 4.12 `models/evaluation.py`

- **Confusion matrix**, plus **accuracy, precision, recall, F1** (`sklearn.metrics`).
- **K-fold cross-validation** (`StratifiedKFold`, shuffled, `n_splits` from config) so results do not hinge on a single split.
- **Permutation feature importance** (`sklearn.inspection.permutation_importance`): a model-agnostic ranking, computed on held-out data.
- Outputs: `reports/classification_metrics.json`, `reports/permutation_importance.csv`.

### 4.13 `stats/descriptive.py`

Group-level summaries (by carrier, origin, dest): flight count, mean `dep_delay`/`arr_delay`, mean `air_time`, mean `distance`. Displayed next to the categorical charts.

---

## 5. Design decisions to settle early

These are gaps or risks in the current proposal that are worth deciding before coding.

### 5.1 Data leakage
If the target is derived from `arr_delay` or `dep_delay`, those columns (and anything computed from them) **must not be model inputs**. Also, `arr_time` and `air_time` are only known after departure/arrival. If the classifier is meant to predict delay for **future** flights from user input, use only information available **before departure** (carrier, origin, dest, scheduled times, distance, month/hour). Decide this explicitly.

### 5.2 Train/test split before fitting transforms
Fit scalers, encoders, RFE, and PCA on the **training split only**, then apply to test. Otherwise scores are optimistic. The OBT should therefore store unscaled features plus a split column, or the transforms should live inside an sklearn `Pipeline`.

### 5.3 High-cardinality categoricals
`flight` and `tailnum` have thousands of unique values. Options: drop, frequency-encode, target-encode (inside CV folds only), or keep only the top-N and bucket the rest as "other". Document the choice and justification.

### 5.4 Defining "late"
The proposal says classify "late or early". Pick one definition and put it in `config.yaml`:
- `arr_delay > 0` (any lateness), or
- `arr_delay > 15` (the standard on-time threshold), or
- `dep_delay > 0`.

Check class balance after choosing; if imbalanced, report precision/recall/F1 (not just accuracy) and consider class weights or resampling.

### 5.5 Missing values
Cancelled flights have null delay and time fields. Decide whether to drop them (usual for delay modeling) or model them separately. Record counts in the data quality report.

### 5.6 LDA
LDA appears in the Gold/OBT description but not in the feature-selection section of the proposal. Either include it explicitly (4.8) or remove it from the text.

### 5.7 "Randomization"
The proposal lists "randomization" as a modeling task; this spec implements it as shuffled K-fold cross-validation and permutation importance. Confirm this matches the course requirement.

### 5.8 Dashboard performance
337k points will not render well as raw scatter plots. Sample (e.g., 5-10k points) or use `plotly` `scattergl` / datashader-style aggregation. Precompute heavy results (PCA, DBSCAN labels, normality tests) offline and load them in the app; do not train models inside callbacks.

---

## 6. Dashboard specification

### 6.1 Layout (shared across tabs)

Matches the wireframe:

```
┌───────────────┬──────────────────────────┬────────────────┐
│ Min/Max       │      Drop down menu      │  How to Use    │
│ selector      │ (airline/airplane/type)  │  pop-up        │
├───────────────┴──────────────────────────┴────────────────┤
│                                                           │
│                Display of selected model                  │
│                                                           │
├───────────────────────────────────────────────────────────┤
│                     USER INPUT AREA                       │
└───────────────────────────────────────────────────────────┘
```

### 6.2 Components

| Component | Dash element | Behavior |
|---|---|---|
| Min/Max selector | `dcc.RangeSlider` or two `dcc.Input` | Filters rows by a chosen numeric variable |
| Dropdown | `dcc.Dropdown` | Select airline, airplane, or analysis type |
| How to Use | `dbc.Modal` + button | Instructions for filters and tabs |
| Display area | `dcc.Graph` / `html.Div` | Updated by callbacks |
| User input area | `dcc.Input`, `dcc.Dropdown`, button | Filters; flight details for MLP predictions |

### 6.3 Tabs and visualizations

**Numerical:** box-and-whisker, scatter (`dep_delay` vs `arr_delay`), Q-Q plots, PCA scatter, DBSCAN scatter (noise highlighted), KS test results with a line plot comparing empirical vs. reference distribution.

**Categorical:** bar plots of flights by carrier / origin / destination, selected flight numbers, grouped bar plots comparing average delays across airlines or airports. Descriptive stats shown alongside.

### 6.4 Callback rules
- Callbacks read from precomputed Gold/report files, not raw CSV.
- Filters (airline, origin, dest, min/max) apply to the data before any plot is built.
- The Prediction tab loads the saved MLP, scaler, and encoder once at startup.

---

## 7. Execution flow

```
download_data.py
      │
      ▼
run_pipeline.py ── bronze.py ─▶ silver.py ─▶ gold_star.py
                                         └──▶ gold_obt.py
      │
      ▼
train_models.py ── selection (RFE) ─▶ reduction (PCA/LDA)
                └─ classifier (MLP) ─▶ evaluation (CV, permutation importance)
                └─ clustering (DBSCAN)
                └─ regression (Elastic Net, optional)
      │
      ▼
generate_reports.py ─▶ reports/*.json, *.csv
      │
      ▼
app/app.py  (Dash)
```

---

## 8. Testing plan

| Test file | Verifies |
|---|---|
| `test_silver.py` | No duplicates, no nulls in required columns, correct dtypes, redundant columns removed |
| `test_transforms.py` | Scaler fit only on train; log transform handles zero/negative values; time encoding ranges |
| `test_normality.py` | Known normal sample passes; known skewed sample fails; output shapes correct |
| `test_gold_schema.py` | Star-schema keys unique in dims; every fact FK has a match; OBT has target and no leaked columns |
| `test_models.py` | MLP trains on a tiny sample and predicts valid labels; DBSCAN returns `-1` label for obvious outliers |

`tests/conftest.py` provides a small (~500 row) sample fixture so tests run fast.

---

## 9. Dependencies

`pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `pyarrow`, `pyyaml`, `joblib`, `plotly`, `dash`, `dash-bootstrap-components`, `pytest`.

---

## 10. Suggested build order

1. `config.py`, `config.yaml`, folder skeleton
2. `bronze.py` and `silver.py`, then write the data quality report
3. `transforms.py` and `gold_star.py` / `gold_obt.py`
4. `normality.py` and `descriptive.py`
5. `selection.py`, `reduction.py`
6. `classifier.py` and `evaluation.py`
7. `clustering.py` (and `regression.py` if time allows)
8. `viz/` plotting functions
9. Dash app: layout first, then one tab at a time
10. Tests and final report generation
