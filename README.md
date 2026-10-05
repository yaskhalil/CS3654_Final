# Flights Analytics Dashboard

**Final Term Project, Phase 1**
Salma Lahbibi & Yaseen Khalil

An end-to-end Python project that cleans the `flights.csv` dataset (~337,000 flights, 16 carriers), runs statistical analysis and machine learning on it, and serves the results in an interactive Dash dashboard.

For the full technical design (schemas, algorithms, module contracts), see [`SPEC.md`](SPEC.md).

---

## What the project does

| Stage | What happens |
|---|---|
| **Ingest + clean** | Load raw CSV, validate, remove duplicates, fix types, handle missing values (Bronze to Silver) |
| **Transform** | Time encoding, log transforms, standardization, star schema + one-big-table (Silver to Gold) |
| **Statistics** | Q-Q plots, Kolmogorov-Smirnov test, D'Agostino's K-squared test |
| **Feature selection** | Recursive Feature Elimination (RFE), PCA, LDA |
| **Machine learning** | MLP delay classifier, DBSCAN clustering + outlier detection, optional Elastic Net regression |
| **Evaluation** | Confusion matrix, accuracy / precision / recall / F1, K-fold cross-validation, permutation feature importance |
| **Dashboard** | Dash app with filters (airline, origin, destination), min/max selectors, a dropdown, a "How to Use" pop-up, and one tab per analysis |

---

## Project structure

Files marked `(to create)` do not exist yet. Everything in this tree is a file you still need to make, except `README.md` and `SPEC.md`.

```
flights-analytics/
├── README.md                          # this file
├── SPEC.md                            # technical specification
├── requirements.txt                   # (to create) pinned dependencies
├── pyproject.toml                     # (to create) package metadata, tool config
├── .gitignore                         # (to create) ignore data/, models/, __pycache__
├── Makefile                           # (to create) shortcuts: make pipeline, make app, make test
│
├── config/
│   └── config.yaml                    # (to create) paths, seeds, hyperparameters, delay threshold
│
├── data/                              # medallion architecture
│   ├── bronze/
│   │   └── flights.csv                # raw download, NEVER modified
│   ├── silver/
│   │   └── flights_clean.parquet      # (generated) cleaned data
│   └── gold/
│       ├── star/
│       │   ├── fact_flights.parquet   # (generated)
│       │   ├── dim_carrier.parquet    # (generated)
│       │   ├── dim_airport.parquet    # (generated)
│       │   ├── dim_aircraft.parquet   # (generated)
│       │   └── dim_date.parquet       # (generated)
│       └── obt/
│           └── flights_obt.parquet    # (generated) ML-ready one big table
│
├── models/                            # (generated) saved artifacts
│   ├── scaler.joblib
│   ├── encoder.joblib
│   ├── pca.joblib
│   ├── mlp_classifier.joblib
│   └── dbscan_results.parquet
│
├── reports/                           # (generated) metrics and exports
│   ├── data_quality_report.json
│   ├── normality_results.csv
│   ├── classification_metrics.json
│   └── permutation_importance.csv
│
├── src/
│   └── flights/
│       ├── __init__.py                # (to create)
│       ├── config.py                  # (to create) loads config.yaml
│       ├── logging_utils.py           # (to create)
│       │
│       ├── pipeline/
│       │   ├── __init__.py            # (to create)
│       │   ├── bronze.py              # (to create) load raw CSV, record row counts / dtypes
│       │   ├── silver.py              # (to create) cleaning: nulls, dupes, dtypes, sanity checks
│       │   ├── gold_star.py           # (to create) build star schema
│       │   ├── gold_obt.py            # (to create) build one-big-table for ML
│       │   └── run_pipeline.py        # (to create) orchestrates bronze -> silver -> gold
│       │
│       ├── features/
│       │   ├── __init__.py            # (to create)
│       │   ├── transforms.py          # (to create) time encoding, log transform, standardize
│       │   ├── selection.py           # (to create) RFE
│       │   ├── reduction.py           # (to create) PCA, LDA
│       │   └── target.py              # (to create) builds the binary delay label
│       │
│       ├── stats/
│       │   ├── __init__.py            # (to create)
│       │   ├── descriptive.py         # (to create) group summaries (counts, mean delay, etc.)
│       │   └── normality.py           # (to create) Q-Q data, KS test, D'Agostino K^2
│       │
│       ├── models/
│       │   ├── __init__.py            # (to create)
│       │   ├── classifier.py          # (to create) MLP classifier
│       │   ├── clustering.py          # (to create) DBSCAN clusters + outliers
│       │   ├── regression.py          # (to create) optional Elastic Net
│       │   └── evaluation.py          # (to create) metrics, K-fold CV, permutation importance
│       │
│       └── viz/
│           ├── __init__.py            # (to create)
│           ├── numerical.py           # (to create) box, scatter, Q-Q, PCA, DBSCAN, KS plots
│           └── categorical.py         # (to create) bar plots, grouped bar plots
│
├── app/
│   ├── app.py                         # (to create) Dash entry point
│   ├── layout.py                      # (to create) shared wireframe layout
│   ├── callbacks.py                   # (to create) filter + tab callbacks
│   ├── components/
│   │   ├── __init__.py                # (to create)
│   │   ├── filters.py                 # (to create) min/max selector, dropdown, input area
│   │   └── how_to_use.py              # (to create) "How to Use" modal
│   ├── tabs/
│   │   ├── __init__.py                # (to create)
│   │   ├── overview_tab.py            # (to create) categorical bar plots + descriptive stats
│   │   ├── distributions_tab.py       # (to create) box plots, Q-Q, KS, scatter
│   │   ├── pca_tab.py                 # (to create)
│   │   ├── clustering_tab.py          # (to create) DBSCAN scatter + outliers
│   │   └── prediction_tab.py          # (to create) MLP prediction from user input + metrics
│   └── assets/
│       └── style.css                  # (to create)
│
├── scripts/
│   ├── download_data.py               # (to create) places flights.csv in data/bronze/
│   ├── train_models.py                # (to create) trains MLP, DBSCAN, saves artifacts
│   └── generate_reports.py            # (to create) writes reports/
│
├── tests/
│   ├── conftest.py                    # (to create) small sample-data fixture
│   ├── test_silver.py                 # (to create)
│   ├── test_transforms.py             # (to create)
│   ├── test_normality.py              # (to create)
│   ├── test_models.py                 # (to create)
│   └── test_gold_schema.py            # (to create)
│
└── notebooks/
    └── eda.ipynb                      # (to create, optional) exploration scratchpad
```

---

## Setup

```bash
# 1. create and activate a virtual environment
python3 -m venv ds_env && source ds_env/bin/activate && pip install --upgrade pip && pip install numpy pandas kagglehub scikit-learn torch torchvision torchaudio scipy matplotlib
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. put the raw dataset in the Bronze layer
#    data/bronze/flights.csv
```

### Suggested `requirements.txt`

```
pandas
numpy
scipy
scikit-learn
statsmodels
pyarrow
pyyaml
joblib
plotly
dash
dash-bootstrap-components
pytest
```

---

## Running

```bash
# Build Bronze -> Silver -> Gold
python -m flights.pipeline.run_pipeline

# Train models (MLP, DBSCAN) and save artifacts
python scripts/train_models.py

# Write metrics and reports
python scripts/generate_reports.py

# Launch the dashboard
python app/app.py
# then open http://127.0.0.1:8050
```

Run the tests with `pytest`.

---

## Dashboard overview

All tabs share one layout: a **min/max selector**, a **dropdown menu** (airline, airplane, or analysis type), a **"How to Use" pop-up**, a **main display area**, and a **user input area** along the bottom.

| Tab | Contents |
|---|---|
| Overview | Bar plots (flights by carrier / origin / destination), grouped delay comparisons, descriptive stats |
| Distributions | Box-and-whisker, scatter (e.g. `dep_delay` vs `arr_delay`), Q-Q plots, KS test + CDF line plot |
| PCA | PCA scatter plot |
| Clustering | DBSCAN scatter, with noise points marked as outliers |
| Prediction | MLP classifier: enter flight details, get delayed / not delayed, plus the confusion matrix and metrics |

---

## Team

- Salma Lahbibi
- Yaseen Khalil
