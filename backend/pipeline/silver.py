# python3 -m backend.pipeline.silver
import json
import os
import pandas as pd
from backend.logger import logger

BRONZE_PATH = "data/bronze/flights.csv"
SILVER_PATH = "data/silver/flights_clean.csv"
REPORT_PATH = "reports/data_quality_report.json"

DELETE_TIME_COLS = ['year', 'month', 'day', 'hour', 'minute']
CATEGORICAL_COLS = ['carrier', 'flight', 'tailnum', 'origin', 'dest', 'name']
HHMM_COLS = ['dep_time', 'sched_dep_time', 'arr_time', 'sched_arr_time']


def valid_hhmm(s):
    return (s >= 0) & (s <= 2400) & (s % 100 < 60)


def clean(df, report=None):
    report = {} if report is None else report
    report["rows_in"] = len(df)

    # Dropping duplicate rows
    report["duplicate_rows_dropped"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    report["duplicate_ids_dropped"] = int(df["id"].duplicated().sum())
    df = df.drop_duplicates(subset = "id")

    # Handling missing values
    report["null_counts_before"] = {c: int(n) for c, n in df.isnull().sum().items() if n > 0}
    report["missing_value_policy"] = "drop rows with nulls in dep_time, dep_delay, arr_time, arr_delay, air_time (cancelled/diverted flights)"
    before = len(df)
    df = df.dropna(subset = ['dep_time', 'dep_delay', 'arr_time', 'arr_delay', 'air_time'])
    report["missing_rows_dropped"] = before - len(df)

    before = len(df)
    df = df[(df["distance"] > 0) & (df["air_time"] > 0)]
    for col in HHMM_COLS:
        df = df[valid_hhmm(df[col])]
    report["sanity_check_rows_dropped"] = before - len(df)

    df = df.drop(columns = DELETE_TIME_COLS)

    df["time_hour"] = pd.to_datetime(df["time_hour"])
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).astype("category")
    for col in HHMM_COLS:
        df[col] = df[col].astype(int)

    report["rows_out"] = len(df)
    for step, n in report.items():
        logger.info(f"{step}: {n}")
    return df.reset_index(drop = True)


def write_silver(df, path = SILVER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    df.to_csv(path, index = False)


def write_report(report, path = REPORT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "w") as f:
        json.dump({"silver": report}, f, indent = 2)


if __name__ == "__main__":
    df_raw = pd.read_csv(BRONZE_PATH)
    report = {}
    df = clean(df_raw, report)
    write_silver(df)
    write_report(report)
    logger.info("Silver layer created successfully.")
