import pandas as pd
import numpy as np
import os
from backend.logger import logger

# Dropping uneeded columns
df_raw = pd.read_csv("data/bronze/flights.csv")
# --------------- transforming the data ---------------
## Droppping rows
df = df_raw.drop(columns = ['year', 'month','day','hour','minute'])
logger.info(f"Duplicate rows found: {df.duplicated().sum()}")

## Handling missing values
df = df.dropna(subset = "arr_delay")

## Setting types
df["time_hour"] = pd.to_datetime(df["time_hour"])
df["carrier"] = df["carrier"].astype("category")
df["flight"] = df["flight"].astype(str)
df["dep_time"] = df["dep_time"].astype(int)

os.makedirs("data/silver", exist_ok = True)
df.to_parquet("data/silver/flights_clean.parquet", index = False)
logger.info("Silver layer created successfully.")
    