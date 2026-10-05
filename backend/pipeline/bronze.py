"""
This file will handle the download of Flights dataset from kaggle.
Then save it to the bronze layer of the data folder.
"""
from pathlib import Path
import pandas as pd
import kagglehub
from backend.logger import logger

# Download latest version (returns the folder the dataset was downloaded to)
path = kagglehub.dataset_download("mahoora00135/flights")

print("Dataset status: ", path)
df_raw = pd.read_csv(Path(path) / "flights.csv")
df_raw.to_csv("data/bronze/flights.csv", index=False)
logger.info("Bronze layer created successfully.")
