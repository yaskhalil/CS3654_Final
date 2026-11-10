# python3 -m backend.pipeline.gold_obt
import numpy as np
import pandas as pd
import os

from backend.logger import logger

# Import dataframe
silver = pd.read_csv('../data/silver/flights_clean.csv')
df_silver = pd.DataFrame(silver)
logger.info("Silver Data Imported.")

