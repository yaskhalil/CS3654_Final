import os
import pandas as pd
from backend.logger import logger

SILVER_PATH = "data/silver/flights_clean.csv"
STAR_DIR = "data/gold/star/"

FACT_COLS = ['id', 'carrier', 'tailnum', 'origin', 'dest', 'time_hour',
             'dep_time', 'sched_dep_time', 'dep_delay',
             'arr_time', 'sched_arr_time', 'arr_delay',
             'air_time', 'distance', 'flight']

def build_dim_carrier(df):
    """One row per carrier with its full airline name."""
    return df[['carrier', 'name']].drop_duplicates().sort_values('carrier').reset_index(drop = True)


def build_dim_airport(df):
    """One row per airport code, covering both origin and dest so the fact table can join to it twice."""
    codes = pd.concat([df['origin'], df['dest']]).drop_duplicates().sort_values()
    return pd.DataFrame({'airport_code': codes.values})


def build_dim_aircraft(df):
    """One row per aircraft (tailnum)."""
    return df[['tailnum']].drop_duplicates().sort_values('tailnum').reset_index(drop = True)

def build_dim_date(df):
    """One row per scheduled departure hour (time_hour) with calendar parts for grouping in the dashboard."""
    t = pd.Series(pd.to_datetime(df['time_hour']).unique()).sort_values()
    return pd.DataFrame({
        'time_hour': t.values,
        'date': t.dt.date.values,
        'year': t.dt.year.values,
        'month': t.dt.month.values,
        'day': t.dt.day.values,
        'hour': t.dt.hour.values,
        'day_of_week': t.dt.dayofweek.values,
        'day_name': t.dt.day_name().values,
        'is_weekend': (t.dt.dayofweek >= 5).values,
    })

def build_fact_flights(df):
    """One row per flight (id). Keys point to the dims; name is dropped since it lives in dim_carrier."""
    fact = df[FACT_COLS].copy()
    fact['time_hour'] = pd.to_datetime(fact['time_hour'])
    return fact


def build_star(df):
    tables = {
        'dim_carrier': build_dim_carrier(df),
        'dim_airport': build_dim_airport(df),
        'dim_aircraft': build_dim_aircraft(df),
        'dim_date': build_dim_date(df),
        'fact_flights': build_fact_flights(df),
    }
    for name, table in tables.items():
        logger.info(f"{name}: {table.shape}")
    return tables

def check_star(tables):
    """Dim keys must be unique and every fact key must have a match in its dim."""
    fact = tables['fact_flights']
    keys = [('dim_carrier', 'carrier', ['carrier']),
            ('dim_airport', 'airport_code', ['origin', 'dest']),
            ('dim_aircraft', 'tailnum', ['tailnum']),
            ('dim_date', 'time_hour', ['time_hour'])]

    assert fact['id'].is_unique, "fact_flights: duplicate ids"
    for dim, key, fact_cols in keys:
        assert tables[dim][key].is_unique, f"{dim}: duplicate {key}"
        for col in fact_cols:
            missing = ~fact[col].isin(tables[dim][key])
            assert not missing.any(), f"fact_flights.{col}: {missing.sum()} rows with no match in {dim}"
    logger.info("Star schema checks passed")


def write_star(tables, out_dir = STAR_DIR):
    os.makedirs(out_dir, exist_ok = True)
    for name, table in tables.items():
        table.to_csv(os.path.join(out_dir, f"{name}.csv"), index = False)


if __name__ == "__main__":
    df_silver = pd.read_csv(SILVER_PATH)
    logger.info(f"Silver data imported: {df_silver.shape}")
    tables = build_star(df_silver)
    check_star(tables)
    write_star(tables)
    logger.info("Gold star schema created successfully.")
