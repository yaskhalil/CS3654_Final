import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from backend.logger import logger
from backend.features.transform import log_transform

SILVER_PATH = "data/silver/flights_clean.csv"
FIG_DIR = "backend/metrics/figures"
COL = "arr_delay"

BEFORE_COLOR = "#6E6E6E"
AFTER_COLOR = "#E4572E"


def plot_hist(series, title, xlabel, path, color, xlim = None, **hist_kwargs):
    fig, ax = plt.subplots(figsize = (7, 4.5))
    sns.histplot(series, kde = True, color = color, edgecolor = "white", linewidth = 0.3, ax = ax, **hist_kwargs)
    ax.set_title(f"{title}   (skew {series.skew():.2f})", loc = "left", fontsize = 13, fontweight = "bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Flights")
    if xlim:
        ax.set_xlim(*xlim)
    sns.despine()
    fig.tight_layout()
    fig.savefig(path, dpi = 200)
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    df = pd.read_csv(SILVER_PATH)
    logger.info(f"Silver data imported: {df.shape}")
    df_log = log_transform(df)

    sns.set_theme(style = "white")
    os.makedirs(FIG_DIR, exist_ok = True)

    plot_hist(df[COL], "Before: arr_delay", "Arrival delay (minutes)",
              os.path.join(FIG_DIR, "arr_delay_before.png"), BEFORE_COLOR,
              xlim = (-80, 300), binwidth = 5)
    plot_hist(df_log[COL], "After: signed log of arr_delay", "sign(x) · log1p(|x|)",
              os.path.join(FIG_DIR, "arr_delay_after.png"), AFTER_COLOR,
              bins = 60)
