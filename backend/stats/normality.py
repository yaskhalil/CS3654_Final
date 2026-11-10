# python3 -m backend.stats.normality
"""
Normality checks: Q-Q plot, Kolmogorov-Smirnov, D'Agostino's K^2.

Run on Silver data before and after log_transform to justify the log step.
With ~327k rows every p-value is ~0, so compare skew, KS D and Q-Q r instead.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from backend.logger import logger
from backend.features.transform import log_transform, LOG_COLS, SIGNED_LOG_COLS

SILVER_PATH = "data/silver/flights_clean.csv"
RESULTS_PATH = "reports/normality_results.csv"
FIG_PATH = "backend/stats/figures/normality_qq_slide.png"
COLS = SIGNED_LOG_COLS + LOG_COLS     # dep_delay, arr_delay, distance, air_time
COLOR = "#E4572E"


def check_column(x):
    """Skew, kurtosis, KS, D'Agostino and Q-Q r for one column."""
    x = x.dropna()
    z = (x - x.mean()) / x.std()                          # KS compares against N(0, 1)
    ks_D, ks_p = stats.kstest(z, "norm")
    _, k2_p = stats.normaltest(x)
    _, (_, _, qq_r) = stats.probplot(x, dist = "norm")
    return {"skew": stats.skew(x), "kurtosis": stats.kurtosis(x),   # both 0 for a normal curve
            "ks_D": ks_D, "ks_p": ks_p, "k2_p": k2_p, "qq_r": qq_r}  # qq_r = 1 is a straight line


def run_normality(df, stage, cols = COLS):
    """One row of results per column."""
    rows = [{"column": col, "stage": stage, **check_column(df[col])} for col in cols]
    logger.info(f"Normality checks done ({stage})")
    return pd.DataFrame(rows)


def plot_qq(df, cols, titles, path):
    """Side-by-side Q-Q plots; points on the line = normal."""
    fig, axes = plt.subplots(1, len(cols), figsize = (10, 5))
    for ax, col, title in zip(axes, cols, titles):
        (theory, observed), (slope, intercept, r) = stats.probplot(df[col].dropna(), dist = "norm")
        step = max(len(theory) // 2000, 1)                # plot ~2,000 points, not 327k
        ax.scatter(theory[::step], observed[::step], s = 8, color = COLOR, alpha = 0.6)
        ends = np.array([theory.min(), theory.max()])
        ax.plot(ends, slope * ends + intercept, color = "black", lw = 1.2)
        ax.set_title(f"{col}: {title}\nQ-Q r = {r:.3f}", loc = "left", fontsize = 15, fontweight = "bold")
        ax.set_xlabel("Theoretical normal quantiles", fontsize = 12)
        ax.set_ylabel("Observed (after log)", fontsize = 12)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok = True)
    fig.savefig(path, dpi = 200)
    plt.close(fig)
    logger.info(f"Saved {path}")


if __name__ == "__main__":
    df = pd.read_csv(SILVER_PATH)
    logger.info(f"Silver data imported: {df.shape}")
    df_log = log_transform(df)

    results = pd.concat([run_normality(df, "raw"), run_normality(df_log, "log")], ignore_index = True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok = True)
    results.to_csv(RESULTS_PATH, index = False)
    logger.info(f"Saved {RESULTS_PATH}")

    plot_qq(df_log, ["air_time", "arr_delay"], ["much closer to normal", "two humps"], FIG_PATH)
    print(results.round(3).to_string(index = False))
