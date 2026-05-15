"""
Shared configuration: analysis thresholds, data paths, and matplotlib style.
Import this at the top of every analysis script.

Paths here point to the synthetic dataset in synthetic_data/.
To use real data, replace DATA_PARQUET and SNOW_SESSIONS_JSON with
paths to the actual TELUS mobility parquet and snow session JSON.
"""
from pathlib import Path
import matplotlib.pyplot as plt

MIN_SNOW_CM      = 5.0
ANY_SNOW_CM      = 1.0
LEAD_DAYS        = 1      # pre/post window; paper uses 1 and 2
BUCKET_HOURS     = 12

_ROOT              = Path(__file__).parent
DATA_PARQUET       = _ROOT / "synthetic_data" / "od_mobility_synthetic.parquet"
SNOW_SESSIONS_JSON = _ROOT / "synthetic_data" / "snow_sessions_synthetic.json"
VKT_PARQUET        = DATA_PARQUET

MOBILITY_COLS = [
    "bucket_start", "weather_label", "count", "baseline",
    "has_snow", "snow_on_ground_cm", "total_snow_cm",
]

VKT_COLS = [
    "bucket_start", "count", "baseline", "total_distance_km",
    "motorway_km", "trunk_km", "primary_km", "secondary_km",
    "tertiary_km", "residential_km", "unclassified_km",
]

BG        = "#ffffff"
FG        = "#1a1a1a"
FG_SUBTLE = "#666666"
FG_TITLE  = "#111111"
GRID      = "#eeeeee"
SPINE     = "#cccccc"

C_BLUE       = "#0072B2"
C_ORANGE     = "#E69F00"
C_GREEN      = "#009E73"
C_PINK       = "#CC79A7"
C_CYAN       = "#56B4E9"
C_VERMILLION = "#D55E00"
C_YELLOW     = "#F0E442"

C_MEAN        = C_BLUE
C_CI          = C_BLUE
C_EVENT_LINE  = "#9ca3af"
C_ZONE_PRE    = "#DBEAFE"
C_ZONE_DURING = "#FEE2E2"
C_ZONE_POST   = "#D1FAE5"

PALETTE = [C_BLUE, C_ORANGE, C_GREEN, C_PINK, C_CYAN, C_VERMILLION]


def apply_plot_style():
    """Apply shared rcParams. Call once per script before plotting."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   BG,
        "axes.edgecolor":   SPINE,
        "axes.labelcolor":  FG,
        "text.color":       FG,
        "xtick.color":      FG_SUBTLE,
        "ytick.color":      FG_SUBTLE,
        "grid.color":       GRID,
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    })


def style_axes(ax):
    """Apply consistent spine + tick styling to an Axes."""
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.grid(True, alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)
