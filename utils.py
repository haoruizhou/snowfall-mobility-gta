"""
Core data-loading and analysis functions shared across all scripts.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from config import (
    DATA_PARQUET, SNOW_SESSIONS_JSON, MOBILITY_COLS,
    MIN_SNOW_CM, BUCKET_HOURS,
)


# DATA LOADING

def load_data(
    parquet_path: Path = DATA_PARQUET,
    sessions_json: Path = SNOW_SESSIONS_JSON,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load the mobility parquet and build the per-day event_ranges DataFrame.

    Returns
    -------
    df : pl.DataFrame
        Full mobility dataset with bucket-level OD rows.
    event_ranges : pl.DataFrame
        One row per snow-session day with columns:
        event_id, event_date, event_start, event_end,
        original_session_id, consecutive_day_index,
        total_storm_days, daily_snow_cm.
    """
    df = pl.read_parquet(str(parquet_path), columns=MOBILITY_COLS)
    df = df.with_columns([
        pl.col("weather_label").cast(pl.Utf8),
        pl.col("count").cast(pl.Int16),
    ]).sort("bucket_start")

    with open(str(sessions_json), "r", encoding="utf-8") as f:
        session_data = json.load(f)

    event_ranges = _build_event_ranges(session_data, df)
    return df, event_ranges


def _build_event_ranges(session_data: dict, df: pl.DataFrame) -> pl.DataFrame:
    """Decompose sessions into uniform 1-day events and attach daily_snow_cm."""
    rows = []
    eid = 1
    for session in session_data["sessions"]:
        sid       = session["session_number"]
        start     = datetime.fromisoformat(session["start_date"])
        end       = datetime.fromisoformat(session["end_date"])
        n_days    = int(session["duration_days"])
        current   = start
        day_index = 1
        while current <= end:
            rows.append({
                "event_id":              eid,
                "event_date":            current,
                "event_start":           current,
                "event_end":             current + timedelta(days=1),
                "original_session_id":   sid,
                "consecutive_day_index": day_index,
                "total_storm_days":      n_days,
            })
            eid       += 1
            day_index += 1
            current   += timedelta(days=1)

    event_ranges = pl.DataFrame(rows)

    # Attach per-day total_snow_cm from the parquet
    daily_snow = (
        df.filter(pl.col("total_snow_cm").is_not_null())
        .with_columns(pl.col("bucket_start").dt.truncate("1d").alias("day"))
        .group_by("day")
        .agg(pl.col("total_snow_cm").first().alias("daily_snow_cm"))
    )
    event_ranges = (
        event_ranges
        .with_columns(pl.col("event_date").dt.truncate("1d").alias("day"))
        .join(daily_snow, on="day", how="left")
        .with_columns(pl.col("daily_snow_cm").fill_null(0.0))
        .drop("day")
    )
    return event_ranges


# DEVIATION COMPUTATION

def compute_daily_deviation(df: pl.DataFrame, target_date: datetime) -> Optional[float]:
    """
    Averaged daily mobility deviation.

    1. Filter to the calendar day.
    2. Aggregate all OD pairs within each 12 h bucket → (raw, baseline).
    3. Compute deviation per bucket: (raw − baseline) / baseline.
    4. Return the mean across buckets — one scalar per day.
    """
    day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end   = day_start + timedelta(days=1)
    day_df    = df.filter(
        (pl.col("bucket_start") >= day_start) &
        (pl.col("bucket_start") <  day_end)
    )
    if day_df.height == 0:
        return None

    bucket_agg = (
        day_df
        .group_by("bucket_start")
        .agg([
            pl.col("count").cast(pl.Int64).sum().alias("raw"),
            pl.col("baseline").sum().alias("base"),
        ])
        .with_columns(
            pl.when(pl.col("base") > 0)
            .then((pl.col("raw") - pl.col("base")) / pl.col("base"))
            .otherwise(None)
            .alias("dev")
        )
        .filter(pl.col("dev").is_not_null())
    )
    if bucket_agg.height == 0:
        return None
    return float(bucket_agg["dev"].mean())


def compute_window_deviation(
    df: pl.DataFrame,
    anchor_date: datetime,
    window_days: int,
    direction: str,   # "pre" or "post"
) -> Optional[float]:
    """
    Average daily deviation over window_days days before or after anchor_date.
    direction="pre"  → days anchor-1, anchor-2, …, anchor-window_days
    direction="post" → days anchor+1, anchor+2, …, anchor+window_days
    """
    devs = []
    for i in range(1, window_days + 1):
        d = anchor_date - timedelta(days=i) if direction == "pre" \
            else anchor_date + timedelta(days=i)
        v = compute_daily_deviation(df, d)
        if v is not None:
            devs.append(v)
    return float(np.mean(devs)) if devs else None


def calculate_mobility_deviation_for_window(
    df: pl.DataFrame, window_df: pl.DataFrame,
    total_snow_threshold: float = 1.0,
) -> pl.DataFrame:
    """
    Per-12h-bucket mobility deviation for a time window (used in story plot).
    Kept for compatibility with narrative / story-plot scripts.
    """
    agg = (
        window_df
        .group_by("bucket_start")
        .agg([
            pl.col("count").sum().alias("raw_count"),
            pl.col("baseline").sum().alias("baseline"),
            ((pl.col("weather_label") == "Snow") &
             (pl.col("total_snow_cm") > total_snow_threshold)).any().alias("has_snow"),
        ])
        .sort("bucket_start")
    )
    return agg.with_columns([
        pl.when(pl.col("baseline") > 0)
        .then((pl.col("raw_count") - pl.col("baseline")) / pl.col("baseline"))
        .otherwise(None)
        .alias("mobility_deviation")
    ])


# SESSION-LEVEL COLLECTION

def collect_session_events(
    event_ranges: pl.DataFrame,
    df: pl.DataFrame,
    min_snow_cm: float = MIN_SNOW_CM,
    lead_days: int = 1,
) -> pl.DataFrame:
    """
    Build session-level pre / during / post mobility deviations.

    For each session where ALL days have >= min_snow_cm:
      - pre:    averaged daily deviation for lead_days days before session start
      - during: mean of per-day averaged deviations across all session days
      - post:   averaged daily deviation for lead_days days after session end

    Returns one row per qualifying session.
    """
    results = []
    n_filtered = n_no_boundary = 0

    for sid in sorted(event_ranges["original_session_id"].unique().to_list()):
        session_events = event_ranges.filter(
            pl.col("original_session_id") == sid
        ).sort("event_date")

        if session_events.height == 0:
            continue
        if session_events["daily_snow_cm"].min() < min_snow_cm:
            n_filtered += 1
            continue

        n_days        = int(session_events["total_storm_days"][0])
        session_start = session_events["event_date"].min()
        session_end   = session_events["event_date"].max()

        day_devs = [
            v for ev in session_events.iter_rows(named=True)
            if (v := compute_daily_deviation(df, ev["event_date"])) is not None
        ]
        if not day_devs:
            n_no_boundary += 1
            continue

        pre_dev  = compute_window_deviation(df, session_start, lead_days, "pre")
        post_dev = compute_window_deviation(df, session_end,   lead_days, "post")

        if pre_dev is None or post_dev is None:
            n_no_boundary += 1
            continue

        results.append({
            "session_id":    sid,
            "session_start": session_start,
            "session_end":   session_end,
            "n_days":        n_days,
            "avg_snow_cm":   float(session_events["daily_snow_cm"].mean()),
            "pre_dev":       pre_dev,
            "during_dev":    float(np.mean(day_devs)),
            "post_dev":      post_dev,
        })

    print(f"  Qualifying sessions : {len(results)}")
    print(f"  Filtered (<{min_snow_cm} cm)  : {n_filtered}  |  missing boundary: {n_no_boundary}")
    return pl.DataFrame(results) if results else _empty_session_df()


def _empty_session_df() -> pl.DataFrame:
    return pl.DataFrame({
        "session_id":    pl.Series([], dtype=pl.Int64),
        "session_start": pl.Series([], dtype=pl.Datetime),
        "session_end":   pl.Series([], dtype=pl.Datetime),
        "n_days":        pl.Series([], dtype=pl.Int64),
        "avg_snow_cm":   pl.Series([], dtype=pl.Float64),
        "pre_dev":       pl.Series([], dtype=pl.Float64),
        "during_dev":    pl.Series([], dtype=pl.Float64),
        "post_dev":      pl.Series([], dtype=pl.Float64),
    })


# INDEPENDENT-EVENT FILTER  (paper methodology)

def build_daily_snow_map(df: pl.DataFrame) -> dict[date, float]:
    """
    Return {calendar_date: daily_snow_cm} for every day in the dataset.
    Used to check whether adjacent days have ≥5 cm snow.
    """
    lookup = (
        df.filter(pl.col("total_snow_cm").is_not_null())
        .with_columns(pl.col("bucket_start").dt.truncate("1d").alias("day"))
        .group_by("day")
        .agg(pl.col("total_snow_cm").first().alias("snow"))
    )

    def _to_date(d):
        if isinstance(d, datetime): return d.date()
        if isinstance(d, date):     return d
        return None

    return {
        _to_date(r["day"]): float(r["snow"])
        for r in lookup.iter_rows(named=True)
        if _to_date(r["day"]) is not None
    }


def filter_independent_events(
    session_event_df: pl.DataFrame,
    daily_snow_map: dict[date, float],
    threshold: float = MIN_SNOW_CM,
) -> pl.DataFrame:
    """
    Keep sessions where neither the day before session_start nor the day after
    session_end has >= threshold cm snow.

    Multi-day sessions are included — the session framework already collapses
    them into a single averaged "during" value, so they are just as valid as
    1-day sessions as long as the boundary days are clean.
    """
    keep = []
    for row in session_event_df.iter_rows(named=True):
        pre_date  = row["session_start"].date() - timedelta(days=1)
        post_date = row["session_end"].date()   + timedelta(days=1)
        if (daily_snow_map.get(pre_date,  0.0) < threshold and
                daily_snow_map.get(post_date, 0.0) < threshold):
            keep.append(row)
    if not keep:
        return _empty_session_df()
    return pl.DataFrame(keep)


# STATISTICAL HELPERS

def paired_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test with Cohen's d.  Returns dict with mean_diff_pct, t, p, d."""
    from scipy import stats as scipy_stats
    diff  = a - b
    t, p  = scipy_stats.ttest_rel(a, b)
    d     = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
    sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return {
        "a_mean_pct":    float(a.mean() * 100),
        "b_mean_pct":    float(b.mean() * 100),
        "mean_diff_pct": float(diff.mean() * 100),
        "t":             float(t),
        "p":             float(p),
        "d":             float(d),
        "n":             len(a),
        "sig":           sig,
    }


def brown_forsythe(*groups: np.ndarray) -> dict:
    """Brown-Forsythe test (Levene with center='median') for equal variance."""
    from scipy import stats as scipy_stats
    stat, p = scipy_stats.levene(*groups, center="median")
    return {"stat": float(stat), "p": float(p),
            "significant": p < 0.05}
