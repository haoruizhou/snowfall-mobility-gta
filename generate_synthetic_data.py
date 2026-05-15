"""
Generate a synthetic OD mobility dataset for code verification.

Creates synthetic_data/od_mobility_synthetic.parquet and
synthetic_data/snow_sessions_synthetic.json.  The dataset has the same
schema as the real TELUS mobility data but contains no real information.
Snow-suppression and pre-event effects are embedded by construction rather
than measured.

Usage: python generate_synthetic_data.py
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SEED       = 42
DATE_START = date(2021, 8, 2)
DATE_END   = date(2025, 12, 30)

N_OD_PAIRS    = 100   # real dataset: ~552,000
BUCKET_HOURS  = [0, 12]

MAJOR_SNOW_CM = 5.0
ANY_SNOW_CM   = 1.0

BASE_TRIPS_MEAN  = 28.0
BASE_TRIPS_SIGMA = 0.55
DOW_FACTORS      = [1.15, 1.12, 1.10, 1.10, 1.08, 0.72, 0.68]  # Mon–Sun
SEASONAL_FACTORS = {
    1: 0.97, 2: 0.96, 3: 0.98, 4: 1.00, 5: 1.01, 6: 1.01,
    7: 0.99, 8: 1.00, 9: 1.02, 10: 1.02, 11: 1.00, 12: 0.98,
}

SNOW_SUPPRESSION = {
    "0-2km":   -0.008,
    "2-5km":   -0.004,
    "5-10km":  -0.015,
    "10-20km": -0.040,
    "20+km":   -0.050,
}
SNOW_NOISE_SD      = 0.055
LIGHT_SNOW_MEAN    = -0.005
LIGHT_SNOW_SD      = 0.030
PRE_EVENT_MEAN     = +0.047
PRE_EVENT_SD       = 0.035
CLEAR_DAY_NOISE_SD = 0.025

# Applied uniformly to all pairs so it does not average away across the cohort
DAY_CORR_SD = 0.045

OUT_DIR = Path(__file__).parent / "synthetic_data"

# 1. SNOWFALL MODEL

WINTER_MONTHS = {10, 11, 12, 1, 2, 3, 4}

SNOW_PROB_BY_MONTH = {
    10: 0.045, 11: 0.10, 12: 0.125, 1: 0.155, 2: 0.14, 3: 0.10, 4: 0.035,
}
P_HEAVY_GIVEN_SNOW = 0.39  # P(≥5 cm | snow occurs)
P_PERSIST         = 0.12   # P(next day also ≥5 cm | today ≥5 cm) — produces ~4 two-day events


def _daily_snowfall_cm(d: date, rng: np.random.Generator) -> float:
    """Stochastic daily snowfall (cm) for date d."""
    if d.month not in WINTER_MONTHS:
        return 0.0
    if rng.random() > SNOW_PROB_BY_MONTH.get(d.month, 0.0):
        return 0.0
    if rng.random() < P_HEAVY_GIVEN_SNOW:
        amount = rng.gamma(shape=2.0, scale=3.0) + 5.0
    else:
        amount = rng.uniform(1.0, 4.9)
    return round(min(amount, 45.0), 1)


def generate_snow_map(rng: np.random.Generator) -> dict[date, float]:
    """Return {calendar date: snowfall_cm} for the full study period."""
    snow_map: dict[date, float] = {}
    d = DATE_START
    prev_heavy = False
    while d <= DATE_END:
        snow = _daily_snowfall_cm(d, rng)
        # Storm persistence: if yesterday was heavy, give a chance today is too
        if prev_heavy and snow < MAJOR_SNOW_CM and d.month in WINTER_MONTHS:
            if rng.random() < P_PERSIST:
                snow = round(min(rng.gamma(shape=2.0, scale=3.0) + 5.0, 45.0), 1)
        snow_map[d] = snow
        prev_heavy = snow >= MAJOR_SNOW_CM
        d += timedelta(days=1)
    return snow_map


# 2. OD PAIR CHARACTERISTICS

def _distance_bin(km: float) -> str:
    if km < 2:   return "0-2km"
    if km < 5:   return "2-5km"
    if km < 10:  return "5-10km"
    if km < 20:  return "10-20km"
    return "20+km"


def create_od_pairs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    distances : (N_OD_PAIRS,)
        Route length in km for each synthetic OD pair.
    road_km   : (N_OD_PAIRS, 7)
        Distance on each road type [motorway, trunk, primary, secondary,
        tertiary, residential, unclassified] — columns sum to distances[i].
    """
    # GTA trips concentrate in 5–25 km range; lognormal captures the tail
    distances = np.exp(rng.normal(loc=np.log(10.0), scale=0.75, size=N_OD_PAIRS))
    distances = np.clip(distances, 0.5, 75.0)

    # Road-type fractions depend on route length
    # Longer trips use more motorway/trunk; shorter trips stay on local roads
    ROAD_PROFILES = {
        "0-2km":   np.array([0.00, 0.00, 0.05, 0.10, 0.15, 0.62, 0.08]),
        "2-5km":   np.array([0.00, 0.02, 0.15, 0.25, 0.20, 0.30, 0.08]),
        "5-10km":  np.array([0.05, 0.08, 0.25, 0.25, 0.15, 0.15, 0.07]),
        "10-20km": np.array([0.20, 0.15, 0.25, 0.20, 0.10, 0.07, 0.03]),
        "20+km":   np.array([0.45, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]),
    }

    road_km = np.zeros((N_OD_PAIRS, 7))
    for i, d in enumerate(distances):
        profile = ROAD_PROFILES[_distance_bin(d)]
        # Dirichlet perturbation preserves road-type character per pair
        fracs = rng.dirichlet(profile * 15 + 0.1)
        road_km[i] = fracs * d

    return distances, road_km


# 3. MOBILITY DATA GENERATION

def build_mobility_dataframe(
    snow_map: dict[date, float],
    distances: np.ndarray,
    road_km: np.ndarray,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """Build a row for every (calendar day × OD pair × 12-hour bucket)."""
    base_vol = np.exp(rng.normal(np.log(BASE_TRIPS_MEAN), BASE_TRIPS_SIGMA, N_OD_PAIRS))
    base_vol = np.clip(base_vol, 1.0, 300.0)

    major_snow_dates = {d for d, s in snow_map.items() if s >= MAJOR_SNOW_CM}

    cols: dict[str, list] = {k: [] for k in [
        "bucket_start", "weather_label", "count", "baseline",
        "has_snow", "snow_on_ground_cm", "total_snow_cm",
        "total_distance_km",
        "motorway_km", "trunk_km", "primary_km", "secondary_km",
        "tertiary_km", "residential_km", "unclassified_km",
    ]}

    all_dates = sorted(snow_map.keys())
    n_dates   = len(all_dates)

    for idx, d in enumerate(all_dates):
        if idx % 200 == 0:
            print(f"  Processing day {idx+1}/{n_dates} …", end="\r", flush=True)

        snow_today   = snow_map[d]
        is_major     = snow_today >= MAJOR_SNOW_CM
        is_any       = snow_today >= ANY_SNOW_CM
        is_pre_event = (d + timedelta(days=1)) in major_snow_dates
        dow_factor   = DOW_FACTORS[d.weekday()]
        seasonal     = SEASONAL_FACTORS[d.month]

        snow_on_ground = min(snow_today * 2.5, 80.0)
        weather_label  = "Snow" if is_any else "Clear"

        day_offset = rng.normal(0.0, DAY_CORR_SD)

        # Per-pair effects, vectorised over N_OD_PAIRS
        if is_major:
            mean_effects = np.array([SNOW_SUPPRESSION[_distance_bin(dist)] for dist in distances])
            effects = mean_effects + rng.normal(0.0, SNOW_NOISE_SD, N_OD_PAIRS) + day_offset
        elif is_any:
            effects = rng.normal(LIGHT_SNOW_MEAN, LIGHT_SNOW_SD, N_OD_PAIRS) + day_offset
        else:
            effects = rng.normal(0.0, CLEAR_DAY_NOISE_SD, N_OD_PAIRS) + day_offset

        if is_pre_event:
            effects += rng.normal(PRE_EVENT_MEAN, PRE_EVENT_SD, N_OD_PAIRS)

        bvol = base_vol * dow_factor * seasonal

        for bh, bf in [(0, 0.46), (12, 0.54)]:
            bb  = bvol * bf
            cnt = np.maximum(0, np.round(bb * (1.0 + effects))).astype(np.int16)
            bsl = np.maximum(0.0, np.round(bb, 2))

            dt = datetime(d.year, d.month, d.day, bh, 0, 0)

            cols["bucket_start"].append(np.full(N_OD_PAIRS, dt, dtype=object))
            cols["weather_label"].append(np.full(N_OD_PAIRS, weather_label))
            cols["count"].append(cnt)
            cols["baseline"].append(bsl)
            cols["has_snow"].append(np.full(N_OD_PAIRS, is_any))
            cols["snow_on_ground_cm"].append(np.full(N_OD_PAIRS, snow_on_ground))
            cols["total_snow_cm"].append(np.full(N_OD_PAIRS, snow_today))
            cols["total_distance_km"].append(distances)
            cols["motorway_km"].append(road_km[:, 0])
            cols["trunk_km"].append(road_km[:, 1])
            cols["primary_km"].append(road_km[:, 2])
            cols["secondary_km"].append(road_km[:, 3])
            cols["tertiary_km"].append(road_km[:, 4])
            cols["residential_km"].append(road_km[:, 5])
            cols["unclassified_km"].append(road_km[:, 6])

    print()

    flat = {k: np.concatenate(v) for k, v in cols.items()}

    df = pl.DataFrame({
        "bucket_start":      pl.Series(flat["bucket_start"].tolist()).cast(pl.Datetime("us")),
        "weather_label":     pl.Series(flat["weather_label"].tolist()).cast(pl.Utf8),
        "count":             pl.Series(flat["count"].tolist()).cast(pl.Int16),
        "baseline":          pl.Series(flat["baseline"].tolist()).cast(pl.Float64),
        "has_snow":          pl.Series(flat["has_snow"].tolist()).cast(pl.Boolean),
        "snow_on_ground_cm": pl.Series(flat["snow_on_ground_cm"].tolist()).cast(pl.Float64),
        "total_snow_cm":     pl.Series(flat["total_snow_cm"].tolist()).cast(pl.Float64),
        "total_distance_km": pl.Series(flat["total_distance_km"].tolist()).cast(pl.Float64),
        "motorway_km":       pl.Series(flat["motorway_km"].tolist()).cast(pl.Float64),
        "trunk_km":          pl.Series(flat["trunk_km"].tolist()).cast(pl.Float64),
        "primary_km":        pl.Series(flat["primary_km"].tolist()).cast(pl.Float64),
        "secondary_km":      pl.Series(flat["secondary_km"].tolist()).cast(pl.Float64),
        "tertiary_km":       pl.Series(flat["tertiary_km"].tolist()).cast(pl.Float64),
        "residential_km":    pl.Series(flat["residential_km"].tolist()).cast(pl.Float64),
        "unclassified_km":   pl.Series(flat["unclassified_km"].tolist()).cast(pl.Float64),
    }).sort("bucket_start")

    return df


# 4. SNOW SESSION DETECTION

def detect_sessions(snow_map: dict[date, float], threshold: float = ANY_SNOW_CM) -> list[dict]:
    """
    Group consecutive calendar days with ≥ threshold cm snow into sessions.
    Returns a list of session dicts matching the JSON schema expected by utils.py.
    """
    sessions = []
    sid      = 1
    sorted_dates = sorted(snow_map.keys())
    i = 0
    while i < len(sorted_dates):
        d = sorted_dates[i]
        if snow_map[d] < threshold:
            i += 1
            continue
        start = d
        end   = d
        j = i + 1
        while j < len(sorted_dates) and sorted_dates[j] == end + timedelta(days=1) \
                and snow_map[sorted_dates[j]] >= threshold:
            end = sorted_dates[j]
            j  += 1
        sessions.append({
            "session_number": sid,
            "start_date":     datetime(start.year, start.month, start.day).isoformat(),
            "end_date":       datetime(end.year,   end.month,   end.day).isoformat(),
            "duration_days":  (end - start).days + 1,
        })
        sid += 1
        i    = j
    return sessions


# 5. MAIN

def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Weather ───────────────────────────────────────────────────────────────
    print("Generating synthetic daily snowfall …")
    snow_map = generate_snow_map(rng)
    n_any   = sum(1 for s in snow_map.values() if ANY_SNOW_CM   <= s)
    n_major = sum(1 for s in snow_map.values() if MAJOR_SNOW_CM <= s)
    print(f"  Days ≥ {ANY_SNOW_CM} cm   : {n_any}")
    print(f"  Days ≥ {MAJOR_SNOW_CM} cm   : {n_major}")
    print(f"  (Paper reports  ≥1 cm: ~96   ≥5 cm: ~37  over 2021–2025)")

    # ── OD pairs ──────────────────────────────────────────────────────────────
    print("\nAssigning route characteristics to synthetic OD pairs …")
    distances, road_km = create_od_pairs(rng)
    print(f"  Distance range : {distances.min():.1f} – {distances.max():.1f} km")
    print(f"  Distance median: {np.median(distances):.1f} km")

    # ── Mobility DataFrame ────────────────────────────────────────────────────
    print("\nBuilding mobility rows …")
    df = build_mobility_dataframe(snow_map, distances, road_km, rng)
    n_rows = df.height
    print(f"  Total rows : {n_rows:,}  ({N_OD_PAIRS} pairs × {n_rows//N_OD_PAIRS//2} days × 2 buckets)")

    parquet_path = OUT_DIR / "od_mobility_synthetic.parquet"
    df.write_parquet(str(parquet_path), compression="zstd")
    size_mb = parquet_path.stat().st_size / 1e6
    print(f"  Saved  : {parquet_path}  ({size_mb:.1f} MB)")

    # ── Snow sessions JSON ────────────────────────────────────────────────────
    print("\nDetecting snow sessions …")
    sessions = detect_sessions(snow_map, threshold=ANY_SNOW_CM)
    json_path = OUT_DIR / "snow_sessions_synthetic.json"
    json_path.write_text(json.dumps({"sessions": sessions}, indent=2))
    print(f"  Sessions detected : {len(sessions)}")
    print(f"  Saved  : {json_path}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\nSanity check — mean mobility deviation on major-snow days:")
    major_dates = [d for d, s in snow_map.items() if s >= MAJOR_SNOW_CM]
    sample_devs = []
    for d in major_dates[:10]:
        dt_start = datetime(d.year, d.month, d.day)
        dt_end   = datetime(d.year, d.month, d.day) + timedelta(days=1)
        sub = df.filter(
            (pl.col("bucket_start") >= dt_start) & (pl.col("bucket_start") < dt_end)
        )
        if sub.height:
            raw  = float(sub["count"].cast(pl.Int64).sum())
            base = float(sub["baseline"].sum())
            if base > 0:
                sample_devs.append((raw - base) / base * 100)
    if sample_devs:
        print(f"  First 10 major-snow days: mean dev = {np.mean(sample_devs):+.2f}%")
        print(f"  (Paper reports ~−1.7% on average for ≥5 cm days)")

    print("\nDone.  Run  paper_figures.py  to generate the three figures.")


if __name__ == "__main__":
    main()
