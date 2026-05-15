"""
Batch route computation for all ADA origin-destination pairs.

For each OD pair, generates multiple route variants by sampling random points
within each ADA polygon and querying 16 parallel OSRM instances. Road-type
distances (motorway, trunk, primary, …) are resolved via the local Overpass API.

Usage
-----
    # Start OSRM and Overpass containers first (see docker-compose-osrm.yml,
    # docker-compose-overpass.yml), then:
    python compute_routes.py

Inputs
------
    ADA_PATH     : GeoJSON of ADA polygons (744 ADAs, EPSG:4326)
    OD_PATH      : Parquet of valid OD pairs with max observed trip count
    OUTPUT_PATH  : Output parquet path

Output schema (one row per route variant)
------------------------------------------
    task_key, original_od_key, input_geoid, output_geoid,
    traffic_volume, route_number,
    start_lat, start_lon, end_lat, end_lon,
    total_distance_km, total_duration_min, osrm_port_used,
    <highway_type>_dist_km, <highway_type>_duration_min, <highway_type>_prop_pct
      for each type in: motorway, trunk, primary, secondary,
                        tertiary, unclassified, residential
"""

import math
import os
import random
import concurrent.futures

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from tqdm.auto import tqdm

from osrm_tools import get_route, analyze_route, ALLOWED_HIGHWAY_TYPES

# Paths
ADA_PATH    = "../gis-data/ADA_GTA_DRIFT.geojson"
OD_PATH     = "../data/processed/od_pairs.parquet"
OUTPUT_PATH = "routing_results.parquet"

# OSRM instances (must match docker-compose-osrm.yml)
OSRM_PORTS  = list(range(5000, 5016))

# Number of route variants per OD pair — scaled by traffic volume
MIN_ROUTES  = 2
MAX_ROUTES  = 10

CRS_WGS84 = "EPSG:4326"


def random_point_in_polygon(polygon: Polygon) -> Point:
    """Return a random point inside polygon; falls back to centroid."""
    prepared = prep(polygon)
    minx, miny, maxx, maxy = polygon.bounds
    for _ in range(100):
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if prepared.contains(p):
            return p
    return polygon.centroid


def num_routes(traffic_volume: float) -> int:
    """Scale route count logarithmically with traffic volume."""
    if traffic_volume <= 0:
        return MIN_ROUTES
    log_val  = math.log10(max(1, min(traffic_volume, 100_000)) + 1)
    log_min  = math.log10(2)
    log_max  = math.log10(100_001)
    scaled   = MIN_ROUTES + (log_val - log_min) / (log_max - log_min) * (MAX_ROUTES - MIN_ROUTES)
    return int(max(MIN_ROUTES, min(MAX_ROUTES, round(scaled))))


def _port_cycle(ports):
    while True:
        yield from ports


def process_task(task):
    route_data = get_route(task["start_coords"], task["end_coords"], task["osrm_port"])
    if not route_data:
        return None

    summary_df, route, _, total_dist, total_dur = analyze_route(route_data)
    if route is None:
        return None

    record = {
        "task_key":          task["task_key"],
        "original_od_key":   task["od_key"],
        "input_geoid":       task["input_geoid"],
        "output_geoid":      task["output_geoid"],
        "traffic_volume":    task["traffic_volume"],
        "route_number":      task["route_number"],
        "start_lat":         task["start_coords"][0],
        "start_lon":         task["start_coords"][1],
        "end_lat":           task["end_coords"][0],
        "end_lon":           task["end_coords"][1],
        "total_distance_km": (total_dist or 0) / 1000,
        "total_duration_min":(total_dur  or 0) / 60,
        "osrm_port_used":    task["osrm_port"],
    }

    if summary_df is not None:
        for _, row in summary_df.iterrows():
            cat = str(row["Road Category"]).replace(" ", "_")
            record[f"{cat}_dist_km"]      = row["Length (km)"]
            record[f"{cat}_duration_min"] = row["Time (min)"]
            record[f"{cat}_prop_pct"]     = row["Proportion (%)"]

    return record


def main():
    print("Loading ADA polygons …")
    ada_gdf = gpd.read_file(ADA_PATH).to_crs(CRS_WGS84).set_index("geoid")

    print("Loading OD pairs …")
    od_pairs = pd.read_parquet(OD_PATH)

    tasks = []
    ports = _port_cycle(OSRM_PORTS)

    print("Building routing tasks …")
    for _, row in tqdm(od_pairs.iterrows(), total=len(od_pairs)):
        origin_id = str(row["input_geoid"])
        dest_id   = str(row["output_geoid"])
        vol       = float(row.get("traffic_volume", 1))
        od_key    = f"{origin_id}-{dest_id}"

        if origin_id not in ada_gdf.index or dest_id not in ada_gdf.index:
            continue

        origin_poly = ada_gdf.loc[origin_id].geometry
        dest_poly   = ada_gdf.loc[dest_id].geometry

        for i in range(num_routes(vol)):
            o = random_point_in_polygon(origin_poly)
            d = random_point_in_polygon(dest_poly)
            tasks.append({
                "task_key":    f"{od_key}_{i}",
                "od_key":      od_key,
                "input_geoid": origin_id,
                "output_geoid":dest_id,
                "traffic_volume": vol,
                "route_number": i,
                "start_coords": (o.y, o.x),
                "end_coords":   (d.y, d.x),
                "osrm_port":    next(ports),
            })

    print(f"Total routing tasks: {len(tasks):,}")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(OSRM_PORTS) * 2) as executor:
        futures = {executor.submit(process_task, t): t for t in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            record = future.result()
            if record:
                results.append(record)

    if not results:
        print("No routes processed.")
        return

    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} routes → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
