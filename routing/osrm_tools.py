"""
OSRM and Overpass API helpers for route computation and road-type classification.
"""

import os
import requests
import polyline
import pandas as pd

ALLOWED_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary", "secondary",
    "tertiary", "unclassified", "residential",
}

OSRM_HOST     = os.environ.get("OSRM_HOST",     "localhost")
OSRM_PORT     = os.environ.get("OSRM_PORT",     "5000")
OVERPASS_HOST = os.environ.get("OVERPASS_HOST", "localhost")
OVERPASS_PORT = os.environ.get("OVERPASS_PORT", "12345")


def get_route(start_coords, end_coords, osrm_port=None):
    """
    Query OSRM for a driving route with full geometry and node annotations.

    Parameters
    ----------
    start_coords : (lat, lon)
    end_coords   : (lat, lon)
    osrm_port    : int, optional — overrides OSRM_PORT (used for round-robin)

    Returns
    -------
    dict | None
    """
    port = osrm_port or OSRM_PORT
    coords = f"{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
    url = (
        f"http://{OSRM_HOST}:{port}/route/v1/driving/{coords}"
        "?overview=full&annotations=true&steps=true"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def get_ways_for_nodes(node_ids):
    """
    Query the local Overpass API for OSM ways that contain the given node IDs.
    Used to resolve highway tags for OSRM route segments.

    Returns
    -------
    dict | None  (raw Overpass JSON response)
    """
    url = f"http://{OVERPASS_HOST}:{OVERPASS_PORT}/api/interpreter"
    nodes_str = "".join(f"node({nid});" for nid in set(node_ids))
    query = f"[out:json][timeout:60];\n({nodes_str});\nway(bn);\nout geom;"
    try:
        r = requests.post(url, data={"data": query}, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def analyze_route(route_data):
    """
    Extract distance and road-type breakdown from an OSRM route response.

    Queries Overpass to classify each route segment by highway type, then
    aggregates distance and travel time per type.

    Returns
    -------
    summary_df : pd.DataFrame | None
        Columns: Road Category, Length (km), Time (min), Proportion (%)
    route : dict | None
        Raw first route from OSRM response.
    segments : list[dict] | None
        [{"geometry": [(lat,lon), ...], "type": str}, ...]
    total_distance_m : float | None
    total_duration_s : float | None
    """
    if not route_data or "routes" not in route_data or not route_data["routes"]:
        return None, None, None, None, None

    route          = route_data["routes"][0]
    total_distance = route["distance"]
    total_duration = route["duration"]
    decoded_geom   = polyline.decode(route["geometry"])

    all_nodes, all_distances, all_durations = [], [], []
    for leg in route["legs"]:
        ann = leg.get("annotation", {})
        all_nodes.extend(ann.get("nodes", []))
        all_distances.extend(ann.get("distance", []))
        all_durations.extend(ann.get("duration", []))

    # Fallback: no annotation data
    if not all_nodes or len(decoded_geom) < len(all_nodes):
        segments = [{"geometry": decoded_geom, "type": "unknown"}]
        df = pd.DataFrame([{
            "Road Category": "unknown",
            "Length (km)":   total_distance / 1000,
            "Time (min)":    total_duration / 60,
            "Proportion (%)": 100.0,
        }])
        return df, route, segments, total_distance, total_duration

    way_data = get_ways_for_nodes(all_nodes)
    if not way_data or "elements" not in way_data:
        segments = [{"geometry": decoded_geom, "type": "unknown"}]
        df = pd.DataFrame([{
            "Road Category": "unknown",
            "Length (km)":   total_distance / 1000,
            "Time (min)":    total_duration / 60,
            "Proportion (%)": 100.0,
        }])
        return df, route, segments, total_distance, total_duration

    # Build node → highway-type lookup from Overpass ways
    node_highway: dict[int, str] = {}
    for element in way_data.get("elements", []):
        if element.get("type") != "way":
            continue
        hw = element.get("tags", {}).get("highway", "excluded")
        hw = hw if hw in ALLOWED_HIGHWAY_TYPES else "excluded"
        for node in element.get("nodes", []):
            node_highway[node] = hw

    # Segment route geometry by road type
    road_stats: dict[str, dict] = {}
    segments = []
    current_type = None
    current_geom = []
    allowed_distance = 0.0
    allowed_duration = 0.0

    for i, node_id in enumerate(all_nodes):
        hw = node_highway.get(node_id, "excluded")
        seg_dist = all_distances[i] if i < len(all_distances) else 0.0
        seg_dur  = all_durations[i]  if i < len(all_durations)  else 0.0

        if hw != current_type:
            if current_geom:
                segments.append({"geometry": current_geom, "type": current_type})
            current_type = hw
            current_geom = []

        if i < len(decoded_geom):
            current_geom.append(decoded_geom[i])

        if hw in ALLOWED_HIGHWAY_TYPES:
            road_stats.setdefault(hw, {"distance": 0.0, "duration": 0.0})
            road_stats[hw]["distance"] += seg_dist
            road_stats[hw]["duration"] += seg_dur
            allowed_distance += seg_dist
            allowed_duration += seg_dur

    if current_geom:
        segments.append({"geometry": current_geom, "type": current_type})

    if not road_stats:
        segments = [{"geometry": decoded_geom, "type": "excluded"}]
        return None, route, segments, total_distance, total_duration

    rows = []
    for hw, stats in road_stats.items():
        proportion = (stats["distance"] / allowed_distance * 100) if allowed_distance > 0 else 0.0
        rows.append({
            "Road Category": hw,
            "Length (km)":   stats["distance"] / 1000,
            "Time (min)":    stats["duration"] / 60,
            "Proportion (%)": proportion,
        })

    df = pd.DataFrame(rows).sort_values("Proportion (%)", ascending=False).reset_index(drop=True)
    return df, route, segments, allowed_distance, allowed_duration
