"""
routing.py
----------
Builds the NER road network as a graph and provides:
  1. Risk-aware shortest path (Dijkstra, via networkx)
  2. Convoy Formation Intelligence — group vehicles heading the same way
     into a single shared route instead of routing each one separately.
  3. Criticality-Based Triage Routing — when a route is risky/constrained,
     decide which vehicles go now vs. which wait, based on cargo tier.

This is intentionally dependency-light (networkx only) so it runs anywhere,
including offline, with no external routing service required for the demo.
"""

import json
import os
import networkx as nx

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "road_network.json")

# Blocked roads become effectively impassable for routing (very high weight)
# rather than removed, so the API can still report "why" a route was avoided.
BLOCKED_WEIGHT_PENALTY = 10_000


def load_network(path: str = DATA_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_graph(network: dict, risk_aversion: float = 0.5) -> nx.Graph:
    """
    Build an undirected weighted graph from the road network.

    risk_aversion (0-1): how much the route planner penalizes risky roads.
      0.0 -> pure shortest-time routing (ignores risk)
      1.0 -> heavily avoids risky roads even if much longer
    """
    g = nx.Graph()
    for node in network["nodes"]:
        g.add_node(node["id"], name=node["name"], lat=node["lat"], lon=node["lon"])

    for edge in network["edges"]:
        risk = edge["risk"]
        time_min = edge["base_time_min"]

        if edge["status"] == "blocked":
            weight = time_min + BLOCKED_WEIGHT_PENALTY
        else:
            # Weight blends travel time with risk scaling without drastically
            # overshadowing distance / travel time.
            weight = time_min * (1.0 + risk * risk_aversion)

        g.add_edge(
            edge["from"], edge["to"],
            id=edge["id"], distance_km=edge["distance_km"],
            base_time_min=time_min, risk=risk, status=edge["status"],
            weight=weight,
        )
    return g


def compute_route(network: dict, start: str, end: str, risk_aversion: float = 0.5) -> dict:
    g = build_graph(network, risk_aversion)

    if start not in g or end not in g:
        return {"error": f"Unknown node(s): {start}, {end}"}

    try:
        path = nx.dijkstra_path(g, start, end, weight="weight")
    except nx.NetworkXNoPath:
        return {"error": f"No route found between {start} and {end}"}

    segments = []
    total_distance = 0.0
    total_time = 0.0
    max_risk = 0.0

    for a, b in zip(path[:-1], path[1:]):
        edge = g[a][b]
        segments.append({
            "from": a, "to": b, "edge_id": edge["id"],
            "distance_km": edge["distance_km"], "time_min": edge["base_time_min"],
            "risk": edge["risk"], "status": edge["status"],
        })
        total_distance += edge["distance_km"]
        total_time += edge["base_time_min"]
        max_risk = max(max_risk, edge["risk"])

    coords = [{"lat": g.nodes[n]["lat"], "lon": g.nodes[n]["lon"], "name": g.nodes[n]["name"]} for n in path]

    return {
        "path": path,
        "coordinates": coords,
        "segments": segments,
        "total_distance_km": round(total_distance, 1),
        "total_time_min": round(total_time, 1),
        "max_risk": round(max_risk, 2),
        "risk_aversion_used": risk_aversion,
    }


# ---------------------------------------------------------------------------
# Convoy Formation Intelligence
# ---------------------------------------------------------------------------
def group_into_convoys(network: dict, vehicle_requests: list, time_window_min: int = 60,
                        risk_aversion: float = 0.5) -> list:
    """
    vehicle_requests: list of dicts like
      {"vehicle_id": "V1", "start": "GAU", "end": "IMP", "ready_time_min": 0, "tier": 2}

    Groups requests that share the same start & end and whose ready_time_min
    falls within time_window_min of each other, then computes ONE route for
    the whole group instead of one per vehicle.
    """
    groups = {}
    for req in vehicle_requests:
        key = (req["start"], req["end"])
        groups.setdefault(key, []).append(req)

    convoys = []
    for (start, end), reqs in groups.items():
        reqs_sorted = sorted(reqs, key=lambda r: r["ready_time_min"])
        current_convoy = [reqs_sorted[0]]

        for req in reqs_sorted[1:]:
            if req["ready_time_min"] - current_convoy[0]["ready_time_min"] <= time_window_min:
                current_convoy.append(req)
            else:
                convoys.append(_finalize_convoy(network, start, end, current_convoy, risk_aversion))
                current_convoy = [req]
        convoys.append(_finalize_convoy(network, start, end, current_convoy, risk_aversion))

    return convoys


def _finalize_convoy(network, start, end, reqs, risk_aversion):
    route = compute_route(network, start, end, risk_aversion)
    return {
        "start": start, "end": end,
        "vehicle_ids": [r["vehicle_id"] for r in reqs],
        "vehicle_count": len(reqs),
        "route": route,
    }


# ---------------------------------------------------------------------------
# Criticality-Based Triage Routing
# ---------------------------------------------------------------------------
def triage_dispatch(pending_vehicles: list, safe_window_min: float) -> dict:
    """
    pending_vehicles: list of dicts like
      {"vehicle_id": "V1", "tier": 1, "eta_min": 15}
    (tier 1 = highest criticality, e.g. medical/emergency; 3 = lowest)

    safe_window_min: how long the risky route is expected to remain
    passable (from the risk model / live monitoring), e.g. before a
    predicted landslide/flood risk crosses threshold.

    Returns which vehicles are cleared to go now vs. held, based on tier
    and whether they can make it inside the safe window.
    """
    ranked = sorted(pending_vehicles, key=lambda v: (v["tier"], v["eta_min"]))

    dispatched, held = [], []
    time_used = 0.0

    for v in ranked:
        # naive model: each dispatched vehicle "consumes" a slice of the
        # remaining safe window equal to its ETA, to keep the demo logic
        # simple and explainable
        if time_used + v["eta_min"] <= safe_window_min:
            dispatched.append({**v, "status": "DISPATCH_NOW"})
            time_used += v["eta_min"]
        else:
            held.append({
                **v, "status": "HOLD",
                "reason": "Safe window closes before this vehicle's tier priority is reached",
            })

    return {
        "safe_window_min": safe_window_min,
        "dispatched": dispatched,
        "held": held,
    }
