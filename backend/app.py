"""
app.py
------
Praneta prototype backend (SIH26002 - Kabutar Coders).

Run:
    pip install -r ../requirements.txt
    python app.py

Then open frontend/index.html in a browser (or serve it with
`python -m http.server` from the frontend/ folder).
"""

from flask import Flask, jsonify, request
import routing
import risk_model

app = Flask(__name__)
NETWORK = routing.load_network()

# In-memory demo stores (swap for SQLite/Postgres later - see README)
VEHICLES = {}
REPORTS = []


@app.after_request
def add_cors_headers(resp):
    # Minimal manual CORS so the static frontend/index.html can call this
    # API directly during local dev, with no extra pip packages required.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------
@app.route("/api/roads", methods=["GET"])
def get_roads():
    return jsonify(NETWORK)


@app.route("/api/roads/<edge_id>", methods=["PATCH"])
def update_road(edge_id):
    """Simulate a live update, e.g. {"risk": 0.9, "status": "high_risk"}"""
    body = request.get_json(force=True)
    for edge in NETWORK["edges"]:
        if edge["id"] == edge_id:
            if "risk" in body:
                edge["risk"] = float(body["risk"])
            if "status" in body:
                edge["status"] = body["status"]
            return jsonify(edge)
    return jsonify({"error": f"Unknown edge_id {edge_id}"}), 404


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
@app.route("/api/route", methods=["GET"])
def get_route():
    start = request.args.get("start")
    end = request.args.get("end")
    risk_aversion = float(request.args.get("risk_aversion", 0.5))
    if not start or not end:
        return jsonify({"error": "start and end query params are required"}), 400
    result = routing.compute_route(NETWORK, start, end, risk_aversion)
    status = 400 if "error" in result else 200
    return jsonify(result), status


# ---------------------------------------------------------------------------
# Convoy Formation Intelligence
# ---------------------------------------------------------------------------
@app.route("/api/convoy", methods=["POST"])
def post_convoy():
    """
    Body: {
      "vehicle_requests": [
        {"vehicle_id": "V1", "start": "GAU", "end": "IMP", "ready_time_min": 0, "tier": 2},
        ...
      ],
      "time_window_min": 60,
      "risk_aversion": 0.5
    }
    """
    body = request.get_json(force=True)
    convoys = routing.group_into_convoys(
        NETWORK,
        body["vehicle_requests"],
        time_window_min=body.get("time_window_min", 60),
        risk_aversion=body.get("risk_aversion", 0.5),
    )
    return jsonify({"convoys": convoys})


# ---------------------------------------------------------------------------
# Criticality-Based Triage Routing
# ---------------------------------------------------------------------------
@app.route("/api/triage", methods=["POST"])
def post_triage():
    """
    Body: {
      "pending_vehicles": [{"vehicle_id": "V1", "tier": 1, "eta_min": 15}, ...],
      "safe_window_min": 40
    }
    """
    body = request.get_json(force=True)
    result = routing.triage_dispatch(body["pending_vehicles"], body["safe_window_min"])
    return jsonify(result)


# ---------------------------------------------------------------------------
# Risk prediction (XGBoost)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Risk prediction (XGBoost)
# ---------------------------------------------------------------------------
@app.route("/api/predict-risk", methods=["POST"])
def post_predict_risk():
    """
    Body: {
      "elevation": 250, "rainfall_mm": 180, "slope_deg": 35
    }
    """
    body = request.get_json(force=True)
    result = risk_model.predict_risk(
        elevation=body.get("elevation", 0),
        rainfall_mm=body.get("rainfall_mm", 0),
        slope_deg=body.get("slope_deg", 0)
    )
    return jsonify(result)


# ---------------------------------------------------------------------------
# Vehicles + field reports (simple in-memory demo store)
# ---------------------------------------------------------------------------
@app.route("/api/vehicles", methods=["GET", "POST"])
def vehicles():
    if request.method == "POST":
        body = request.get_json(force=True)
        VEHICLES[body["vehicle_id"]] = body
        return jsonify(body), 201
    return jsonify(list(VEHICLES.values()))


@app.route("/api/reports", methods=["GET", "POST"])
def reports():
    if request.method == "POST":
        body = request.get_json(force=True)
        REPORTS.append(body)
        return jsonify(body), 201
    return jsonify(REPORTS)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
