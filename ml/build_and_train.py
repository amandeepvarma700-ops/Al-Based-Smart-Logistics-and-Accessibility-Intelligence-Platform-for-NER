"""
build_and_train.py
-------------------
Turns the GSI Landslide Inventory (which ONLY contains places where a
landslide happened) into a proper two-class dataset, adds terrain
features, then trains XGBoost.

Why this is needed
    XGBoost learns by comparing "landslide" places against "no landslide"
    places. The GSI file has only the first kind, so a model trained on it
    alone learns nothing (that is what happened with the first model file).
    And latitude/longitude alone cannot tell the model WHY a spot fails:
    landslides depend on the ground (steepness, height), so we add
    elevation and slope for every point.

What this script does
    1. Loads the GSI GeoJSON and keeps only the 8 North Eastern states.
    2. Labels every one of those points  landslide = 1.
    3. Creates "no landslide" points (landslide = 0): random locations
       inside the same surveyed map sheets (TOPOSHEET areas), at least
       1 km away from any known landslide. Sampling inside surveyed areas
       stops the model from just learning "where GSI happened to survey".
    4. Looks up elevation for every point from the free Open-Meteo
       Elevation API (Copernicus 90 m DEM, no key needed) and computes
       slope from it. Results are cached, so if it stops you can simply
       run the script again and it continues where it left off.
    5. Trains XGBoost on elevation + slope and reports AUC on map sheets
       the model never saw during training (an honest score).
    6. Saves the model and checks that its trees really contain splits.

Usage (from the ml/ folder):
    pip install xgboost pandas numpy scikit-learn rasterio

    # Recommended: use a local DEM file (fast, no internet, no rate limits)
    python build_and_train.py "C:\\path\\GSI_Landslide_Inventory_DATASET.geojson" "C:\\path\\ner_dem.tif"

    # Without a DEM file: looks elevations up on the web (rate limited)
    python build_and_train.py "C:\\path\\GSI_Landslide_Inventory_DATASET.geojson"

Outputs (in ml/output/):
    terrain_cache.csv                 (lookup cache, safe to keep)
    ner_landslide_training_data.csv
    ner_landslide_xgb_model.json

Data credit: elevation from Copernicus DEM GLO-90 via Open-Meteo.
"""

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import BallTree

NER_STATES = {
    "assam", "arunachal pradesh", "manipur", "meghalaya",
    "mizoram", "nagaland", "sikkim", "tripura",
}
FEATURES = ["elevation_m", "slope_deg"]
EARTH_RADIUS_KM = 6371.0088
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CACHE_PATH = os.path.join(OUT_DIR, "terrain_cache.csv")

ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
BATCH_SIZE = 100          # the API accepts up to 100 coordinates per request
SLOPE_OFFSET_DEG = 0.001  # ~110 m step used to measure how steep the ground is


class RateLimited(Exception):
    pass


# ---------------------------------------------------------------------------
# 1. Load positives (real landslides)
# ---------------------------------------------------------------------------
def load_positives(geojson_path: str) -> pd.DataFrame:
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for feat in data["features"]:
        p = feat["properties"]
        rows.append({
            "latitude": p.get("LATITUDE"),
            "longitude": p.get("LONGITUDE"),
            "state": (p.get("STATE") or "").strip(),
            "toposheet": (p.get("TOPOSHEET") or "").strip(),
        })
    df = pd.DataFrame(rows)

    df = df[df["state"].str.lower().isin(NER_STATES)].copy()
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(21, 30)) & (df["longitude"].between(88, 98))]

    # Several inventory rows share the exact same coordinates; keep one.
    df["_lat_r"] = df["latitude"].round(5)
    df["_lon_r"] = df["longitude"].round(5)
    df = df.drop_duplicates(subset=["_lat_r", "_lon_r"]).drop(columns=["_lat_r", "_lon_r"])

    # Group id used for sampling negatives and for honest validation.
    df["area"] = np.where(df["toposheet"] != "", df["toposheet"], "UNKNOWN_" + df["state"])
    df["landslide"] = 1
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Create negatives (places with no recorded landslide)
# ---------------------------------------------------------------------------
def sample_negatives(pos: pd.DataFrame, ratio: float = 1.0, min_dist_km: float = 1.0,
                     pad_deg: float = 0.03, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tree = BallTree(np.radians(pos[["latitude", "longitude"]].values), metric="haversine")
    min_rad = min_dist_km / EARTH_RADIUS_KM

    parts = []
    for area, g in pos.groupby("area"):
        target = max(1, int(round(len(g) * ratio)))
        lat_lo, lat_hi = g["latitude"].min() - pad_deg, g["latitude"].max() + pad_deg
        lon_lo, lon_hi = g["longitude"].min() - pad_deg, g["longitude"].max() + pad_deg

        got = np.empty((0, 2))
        for _ in range(60):  # rounds of "try random points, keep the valid ones"
            need = target - len(got)
            if need <= 0:
                break
            cand = np.column_stack([
                rng.uniform(lat_lo, lat_hi, need * 4),
                rng.uniform(lon_lo, lon_hi, need * 4),
            ])
            dist, _ = tree.query(np.radians(cand), k=1)
            keep = cand[dist[:, 0] > min_rad][:need]
            got = np.vstack([got, keep])

        if len(got):
            parts.append(pd.DataFrame({
                "latitude": got[:, 0], "longitude": got[:, 1],
                "state": g["state"].iloc[0], "toposheet": g["toposheet"].iloc[0],
                "area": area, "landslide": 0,
            }))
    return pd.concat(parts, ignore_index=True)


def build_dataset(geojson_path: str) -> pd.DataFrame:
    pos = load_positives(geojson_path)
    neg = sample_negatives(pos)
    return pd.concat([pos, neg], ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Terrain features (elevation + slope)
# ---------------------------------------------------------------------------
def _key(lat: float, lon: float) -> str:
    return f"{lat:.5f},{lon:.5f}"


def fetch_elevation_batch(lats, lons, retries: int = 4):
    """Ask Open-Meteo for the elevation (metres) of up to 100 coordinates."""
    url = (f"{ELEVATION_URL}?latitude={','.join(f'{v:.5f}' for v in lats)}"
           f"&longitude={','.join(f'{v:.5f}' for v in lons)}")
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                elev = json.load(resp)["elevation"]
            if len(elev) != len(lats):
                raise ValueError("API returned a different number of values than requested")
            return elev
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimited() from e
            last_err = e
        except Exception as e:  # network hiccup, timeout, bad JSON...
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Elevation lookup failed after {retries} tries: {last_err}")


def _load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    c = pd.read_csv(CACHE_PATH)
    return dict(zip(c["key"], c["elevation"]))


def _save_cache(cache: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({"key": list(cache.keys()), "elevation": list(cache.values())}).to_csv(
        CACHE_PATH, index=False)


def add_terrain(df: pd.DataFrame, pause_s: float = 0.15) -> pd.DataFrame:
    """Adds elevation_m and slope_deg. Each point needs 3 lookups: the point
    itself, a point ~110 m north and a point ~110 m east; slope comes from the
    height differences."""
    df = df.copy()
    north = df["latitude"] + SLOPE_OFFSET_DEG
    east = df["longitude"] + SLOPE_OFFSET_DEG

    needed = {}
    for lat, lon, n_lat, e_lon in zip(df["latitude"], df["longitude"], north, east):
        needed[_key(lat, lon)] = (lat, lon)
        needed[_key(n_lat, lon)] = (n_lat, lon)
        needed[_key(lat, e_lon)] = (lat, e_lon)

    cache = _load_cache()
    todo = [(k, v) for k, v in needed.items() if k not in cache]
    total_batches = math.ceil(len(todo) / BATCH_SIZE)
    if todo:
        print(f"Looking up elevations: {len(todo)} coordinates in {total_batches} requests "
              f"({len(needed) - len(todo)} already cached)...")

    try:
        for b in range(total_batches):
            chunk = todo[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            elev = fetch_elevation_batch([c[1][0] for c in chunk], [c[1][1] for c in chunk])
            for (k, _), z in zip(chunk, elev):
                cache[k] = z if z is not None else float("nan")
            if (b + 1) % 25 == 0 or b + 1 == total_batches:
                _save_cache(cache)
                print(f"  {b + 1}/{total_batches} requests done")
            time.sleep(pause_s)
    except RateLimited:
        _save_cache(cache)
        raise SystemExit(
            "\nThe elevation service asked us to slow down (rate limit). Progress is saved.\n"
            "Wait a few minutes (or until tomorrow) and run the same command again; "
            "it will continue where it stopped.")

    z0 = np.array([cache[_key(a, o)] for a, o in zip(df["latitude"], df["longitude"])])
    zn = np.array([cache[_key(a, o)] for a, o in zip(north, df["longitude"])])
    ze = np.array([cache[_key(a, o)] for a, o in zip(df["latitude"], east)])

    lat_rad = np.radians(df["latitude"].values)
    dy_m = SLOPE_OFFSET_DEG * 110540.0
    dx_m = SLOPE_OFFSET_DEG * 111320.0 * np.cos(lat_rad)
    dz_dy = (zn - z0) / dy_m
    dz_dx = (ze - z0) / dx_m

    df["elevation_m"] = z0
    df["slope_deg"] = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    before = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} points with no elevation data.")
    return df


def add_terrain_from_dem(df: pd.DataFrame, dem_path: str) -> pd.DataFrame:
    """Same two features (elevation_m, slope_deg) but read from a local DEM
    file (GeoTIFF) instead of the web. No internet, no rate limits, fast.
    Slope is measured from the height of the pixels to the east/west and
    north/south of each point."""
    try:
        import rasterio
    except ImportError:
        raise SystemExit("rasterio is not installed. Run: pip install rasterio")

    df = df.copy()
    lat = df["latitude"].values
    lon = df["longitude"].values

    with rasterio.open(dem_path) as src:
        if src.crs is None or not src.crs.is_geographic:
            raise SystemExit(
                f"The DEM is not in latitude/longitude coordinates (its crs is: {src.crs}).\n"
                "Send that line to Claude and we will adapt the script.")
        rx, ry = abs(src.res[0]), abs(src.res[1])   # pixel size in degrees
        b = src.bounds
        print(f"Reading elevation from the DEM file ({dem_path})...")

        steps = {"c": (0, 0), "e": (rx, 0), "w": (-rx, 0), "n": (0, ry), "s": (0, -ry)}
        z = {}
        for name, (dx, dy) in steps.items():
            xs, ys = lon + dx, lat + dy
            vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype="float64")
            outside = ~((xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top))
            bad = outside | (vals < -100)            # off the map, or "no data" fillers like -32768
            if src.nodata is not None:
                bad |= np.isclose(vals, src.nodata)
            vals[bad] = np.nan
            z[name] = vals

    dx_m = rx * 111320.0 * np.cos(np.radians(lat))   # pixel width in metres at this latitude
    dy_m = ry * 110540.0                              # pixel height in metres
    dz_dx = (z["e"] - z["w"]) / (2 * dx_m)
    dz_dy = (z["n"] - z["s"]) / (2 * dy_m)

    df["elevation_m"] = z["c"]
    df["slope_deg"] = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    before = len(df)
    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    if len(df) < before:
        print(f"Dropped {before - len(df)} points that fall outside the DEM or have no data.")
    return df


# ---------------------------------------------------------------------------
# 4. Train + verify
# ---------------------------------------------------------------------------
def count_splits(model_path: str) -> int:
    with open(model_path) as f:
        m = json.load(f)
    trees = m["learner"]["gradient_booster"]["model"]["trees"]
    return sum(1 for t in trees for left in t["left_children"] if left != -1)


def train(df: pd.DataFrame):
    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit("xgboost is not installed. Run: pip install xgboost")

    def make_model():
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, eval_metric="logloss", random_state=42,
        )

    X, y, groups = df[FEATURES], df["landslide"], df["area"]

    # Honest check: test on whole map sheets the model never saw.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    check = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = check.predict_proba(X.iloc[test_idx])[:, 1]
    print(f"Held-out map sheets  ROC-AUC:       {roc_auc_score(y.iloc[test_idx], proba):.3f}"
          "   (0.5 = coin flip, 1.0 = perfect)")
    print(f"Held-out map sheets  Avg precision: {average_precision_score(y.iloc[test_idx], proba):.3f}")

    # Final model uses all the data.
    final = make_model().fit(X, y)
    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, "ner_landslide_xgb_model.json")
    final.save_model(model_path)

    n_splits = count_splits(model_path)
    print(f"\nSaved: {model_path}")
    print(f"Splits inside the trees: {n_splits}")
    if n_splits == 0:
        raise SystemExit("FAIL: the model learned nothing. Check the class balance printed above.")
    print("PASS: the model has real splits.")
    print("Feature importance:", dict(zip(FEATURES, [round(float(v), 3) for v in final.feature_importances_])))


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: python build_and_train.py "GSI_Landslide_Inventory_DATASET.geojson" ["ner_dem.tif"]')

    dem_path = sys.argv[2] if len(sys.argv) == 3 else None

    df = build_dataset(sys.argv[1])
    print("Rows per class (0 = no landslide, 1 = landslide):")
    print(df["landslide"].value_counts().to_string())
    print(f"Map sheets (areas): {df['area'].nunique()}\n")

    df = add_terrain_from_dem(df, dem_path) if dem_path else add_terrain(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "ner_landslide_training_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset: {csv_path}\n")

    train(df)


if __name__ == "__main__":
    main()
