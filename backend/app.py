from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import pandas as pd
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "superconductor_model.json"
DATA_PATH = ROOT / "Super-Conductor-Critical-Temperature-Prediction-main" / "CriticalTemperature.csv"
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://cyro-nex-rqob.vercel.app",
]}})
model = xgb.XGBRegressor()
model.load_model(str(MODEL_PATH))
FEATURES = list(model.feature_names_in_)
TARGET = "critical_temp"

def dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def feature_groups():
    families = ["atomic_mass", "Fie", "atomic_radius", "Density", "ElectronAffinity", "FusionHeat", "ThermalConductivity", "Valence"]
    groups = [{"name": "Composition", "features": [FEATURES[0]]}]
    for family in families:
        names = [f for f in FEATURES if family.lower() in f.lower()]
        if names:
            groups.append({"name": family.replace("_", " ").title(), "features": names})
    assigned = {f for group in groups for f in group["features"]}
    remaining = [f for f in FEATURES if f not in assigned]
    if remaining: groups.append({"name": "Other descriptors", "features": remaining})
    return groups

@app.get("/")
def home():
    return jsonify({"name": "CryoNex API", "features_required": len(FEATURES), "dataset_available": DATA_PATH.exists()})

@app.get("/features")
def features():
    return jsonify({"features": FEATURES, "groups": feature_groups(), "count": len(FEATURES), "target": TARGET})

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict): return jsonify({"error": "Send a JSON object containing all model features."}), 400
    missing = [f for f in FEATURES if f not in data]
    if missing: return jsonify({"error": "Missing required features.", "missing_features": missing}), 400
    try:
        if any(isinstance(data[f], bool) for f in FEATURES):
            raise ValueError("All feature values must be finite numbers.")
        vals = [float(data[f]) for f in FEATURES]
        if not all(math.isfinite(v) for v in vals): raise ValueError("All feature values must be finite numbers.")
        result = float(model.predict(pd.DataFrame([vals], columns=FEATURES))[0])
        return jsonify({"predicted_critical_temperature_K": result, "unit": "Kelvin", "label": "Model estimate"})
    except (ValueError, TypeError) as e: return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed."}), 500

@app.post("/model/explain")
def explain():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object containing all model features."}), 400
    missing = [feature for feature in FEATURES if feature not in data]
    if missing:
        return jsonify({"error": "Missing required features.", "missing_features": missing}), 400
    try:
        if any(isinstance(data[feature], bool) for feature in FEATURES):
            raise ValueError("All feature values must be finite numbers.")
        values = [float(data[feature]) for feature in FEATURES]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("All feature values must be finite numbers.")
        frame = pd.DataFrame([values], columns=FEATURES)
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    try:
        matrix = xgb.DMatrix(frame, feature_names=FEATURES)
        contributions = model.get_booster().predict(matrix, pred_contribs=True)[0]
        prediction = float(model.predict(frame)[0])
        contribution_values = [float(value) for value in contributions]
        base_value = contribution_values[-1]
        feature_contributions = contribution_values[:-1]
        if len(feature_contributions) != len(FEATURES) or not all(
            math.isfinite(value) for value in contribution_values
        ):
            raise ValueError("The model returned invalid explanation values.")
        if not math.isclose(
            sum(contribution_values),
            prediction,
            rel_tol=1e-4,
            abs_tol=1e-4,
        ):
            raise ValueError("The explanation does not reconcile with the model prediction.")
        return jsonify({
            "method": "XGBoost TreeSHAP contributions",
            "prediction": prediction,
            "base_value": base_value,
            "contributions": [
                {
                    "feature": feature,
                    "value": values[index],
                    "contribution": feature_contributions[index],
                }
                for index, feature in enumerate(FEATURES)
            ],
        })
    except Exception:
        app.logger.exception("Model explanation failed")
        return jsonify({"error": "Model explanation is unavailable."}), 503

@app.get("/sample")
def sample():
    df = dataset()
    if TARGET not in df or any(f not in df for f in FEATURES): return jsonify({"error": "Dataset schema does not match the model."}), 500
    idx = int(request.args.get("index", 0))
    if idx < 0 or idx >= len(df): return jsonify({"error": "Sample index is out of range."}), 404
    row = df.iloc[idx]
    return jsonify({"index": idx, "features": {f: float(row[f]) for f in FEATURES}, "critical_temp": float(row[TARGET]), "unit": "Kelvin"})

@app.get("/dataset/summary")
def summary():
    df = dataset()
    y = pd.to_numeric(df[TARGET], errors="coerce").dropna()
    clean = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    bins = pd.cut(y, bins=24)
    hist = y.groupby(bins, observed=False).size()
    return jsonify({"rows": int(len(df)), "features": len(FEATURES), "target": TARGET,
        "missing_values": int(df.isna().sum().sum()), "target_stats": {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "median": float(y.median()), "std": float(y.std())},
        "distribution": [{"range": str(k), "count": int(v), "mid": float(k.mid)} for k,v in hist.items() if pd.notna(k)],
        "feature_stats": [{"feature": f, "mean": None if pd.isna(clean[f].mean()) else float(clean[f].mean()), "std": None if pd.isna(clean[f].std()) else float(clean[f].std()), "min": None if pd.isna(clean[f].min()) else float(clean[f].min()), "max": None if pd.isna(clean[f].max()) else float(clean[f].max()), "missing": int(clean[f].isna().sum())} for f in FEATURES]})

@app.get("/dataset/records")
def records():
    df = dataset()
    q = request.args.get("q", "").strip()
    page = max(1, int(request.args.get("page", 1)))
    size = min(100, max(1, int(request.args.get("page_size", 20))))
    if q:
        numeric = pd.to_numeric(q, errors="coerce")
        if pd.notna(numeric): df = df[(df[TARGET] - numeric).abs() < 0.000001]
        else: return jsonify({"rows": [], "total": 0, "page": page, "page_size": size})
    total = len(df); part = df.iloc[(page-1)*size:page*size]
    rows = [{**{f: (None if pd.isna(row[f]) else float(row[f])) for f in FEATURES}, TARGET: (None if pd.isna(row[TARGET]) else float(row[TARGET]))} for _,row in part.iterrows()]
    return jsonify({"rows": rows, "total": int(total), "page": page, "page_size": size})

@app.get("/model/insights")
def insights():
    imp = model.feature_importances_
    importance = sorted([{"feature": f, "importance": float(v)} for f,v in zip(FEATURES, imp)], key=lambda x: -x["importance"])
    return jsonify({"model": "XGBoost regressor", "feature_count": len(FEATURES), "importance": importance})

if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, debug=False)
