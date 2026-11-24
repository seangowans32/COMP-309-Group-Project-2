
from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)


ROOT = Path(__file__).resolve().parents[1]


model = joblib.load(ROOT / "models" / "hotspot_model.joblib")


with open(ROOT / "models" / "hotspot_model.json", encoding="utf-8") as file:
    config = json.load(file)


data = pd.read_csv(ROOT / "data" / "features_lookup.csv")

FEATURES = config["features"]
THRESHOLD = config["threshold"]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Auto Theft Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify({"error": "Please provide a JSON object"}), 400

    neighbourhood = body.get("neighbourhood")
    forecast_month = body.get("forecast_month")

    if not isinstance(neighbourhood, str) or not isinstance(forecast_month, str):
        return jsonify({
            "error": "neighbourhood and forecast_month must be strings"
        }), 400

  
    match = data[
        (data["neighbourhood"] == neighbourhood) &
        (data["forecast_month"] == forecast_month)
    ]

    if match.empty:
        return jsonify({
            "error": "Neighbourhood or forecast month not found"
        }), 404

    row = match.iloc[[0]]


    probability = float(model.predict_proba(row[FEATURES])[0][1])
    is_hotspot = probability >= THRESHOLD

    return jsonify({
        "neighbourhood": neighbourhood,
        "forecast_month": forecast_month,
        "predicted_hotspot": bool(is_hotspot),
        "hotspot_probability": round(probability, 4),
        "threshold": THRESHOLD,
        "recent_theft_count": int(row["last1"].iloc[0]),
        "explanation": (
            "Predicted to be a hotspot relative to other Toronto neighbourhoods."
            if is_hotspot else
            "Not predicted to be a hotspot relative to other Toronto neighbourhoods."
        )
    })


@app.route("/neighbourhoods", methods=["GET"])
def get_neighbourhoods():
    neighbourhoods = sorted(data["neighbourhood"].dropna().unique().tolist())

    return jsonify({
        "count": len(neighbourhoods),
        "neighbourhoods": neighbourhoods
    })


@app.route("/months", methods=["GET"])
def get_months():
    months = sorted(data["forecast_month"].dropna().unique().tolist())

    return jsonify({
        "count": len(months),
        "months": months
    })

if __name__ == "__main__":
    app.run(debug=True)
