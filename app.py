import json
import os
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")
MAX_DAYS = 30

app = Flask(__name__)

# Dimuat sekali saat server start
model = load_model(os.path.join(ART_DIR, "lstm_model.keras"))
scaler = joblib.load(os.path.join(ART_DIR, "scaler.joblib"))
with open(os.path.join(ART_DIR, "meta.json")) as f:
    meta = json.load(f)
history = pd.read_csv(os.path.join(ART_DIR, "series.csv"), parse_dates=["Date"])
test_pred = pd.read_csv(os.path.join(ART_DIR, "test_predictions.csv"), parse_dates=["Date"])

TIMESTEPS = int(meta["timesteps"])


def dates_to_str(s):
    return [d.strftime("%Y-%m-%d") for d in s]


def forecast(days):
    """Peramalan rekursif: hasil prediksi hari ke-t dipakai sebagai input hari ke-(t+1)."""
    window = np.array(meta["last_window"], dtype="float64").reshape(-1, 1)
    scaled = scaler.transform(window).flatten().tolist()

    preds_scaled = []
    for _ in range(days):
        x = np.array(scaled[-TIMESTEPS:], dtype="float32").reshape(1, TIMESTEPS, 1)
        p = float(model.predict(x, verbose=0)[0][0])
        scaled.append(p)
        preds_scaled.append(p)

    values = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    values = np.clip(values, 0, None)

    last = date.fromisoformat(meta["last_date"])
    dates = [(last + timedelta(days=i + 1)).isoformat() for i in range(days)]
    return dates, [round(float(v), 2) for v in values]


@app.route("/")
def index():
    return render_template("index.html", meta=meta)


@app.route("/api/data")
def api_data():
    return jsonify({
        "history": {
            "dates": dates_to_str(history["Date"]),
            "revenue": [round(float(v), 2) for v in history["revenue"]],
        },
        "test": {
            "dates": dates_to_str(test_pred["Date"]),
            "actual": [round(float(v), 2) for v in test_pred["actual"]],
            "predicted": [round(float(v), 2) for v in test_pred["predicted"]],
        },
        "metrics": meta["metrics"],
        "baselines": meta["baselines"],
        "train_period": meta["train_period"],
        "test_period": meta["test_period"],
        "timesteps": TIMESTEPS,
    })


@app.route("/api/forecast")
def api_forecast():
    try:
        days = int(request.args.get("days", 7))
    except ValueError:
        return jsonify({"error": "Parameter 'days' harus berupa bilangan bulat."}), 400
    if not 1 <= days <= MAX_DAYS:
        return jsonify({"error": f"Jumlah hari harus antara 1 dan {MAX_DAYS}."}), 400

    dates, values = forecast(days)
    return jsonify({"dates": dates, "predicted": values, "last_data_date": meta["last_date"]})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
