import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


@app.route("/")
def home():
    return render_template_string("""
        <!doctype html>
        <html>
        <head>
            <title>California Housing Predictor</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                form { max-width: 500px; }
                .row { margin-bottom: 10px; }
                input { width: 100%; padding: 8px; margin-top: 5px; }
                button { padding: 10px 15px; background: #007bff; color: white; border: none; cursor: pointer; }
            </style>
        </head>
        <body>
            <h2>California Housing Price Prediction</h2>
            <form action="/predict" method="POST">
                {% for feature in ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude'] %}
                    <div class="row">
                        <label>{{ feature }}</label>
                        <input type="number" step="any" name="{{ feature }}" required>
                    </div>
                {% endfor %}
                <button type="submit">Predict</button>
            </form>
        </body>
        </html>
    """)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not found. Save your trained model as model.pkl first."}), 500

    form_data = request.form.to_dict()
    input_values = [float(form_data.get(feature, 0)) for feature in FEATURES]
    input_df = pd.DataFrame([input_values], columns=FEATURES)

    prediction = model.predict(input_df)[0]
    return jsonify({
        "predicted_house_value": round(float(prediction), 2),
        "features_used": FEATURES,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model not found. Save your trained model as model.pkl first."}), 500

    data = request.get_json(silent=True)
    if data is None:
        data = request.form.to_dict()

    input_values = [float(data.get(feature, 0)) for feature in FEATURES]
    input_df = pd.DataFrame([input_values], columns=FEATURES)
    prediction = model.predict(input_df)[0]

    return jsonify({
        "prediction": round(float(prediction), 2)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

