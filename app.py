from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("models/house_price_SVR_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# Load dataset (use relative path for deployment)
dataset_path = "dataset/house_prices.csv"
df = pd.read_csv(dataset_path)

# Columns expected by model
required_columns = [
    "id", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "view", "grade", "sqft_above", "sqft_basement", "yr_built",
    "yr_renovated", "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        method = request.form.get("method")

        if method == "dataset":
            row_index = int(request.form.get("row_index"))
            if row_index < 0 or row_index >= len(df):
                return render_template("index.html", prediction_text="❌ Invalid row index.")
            features = df.loc[row_index, required_columns].values.astype(float)

        elif method == "manual":
            features = []
            for col in required_columns:
                val = request.form.get(col)
                if val is None or val.strip() == "":
                    return render_template("index.html", prediction_text=f"❌ Missing input: {col}")
                features.append(float(val))
        else:
            return render_template("index.html", prediction_text="❌ Please select a valid input method.")

        # Scale and predict
        scaled_input = scaler.transform([features])
        prediction = model.predict(scaled_input)[0]
        formatted_prediction = f"${prediction:,.2f}"

        return render_template("index.html", prediction_text=f"✅ Predicted House Price: {formatted_prediction}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {e}")

# This runs the app on port assigned by environment (for Render, Railway, etc.)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
