# File: app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__, template_folder="templates")

# ---------------- Load Artifacts ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)  # {"TransactionType": encoder, "Location": encoder}


def safe_label_transform(encoder, value):
    """Return encoded integer for value or -1 for unseen / error."""
    try:
        # LabelEncoder.transform expects array-like; return scalar int
        return int(encoder.transform([value])[0])
    except Exception:
        return -1


def prepare_features_for_scaling(df_features):
    """
    Reorder features if scaler knows the feature order, coerce to numeric,
    and fill NaNs with -1 (safe default).
    """
    X = df_features.copy()

    # If scaler was fitted on a DataFrame, it may have feature_names_in_
    if hasattr(scaler, "feature_names_in_"):
        desired_order = list(scaler.feature_names_in_)
        # If some expected features are missing, add them as -1
        for c in desired_order:
            if c not in X.columns:
                X[c] = -1
        X = X[desired_order]

    # Coerce all columns to numeric where possible; non-numeric -> NaN -> fill -1
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-1)

    return X


# ---------------- Serve HTML Pages ----------------
@app.route("/")
def home():
    return render_template("welcome.html")


@app.route("/welcome.html")
def welcome_page():
    return render_template("welcome.html")


@app.route("/input.html")
def input_page():
    return render_template("input.html")


@app.route("/output.html")
def output_page():
    return render_template("output.html")


# ---------------- Predict Single Transaction ----------------
@app.route("/predict", methods=["POST"])
def predict_single():
    # Accept JSON, form data, or query string for convenience
    data = request.get_json(silent=True)
    if not data:
        data = request.form.to_dict() or request.args.to_dict() or {}

    txid = data.get("TransactionID", "")
    txdate = data.get("TransactionDate", "")
    amount = data.get("Amount", 0)
    merchant = data.get("MerchantID", "")
    ttype = data.get("TransactionType", "")
    location = data.get("Location", "")

    # Build single-row DataFrame
    df = pd.DataFrame([{
        "TransactionID": txid,
        "TransactionDate": txdate,
        "Amount": amount,
        "MerchantID": merchant,
        "TransactionType": ttype,
        "Location": location
    }])

    # Drop TransactionDate (not used by model), keep TransactionID to remove later
    X = df.drop(["TransactionDate"], axis=1)

    # Apply safe label encoding for TransactionType and Location
    X.loc[:, "TransactionType"] = X["TransactionType"].apply(
        lambda v: safe_label_transform(encoders["TransactionType"], v)
    )
    X.loc[:, "Location"] = X["Location"].apply(
        lambda v: safe_label_transform(encoders["Location"], v)
    )

    # Drop TransactionID before scaling (the model was trained without TransactionID)
    X_for_scale = X.drop(["TransactionID"], axis=1)

    # Prepare features (reorder & coerce) then scale
    X_prepared = prepare_features_for_scaling(X_for_scale)
    X_scaled = scaler.transform(X_prepared)

    # Predict
    pred = int(model.predict(X_scaled)[0])  # 0 or 1

    # IMPORTANT: 0 = Fraud, 1 = Not Fraud
    result = {
        "TransactionID": txid,
        "Amount": float(amount) if amount is not None and str(amount) != "" else None,
        "MerchantID": merchant,
        "TransactionType": ttype,
        "Location": location,
        "prediction_numeric": pred,
        "prediction_text": "Fraudulent Transaction" if pred == 0 else "Legit Transaction"
    }

    return jsonify(result)


# ---------------- Predict CSV Batch ----------------
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    df = pd.read_csv(file)

    required_cols = ["TransactionID", "TransactionDate", "Amount", "MerchantID", "TransactionType", "Location"]
    for col in required_cols:
        if col not in df.columns:
            return jsonify({"error": f"Missing column: {col}"}), 400

    # Work on a copy and drop TransactionDate
    X = df.drop(["TransactionDate"], axis=1).copy()

    # Safe encoding row by row
    X["TransactionType"] = [
        safe_label_transform(encoders["TransactionType"], val)
        for val in df["TransactionType"]
    ]
    X["Location"] = [
        safe_label_transform(encoders["Location"], val)
        for val in df["Location"]
    ]

    # Drop TransactionID prior to scaling and prepare features
    X_for_scale = X.drop(["TransactionID"], axis=1)
    X_prepared = prepare_features_for_scaling(X_for_scale)
    X_scaled = scaler.transform(X_prepared)
    preds = model.predict(X_scaled)

    # Pair each original row with its prediction (use iloc to keep correct row order)
    results = []
    for i, pred in enumerate(preds):
        row = df.iloc[i]
        results.append({
            "TransactionID": row["TransactionID"],
            "Amount": row["Amount"],
            "MerchantID": row["MerchantID"],
            "TransactionType": row["TransactionType"],
            "Location": row["Location"],
            "prediction_numeric": int(pred),
            "prediction_text": "Fraudulent Transaction" if int(pred) == 0 else "Legit Transaction"
        })

    return jsonify(results)


# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
