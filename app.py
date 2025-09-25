# File: app.py
from flask import Flask, request, render_template, redirect, jsonify
import pandas as pd
import pickle
import os
import numpy as np
from io import StringIO

app = Flask(__name__)

# ---------------- Load Model and Preprocessors ----------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Fix: Load separate encoder files or handle properly
    # Assuming you have separate encoder files
    try:
        with open("type_encoder.pkl", "rb") as f:
            enc_type = pickle.load(f)
        with open("location_encoder.pkl", "rb") as f:
            enc_loc = pickle.load(f)
    except FileNotFoundError:
        # Fallback: create mock encoders or use the same encoder
        with open("encoders.pkl", "rb") as f:
            encoder = pickle.load(f)
        enc_type = encoder  # You'll need to modify this based on your actual encoder structure
        enc_loc = encoder
    
    print("Model and preprocessors loaded successfully")
    
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    scaler = None
    enc_type = None
    enc_loc = None


# ---------------- Helper Functions ----------------
def preprocess_single_transaction(data):
    """Preprocess a single transaction for prediction"""
    try:
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Handle categorical encoding
        if enc_type and enc_loc:
            # Check if the transaction type and location are in the encoder's vocabulary
            try:
                df["TransactionType_encoded"] = enc_type.transform([data["TransactionType"]])[0]
            except ValueError:
                # Handle unknown categories
                df["TransactionType_encoded"] = 0  # or some default value
                
            try:
                df["Location_encoded"] = enc_loc.transform([data["Location"]])[0]
            except ValueError:
                # Handle unknown categories
                df["Location_encoded"] = 0  # or some default value
        else:
            # Simple mapping for demo purposes
            type_mapping = {"purchase": 0, "withdrawal": 1, "transfer": 2, "deposit": 3}
            location_mapping = {"Mumbai": 0, "Delhi": 1, "Bangalore": 2, "Chennai": 3, 
                              "Kolkata": 4, "Hyderabad": 5, "Pune": 6}
            
            df["TransactionType_encoded"] = type_mapping.get(data["TransactionType"], 0)
            df["Location_encoded"] = location_mapping.get(data["Location"], 0)
        
        # Scale amount if scaler is available
        if scaler:
            df["Amount_scaled"] = scaler.transform([[float(data["Amount"])]])[0][0]
        else:
            df["Amount_scaled"] = float(data["Amount"])
        
        # Prepare final feature array for prediction
        # Adjust this based on your model's expected input format
        features = [
            df["Amount_scaled"].iloc[0],
            df["TransactionType_encoded"].iloc[0],
            df["Location_encoded"].iloc[0],
            # Add other features as needed
        ]
        
        return np.array([features])
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def preprocess_bulk_transactions(df):
    """Preprocess bulk transactions for prediction"""
    try:
        processed_data = []
        
        for _, row in df.iterrows():
            data = {
                "TransactionID": row.get("TransactionID", ""),
                "Amount": row.get("Amount", 0),
                "MerchantID": row.get("MerchantID", ""),
                "TransactionType": row.get("TransactionType", "purchase"),
                "Location": row.get("Location", "Mumbai")
            }
            
            features = preprocess_single_transaction(data)
            if features is not None:
                processed_data.append({
                    "original": data,
                    "features": features[0]
                })
        
        return processed_data
        
    except Exception as e:
        print(f"Error in bulk preprocessing: {e}")
        return []


# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("welcome.html")


@app.route("/input")
def input_page():
    return render_template("input.html")


@app.route("/results")
def results_page():
    return render_template("output.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for single transaction prediction"""
    try:
        if not model:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.json
        
        # Validate required fields
        required_fields = ["TransactionID", "Amount", "MerchantID", "TransactionType", "Location"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Preprocess data
        features = preprocess_single_transaction(data)
        if features is None:
            return jsonify({"error": "Error preprocessing data"}), 400
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
        
        result = {
            "TransactionID": data["TransactionID"],
            "Amount": float(data["Amount"]),
            "MerchantID": data["MerchantID"],
            "TransactionType": data["TransactionType"],
            "Location": data["Location"],
            "prediction": int(prediction),
            "probability": float(probability[1]) if len(probability) > 1 else 0.5,
            "risk_level": "High Risk" if prediction == 1 else "Low Risk"
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict_bulk", methods=["POST"])
def api_predict_bulk():
    """API endpoint for bulk transaction prediction"""
    try:
        if not model:
            return jsonify({"error": "Model not loaded"}), 500
        
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Read CSV file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400
        
        # Validate CSV structure
        required_columns = ["TransactionID", "Amount", "MerchantID", "TransactionType", "Location"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns in CSV: {missing_columns}"}), 400
        
        # Preprocess data
        processed_data = preprocess_bulk_transactions(df)
        if not processed_data:
            return jsonify({"error": "No valid data to process"}), 400
        
        # Make predictions
        results = []
        for item in processed_data:
            try:
                features = np.array([item["features"]])
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else [0.5, 0.5]
                
                result = {
                    **item["original"],
                    "prediction": int(prediction),
                    "probability": float(probability[1]) if len(probability) > 1 else 0.5,
                    "risk_level": "High Risk" if prediction == 1 else "Low Risk"
                }
                results.append(result)
                
            except Exception as e:
                print(f"Error predicting for transaction {item['original'].get('TransactionID', 'Unknown')}: {e}")
                continue
        
        if not results:
            return jsonify({"error": "No valid predictions generated"}), 400
        
        return jsonify({"results": results, "total": len(results)})
        
    except Exception as e:
        print(f"Error in bulk prediction: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------- Legacy Routes (for backward compatibility) ----------------
@app.route("/predict", methods=["POST"])
def predict():
    """Legacy route for form-based single prediction"""
    try:
        # Get form data
        data = {
            "TransactionID": request.form["TransactionID"],
            "Amount": request.form["Amount"],
            "MerchantID": request.form["MerchantID"],
            "TransactionType": request.form["TransactionType"],
            "Location": request.form["Location"]
        }
        
        # Use API function
        request.json = data
        response = api_predict()
        
        if response.status_code == 200:
            result_data = response.json
            return render_template("output.html", single_result=result_data)
        else:
            return f"Error: {response.json.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error: {e}"


@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    """Legacy route for form-based bulk prediction"""
    try:
        response = api_predict_bulk()
        
        if response.status_code == 200:
            result_data = response.json
            return render_template("output.html", bulk_result=result_data.get("results", []))
        else:
            return f"Error: {response.json.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error: {e}"


# ---------------- Health Check ----------------
@app.route("/health")
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": enc_type is not None and enc_loc is not None
    }
    return jsonify(status)


# ---------------- Error Handlers ----------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "True").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)