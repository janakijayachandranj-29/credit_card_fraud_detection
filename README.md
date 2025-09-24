This project is a Credit Card Fraud Detection Web App built with Flask.

Users can either upload a CSV file of transactions or enter a single transaction manually.

The app uses a trained ML model with preprocessing (scaling + label encoding) to predict whether a transaction is Fraudulent or Legit.

Results are shown on a clean HTML interface (welcome, input, and output pages).

Frontend communicates with Flask via REST API endpoints (/predict for single, /predict_csv for batch).
