# File: model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------- Load Dataset ----------------
df = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/AI PRACTICE/credit_card_fraud_dataset.csv")

# ---------------- Explore Location column ----------------
print("📍 First 10 values in Location column:\n", df["Location"].head(10))
print("\n📍 Unique values in Location column:\n", df["Location"].unique())
print("\n📍 Number of unique locations:", df["Location"].nunique())

# ---------------- Encode Categorical Features ----------------
enc_type = LabelEncoder()
enc_loc = LabelEncoder()

df["TransactionType"] = enc_type.fit_transform(df["TransactionType"].astype(str))
df["Location"] = enc_loc.fit_transform(df["Location"].astype(str))

print("\n📍 Location encoding mapping:")
for label, code in zip(enc_loc.classes_, range(len(enc_loc.classes_))):
    print(f"  {label} -> {code}")

# ---------------- Normalize target so 0 == Fraud, 1 == Not Fraud ----------------
raw_target = df["IsFraud"]
print("\nOriginal IsFraud value counts:\n", raw_target.value_counts(dropna=False))

# Create y such that 0 => Fraud, 1 => Not Fraud
if raw_target.dropna().dtype.kind in "biuf" and set(raw_target.dropna().unique()) <= {0, 1}:
    counts = raw_target.value_counts()
    minority_label = counts.idxmin()
    if minority_label == 1:
        print("Detected 1 as minority -> assuming 1 means FRAUD. Inverting so 0 == Fraud.")
        y = 1 - raw_target.astype(int)
        target_map = {0: "Fraud (was 1)", 1: "Not Fraud (was 0)"}
    else:
        print("Detected 0 as minority or balanced -> keeping mapping so 0 == Fraud.")
        y = raw_target.astype(int)
        target_map = {0: "Fraud", 1: "Not Fraud"}
else:
    def map_to_binary(v):
        s = str(v).strip().lower()
        if s in ("1", "yes", "y", "true", "t", "fraud", "fraudulent"):
            return 0
        return 1
    y = raw_target.apply(map_to_binary).astype(int)
    target_map = {0: "Fraud", 1: "Not Fraud"}

print("\nMapped target counts (0=Fraud, 1=Not Fraud):\n", y.value_counts())

# ---------------- Features & Target ----------------
X = df.drop(["TransactionDate", "IsFraud", "TransactionID"], axis=1, errors="ignore")

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ---------------- Scale Numeric Features ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- Train Model ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---------------- Evaluate ----------------
acc = model.score(X_test_scaled, y_test)
print(f"\n✅ Model trained successfully. Accuracy: {acc:.3f}")

# ---------------- Save Artifacts ----------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump({"TransactionType": enc_type, "Location": enc_loc}, f)

with open("target_map.pkl", "wb") as f:
    pickle.dump(target_map, f)

print("\n✅ Artifacts saved: model.pkl, scaler.pkl, encoders.pkl, target_map.pkl")
print("Note: model.predict(...) will now return 0 for Fraud and 1 for Not Fraud.")
