import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_ml_dataset.parquet")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    print(" Starting Final Model Training...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)
    print(f" Loaded Dataset: {len(df)} rows, {len(df.columns)} columns")

    # 2. Define Target
    if 'actual_scrap_qty' not in df.columns:
        print("❌ Error: 'actual_scrap_qty' column missing.")
        return

    # Create Target (High Risk = 1, Low Risk = 0)
    threshold = df['actual_scrap_qty'].quantile(0.75)
    df['target'] = (df['actual_scrap_qty'] > threshold).astype(int)
    print(f"Risk Threshold: > {threshold:.0f} scrap parts")

    # 3. SAFETY FILTER: Select Features
    # First, drop known cheating columns
    forbidden = ['target', 'actual_scrap_qty', 'actual_qty', 'machine_id', 'date', 
                 'order_start', 'order_stop', 'shift_production_time', 'shift_scrap']
    
    # Drop columns that contain "actual" or "shift" or are in forbidden list
    cols_to_drop = [c for c in df.columns if c in forbidden or 'actual' in c or 'shift' in c]
    features = df.drop(columns=cols_to_drop)

    # --- THE FIX: FORCE NUMERIC ONLY ---
    print("Cleaning data...")
    
    # Try to convert everything to numbers. If it fails (like a date), turn it into NaN.
    features = features.apply(pd.to_numeric, errors='coerce')
    
    # Drop columns that are completely empty/broken after conversion
    features = features.dropna(axis=1, how='all')
    
    # Fill remaining missing values with 0
    features = features.fillna(0)
    
    # Final check: Keep ONLY Valid Numbers
    X = features.select_dtypes(include=[np.number])
    y = df['target']
    
    print(f"Final Training Features: {X.shape[1]} columns")

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train Random Forest
    print("   -> Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 6. Evaluate
    print("\n Model Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 7. Save
    joblib.dump(model, os.path.join(MODEL_DIR, "final_scrap_model.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "final_model_features.pkl"))
    print(f"Model Saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train_model()