import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")
INPUT_FILE = os.path.join(DATA_DIR, "training_data_full.csv")

def train_sensor_brain():
    print("🚀 STARTING: Training the Physics-Based AI...")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: 'training_data_full.csv' not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"   ✅ Loaded {len(df)} rows of Sensor+Scrap data.")

    # 1. Identify Features
    # Everything is a feature EXCEPT: Date, IDs, and the Target (Scrap)
    ignore_cols = ['date', 'machine_id', 'tool_id', 'part_number', 'scrap_quantity', 'machine_definition']
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    print(f"   🧠 Learning from {len(feature_cols)} sensors:")
    print(f"      Examples: {feature_cols[:5]}...")
    
    # 2. Encode Text IDs (Machine/Tool) to Numbers
    le_machine = LabelEncoder()
    df['machine_code'] = le_machine.fit_transform(df['machine_id'])
    
    # Add machine code to features
    final_features = ['machine_code'] + feature_cols
    
    # 3. Prepare Data
    X = df[final_features]
    y = df['scrap_quantity']
    
    # Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train XGBoost
    print(f"   -> Training on {len(X_train)} examples...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print("-" * 50)
    print(f"📊 PHYSICS MODEL ACCURACY:")
    print(f"   📉 Average Error: ±{mae:.1f} scrap units")
    print("-" * 50)

    # 6. Save
    joblib.dump(model, os.path.join(MODEL_DIR, "sensor_model_xgb.pkl"))
    joblib.dump(final_features, os.path.join(MODEL_DIR, "sensor_features.pkl"))
    joblib.dump(le_machine, os.path.join(MODEL_DIR, "machine_encoder.pkl"))
    
    print(f"✅ Physics Brain Saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train_sensor_brain()