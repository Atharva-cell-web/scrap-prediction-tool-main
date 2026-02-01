import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models") # forecasting_v2/models
INPUT_FILE = os.path.join(DATA_DIR, "daily_scrap_forecast.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

def train_forecaster():
    print("🚀 STARTING: Training the Forecasting AI (XGBoost)...")
    
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print("❌ Error: processed data not found. Run script 1 first.")
        return
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"   ✅ Loaded {len(df)} daily records.")

    # 2. Create "Time Travel" Features (Lags)
    # The AI needs to know the PAST to predict the FUTURE.
    print("   -> Creating History Features (Lags)...")
    
    # Sort just in case
    df = df.sort_values(by=['machine_id', 'tool_id', 'part_number', 'date'])

    # Lag 1 = "Scrap Quantity Yesterday"
    df['scrap_lag_1'] = df.groupby(['machine_id', 'tool_id', 'part_number'])['scrap_quantity'].shift(1)
    
    # Lag 2 = "Scrap Quantity 2 Days Ago"
    df['scrap_lag_2'] = df.groupby(['machine_id', 'tool_id', 'part_number'])['scrap_quantity'].shift(2)
    
    # Rolling Mean = "Average Scrap of Last 3 Days"
    df['rolling_avg_3'] = df.groupby(['machine_id', 'tool_id', 'part_number'])['scrap_quantity'].transform(lambda x: x.rolling(3).mean())

    # Drop the first few days where history is empty (NaN)
    df_clean = df.dropna()
    print(f"   -> {len(df_clean)} rows ready for training (after removing empty history).")
    
    if len(df_clean) < 10:
        print("⚠️ WARNING: Very little data left after cleaning! The model might be weak.")

    # 3. Convert Names to Numbers (Encoding)
    le_machine = LabelEncoder()
    df_clean['machine_code'] = le_machine.fit_transform(df_clean['machine_id'])
    
    le_tool = LabelEncoder()
    df_clean['tool_code'] = le_tool.fit_transform(df_clean['tool_id'])
    
    le_part = LabelEncoder()
    df_clean['part_code'] = le_part.fit_transform(df_clean['part_number'])

    # 4. Split Train vs Test (Past vs Future)
    # We train on first 80%, Test on last 20%
    dates = df_clean['date'].sort_values().unique()
    split_idx = int(len(dates) * 0.8)
    
    # Safety check for very small data
    if split_idx == 0: split_idx = 1 
    
    split_date = dates[split_idx]
    print(f"   -> Splitting data at date: {pd.to_datetime(split_date).date()}")
    
    train = df_clean[df_clean['date'] < split_date]
    test = df_clean[df_clean['date'] >= split_date]
    
    features = ['machine_code', 'tool_code', 'part_code', 'scrap_lag_1', 'scrap_lag_2', 'rolling_avg_3']
    target = 'scrap_quantity'
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # 5. Train XGBoost
    print(f"   -> Training on {len(X_train)} rows...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate
    if len(X_test) > 0:
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print("-" * 60)
        print(f"📊 RESULTS:")
        print(f"   📉 Average Error (MAE): ±{mae:.1f} parts")
        print("-" * 60)
    else:
        print("⚠️ Not enough data to test properly, but model is saved.")

    # 7. Save Brain
    joblib.dump(model, os.path.join(MODEL_DIR, "forecast_xgb.pkl"))
    joblib.dump({
        'machine': le_machine,
        'tool': le_tool,
        'part': le_part
    }, os.path.join(MODEL_DIR, "encoders.pkl"))
    
    print(f"✅ Brain Saved to: {MODEL_DIR}")

if __name__ == "__main__":
    train_forecaster()