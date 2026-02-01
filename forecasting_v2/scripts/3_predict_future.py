import pandas as pd
import joblib
import os

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")

MODEL_PATH = os.path.join(MODEL_DIR, "forecast_xgb.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
DATA_PATH = os.path.join(DATA_DIR, "daily_scrap_forecast.csv")

def predict_future():
    print("\n🔮 AI FORECAST SYSTEM: PREDICTING TOMORROW")
    print("=" * 60)

    # 1. Load the Brains
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model not found. Run Script 2 first.")
        return
    
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    
    # 2. Load History (To find valid Machines/Tools)
    df = pd.read_csv(DATA_PATH)
    
    # Let's pick the "Most Active" combination to test automatically
    # (So you don't have to type it in manually and get errors)
    most_active = df.groupby(['machine_id', 'tool_id', 'part_number']).size().idxmax()
    test_machine, test_tool, test_part = most_active
    
    print(f"📋 TEST SCENARIO (Auto-Selected):")
    print(f"   Machine: {test_machine}")
    print(f"   Tool:    {test_tool}")
    print(f"   Part:    {test_part}")
    
    # 3. Get Recent History (The Context)
    # We need the LAST known data points to predict the NEXT one
    subset = df[
        (df['machine_id'] == test_machine) & 
        (df['tool_id'] == test_tool) & 
        (df['part_number'] == test_part)
    ].sort_values(by='date')
    
    if len(subset) < 3:
        print("❌ Not enough history for this tool to predict.")
        return

    # Grab the last known values
    last_row = subset.iloc[-1]
    scrap_yesterday = last_row['scrap_quantity']
    scrap_2_days_ago = subset.iloc[-2]['scrap_quantity']
    avg_last_3 = subset.iloc[-3:]['scrap_quantity'].mean()
    
    print(f"   History: {scrap_yesterday} (Yesterday), {scrap_2_days_ago} (2 Days Ago)")
    print("-" * 60)

    # 4. Prepare Input for AI
    try:
        # Encode inputs (Convert "M-607" -> 5)
        m_code = encoders['machine'].transform([test_machine])[0]
        t_code = encoders['tool'].transform([test_tool])[0]
        p_code = encoders['part'].transform([test_part])[0]
        
        # Create the feature row
        # Must match training order: [machine, tool, part, lag1, lag2, rolling3]
        features = pd.DataFrame([[
            m_code, t_code, p_code, 
            scrap_yesterday, scrap_2_days_ago, avg_last_3
        ]], columns=['machine_code', 'tool_code', 'part_code', 'scrap_lag_1', 'scrap_lag_2', 'rolling_avg_3'])
        
        # 5. Predict
        prediction = model.predict(features)[0]
        
        print(f"\n🚀 AI PREDICTION FOR TOMORROW:")
        print(f"   Expected Scrap: {prediction:,.0f} units")
        print(f"   (Based on trends from Tool {test_tool})")
        
        if prediction > 100:
            print("   ⚠️ STATUS: HIGH RISK. Schedule Maintenance.")
        else:
            print("   ✅ STATUS: Normal Operation.")
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

if __name__ == "__main__":
    predict_future()