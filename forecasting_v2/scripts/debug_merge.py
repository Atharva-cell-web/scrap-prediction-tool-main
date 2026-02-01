import pandas as pd
import os
import glob

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
SCRAP_FILE = os.path.join(DATA_DIR, "daily_scrap_forecast.csv")

def debug_mismatch():
    print("🕵️ DEBUGGING MERGE FAILURE...")
    
    # 1. Check Scrap File Names
    if os.path.exists(SCRAP_FILE):
        df_scrap = pd.read_csv(SCRAP_FILE)
        print(f"\n📄 SCRAP FILE (Target):")
        print(f"   Unique Machine IDs: {df_scrap['machine_id'].unique()}")
        print(f"   Date Format Example: {df_scrap['date'].iloc[0]}")
    else:
        print("❌ Scrap file missing.")

    # 2. Check Sensor File Names
    files = glob.glob(os.path.join(DATA_DIR, "M*Jan.csv"))
    if len(files) > 0:
        target = files[0] # Just check the first one
        df_sensor = pd.read_csv(target, nrows=1000) # Read a bit
        
        # Check if 'machine_definition' exists
        col_name = 'machine_definition' if 'machine_definition' in df_sensor.columns else 'machine_id'
        
        if col_name in df_sensor.columns:
            print(f"\n📄 SENSOR FILE ({os.path.basename(target)}):")
            print(f"   Unique IDs found: {df_sensor[col_name].unique()}")
            print(f"   Date Format Example: {df_sensor['date'].iloc[0]}")
        else:
            print(f"❌ Column '{col_name}' not found in sensor file.")
    else:
        print("❌ No sensor files found.")

    print("\n👉 COMPARE THE TWO LISTS ABOVE!")
    print("   Does one have a hyphen (-) and the other does not?")

if __name__ == "__main__":
    debug_mismatch()