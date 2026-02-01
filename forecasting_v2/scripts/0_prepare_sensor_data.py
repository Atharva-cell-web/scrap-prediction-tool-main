import pandas as pd
import os
import glob

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
SCRAP_FILE = os.path.join(DATA_DIR, "daily_scrap_forecast.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "training_data_full.csv")

def prepare_data():
    print("🚀 STARTING: Merging Sensors with Scrap Data (Final Fix)...")
    
    # 1. Load Scrap Data
    if not os.path.exists(SCRAP_FILE):
        print("❌ Error: Daily scrap file not found.")
        return
    scrap_df = pd.read_csv(SCRAP_FILE)
    scrap_df['date'] = pd.to_datetime(scrap_df['date'])
    print(f"   ✅ Loaded Scrap History ({len(scrap_df)} rows).")

    # 2. Process Machine Files
    all_files = glob.glob(os.path.join(DATA_DIR, "M*Jan.csv"))
    print(f"   Found {len(all_files)} machine files.")
    
    sensor_data_list = []
    
    for f in all_files:
        try:
            # Read file
            df = pd.read_csv(f, low_memory=False)
            
            # --- FIX 1: CLEAN MACHINE NAMES ---
            # Turn "M231-11" -> "M-231"
            # Step A: "M231-11" -> "M-231-11" (Add hyphen after M)
            if 'machine_definition' in df.columns:
                 # Ensure string format
                df['machine_definition'] = df['machine_definition'].astype(str)
                
                # If it looks like "M231...", insert the hyphen
                # This logic splits "M231-11" into parts and reconstructs "M-231"
                def clean_name(name):
                    if name.startswith("M") and "-" not in name[:2]:
                         # "M231-11" -> "M-231"
                         return name.replace("M", "M-").split("-")[0] + "-" + name.replace("M", "M-").split("-")[1]
                    return name

                df['machine_id'] = df['machine_definition'].apply(clean_name)
            else:
                # Fallback if column missing
                df['machine_id'] = os.path.basename(f)[:5] # Take "M231J" (Risky, but fallback)

            # --- FIX 2: FORCE NUMBERS ---
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            
            # Fix Date
            df['date'] = pd.to_datetime(df['date'])
            
            # Pivot
            print(f"      -> Pivoting {os.path.basename(f)} (ID: {df['machine_id'].iloc[0]})...")
            daily_sensors = df.pivot_table(
                index=['date', 'machine_id'], 
                columns='variable_name', 
                values='value', 
                aggfunc='mean'
            ).reset_index()
            
            sensor_data_list.append(daily_sensors)
            
        except Exception as e:
            print(f"      ❌ Failed {os.path.basename(f)}: {e}")

    if not sensor_data_list:
        print("❌ No data processed.")
        return

    # 3. Merge
    full_sensor_df = pd.concat(sensor_data_list, ignore_index=True)
    print(f"   ✅ Combined Sensor Data: {len(full_sensor_df)} rows.")

    print("   -> Merging Sensors + Scrap...")
    merged_df = pd.merge(scrap_df, full_sensor_df, on=['date', 'machine_id'], how='inner')
    merged_df = merged_df.fillna(0)
    
    # 4. Save
    if len(merged_df) > 0:
        merged_df.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 SAVED: {OUTPUT_FILE}")
        print(f"   ✅ Rows Matched: {len(merged_df)}")
    else:
        print("❌ CRITICAL: Merge resulted in 0 rows.")
        print("   (The dates in Scrap vs Sensors might not overlap!)")

if __name__ == "__main__":
    prepare_data()