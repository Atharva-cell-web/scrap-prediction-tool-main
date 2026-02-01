import pandas as pd
import os

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
INPUT_FILE = os.path.join(DATA_DIR, "hydra_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "daily_scrap_forecast.csv")

def create_daily_dataset():
    print(f"🚀 STARTING: Aggregating Data (Correction)...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: File not found at {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"   ✅ Loaded {len(df)} raw rows.")

    # 1. Rename Columns
    # We are switching to 'actual_scrap_qty' which has the real data
    rename_map = {
        'machine_nr': 'machine_id',
        'tool_number': 'tool_id',
        'article': 'part_number',
        'actual_scrap_qty': 'scrap_quantity', # <--- USING THE BETTER COLUMN
        'date': 'date'
    }
    
    df = df.rename(columns=rename_map)
    df['date'] = pd.to_datetime(df['date'])

    # 2. AGGREGATION (THE FIX)
    # The raw data repeats the same number many times.
    # We must use MAX() to get the final count, not SUM().
    print("   -> Grouping data (Using MAX instead of SUM to fix huge numbers)...")
    
    group_cols = ['date', 'machine_id', 'tool_id', 'part_number']
    
    # OLD WAY: .sum() -> Caused 2.5 million error
    # NEW WAY: .max() -> Gets the true daily value (e.g., 3840)
    daily_df = df.groupby(group_cols)['scrap_quantity'].max().reset_index()
    
    daily_df = daily_df.sort_values(by=['machine_id', 'tool_id', 'part_number', 'date'])
    
    print(f"   ✅ Fixed! Created {len(daily_df)} Daily Summaries.")
    print("   Example Row (Real Numbers Now):")
    print(daily_df.head(1))

    # 3. Save
    daily_df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 SAVED processed data to: {OUTPUT_FILE}")

if __name__ == "__main__":
    create_daily_dataset()