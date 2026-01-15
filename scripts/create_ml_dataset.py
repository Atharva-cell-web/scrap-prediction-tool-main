import sys
import os

# Fix: Add the project root directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import glob
import re
from src.config import RAW_PARAM_DIR, PROCESSED_DIR, HYDRA_FILE
# FIX: Removed 'load_machine_data' because we don't use it anymore
from src.data_loading import load_hydra_data
from src.data_aggregation import aggregate_param_data
from src.data_joining import join_hydra_param

def extract_machine_id(filename):
    """
    Smart function to find 'M-123' or 'M123' inside a long filename.
    """
    match = re.search(r'M[-_\s]?(\d{3})', filename, re.IGNORECASE)
    if match:
        return f"M-{match.group(1)}"
    return "Unknown"

def process_all_machines():
    print(f"🚀 Starting Bulk Processing...")
    print(f"📂 Looking for files in: {RAW_PARAM_DIR}")
    
    machine_files = list(RAW_PARAM_DIR.glob("*.csv"))
    
    if not machine_files:
        print("❌ No files found! Check your 'data/raw_param' folder.")
        return

    print(f"Found {len(machine_files)} files to process.\n")
    
    all_machine_data = []

    for file_path in machine_files:
        filename = file_path.name
        machine_id = extract_machine_id(filename)
        
        print(f"Processing: {filename} -> Detected ID: {machine_id}")
        
        try:
            # Load Param Data directly here
            print(f"   - Loading sensor data (this may take a moment)...")
            df_param = pd.read_csv(file_path, parse_dates=['timestamp'], low_memory=False)
            print(f"   - Loaded {len(df_param)} rows.")
            
            # Aggregate
            df_agg = aggregate_param_data(df_param)
            print(f"   - Aggregated into {len(df_agg)} daily summaries.")
            
            # Load Hydra (Target)
            df_hydra = load_hydra_data(str(HYDRA_FILE), machine_id)
            
            # Join
            df_joined = join_hydra_param(df_hydra, df_agg)
            
            if not df_joined.empty:
                df_joined['machine_id'] = machine_id
                all_machine_data.append(df_joined)
                print(f"   ✅ Success! Added {len(df_joined)} rows.")
            else:
                print(f"   ⚠️ No matching dates found for {machine_id}.")

        except Exception as e:
            print(f"   ❌ Error processing {machine_id}: {e}")
            continue
        
        print("-" * 40)

    if all_machine_data:
        final_df = pd.concat(all_machine_data, ignore_index=True)
        output_path = PROCESSED_DIR / "final_ml_dataset.parquet"
        final_df.to_parquet(output_path)
        print(f"\n🎉 DONE! Final dataset saved with {len(final_df)} rows.")
        print(f"📍 Location: {output_path}")
    else:
        print("\n⚠️ Process finished, but no data was generated.")

if __name__ == "__main__":
    process_all_machines()