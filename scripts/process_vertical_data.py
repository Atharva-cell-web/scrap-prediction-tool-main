import pandas as pd
import glob
import os
import re
import sys

# --- CONFIGURATION (Adjust paths if needed) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PARAM_DIR = os.path.join(BASE_DIR, "data", "raw_param")
RAW_HYDRA_DIR = os.path.join(BASE_DIR, "data", "raw_hydra")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def get_hydra_file():
    files = glob.glob(os.path.join(RAW_HYDRA_DIR, "*.csv"))
    return files[0] if files else None

def clean_machine_id(id_str):
    """
    Converts 'M612-33' or 'M-612' to a standard 'M612' for matching.
    """
    # Find the first sequence of numbers
    match = re.search(r'(\d{3})', str(id_str))
    if match:
        return f"M{match.group(1)}" # Returns M612
    return "Unknown"

def process_vertical_param_data(file_path):
    print(f"   - Reading file...")
    # Read the CSV
    df = pd.read_csv(file_path, low_memory=False)
    
    # Check if this is actually the vertical format
    if 'variable_name' not in df.columns or 'value' not in df.columns:
        print("   ⚠️ This file is not in Vertical format (variable_name/value). Skipping.")
        return pd.DataFrame(), None

    # 1. Clean Values
    # Your screenshot shows some values are times ("4:22:10"). We need Numbers.
    # errors='coerce' turns non-numbers into NaN (which we drop)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])

    # 2. Extract Date
    # Your screenshot shows a 'date' column (e.g., "1/7/2026")
    if 'date' in df.columns:
        df['date_clean'] = pd.to_datetime(df['date'])
    else:
        print("   ⚠️ No 'date' column found.")
        return pd.DataFrame(), None

    # 3. Extract Machine ID (from 'machine_definition' column)
    machine_id_raw = df['machine_definition'].iloc[0] if 'machine_definition' in df.columns else "Unknown"
    machine_id_clean = clean_machine_id(machine_id_raw)
    
    print(f"   - Pivoting data for {machine_id_clean} (This takes a moment)...")

    # 4. PIVOT & AGGREGATE
    # We group by Date and Variable Name, then calculate stats
    df_agg = df.groupby(['date_clean', 'variable_name'])['value'].agg(['mean', 'std', 'min', 'max'])
    
    # Unstack to make variable names into columns
    df_pivot = df_agg.unstack()
    
    # Flatten the complex column names (e.g. ('mean', 'Cushion') -> 'Cushion_mean')
    df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index().rename(columns={'date_clean': 'date'})
    
    return df_pivot, machine_id_clean

def main():
    print(f"🚀 Starting Vertical Data Processing...\n")
    
    # 1. Load Hydra (Target)
    hydra_path = get_hydra_file()
    if not hydra_path:
        print("❌ No Hydra file found!")
        return
        
    print(f"📦 Loading Hydra Target: {os.path.basename(hydra_path)}")
    df_hydra = pd.read_csv(hydra_path)
    
    # Clean Hydra IDs for matching
    if 'machine_nr' in df_hydra.columns:
        df_hydra['id_match'] = df_hydra['machine_nr'].apply(clean_machine_id)
        # Parse Dates
        if 'order_start' in df_hydra.columns:
            df_hydra['date'] = pd.to_datetime(df_hydra['order_start']).dt.date
            df_hydra['date'] = pd.to_datetime(df_hydra['date'])
    else:
        print("❌ Hydra file missing 'machine_nr'")
        return

    # 2. Process Each Machine File
    machine_files = glob.glob(os.path.join(RAW_PARAM_DIR, "*.csv"))
    print(f"📂 Found {len(machine_files)} machine parameter files.\n")
    
    all_data = []

    for file_path in machine_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        try:
            # Process & Pivot
            df_param, machine_id = process_vertical_param_data(file_path)
            
            if df_param.empty: continue
            
            # Filter Hydra for this machine
            df_target = df_hydra[df_hydra['id_match'] == machine_id].copy()
            
            if df_target.empty:
                print(f"   ⚠️ No Hydra targets found for ID: {machine_id}")
                continue

            # Join
            df_merged = pd.merge(df_target, df_param, on='date', how='inner')
            
            if not df_merged.empty:
                df_merged['machine_id'] = machine_id
                all_data.append(df_merged)
                print(f"   ✅ Success! Merged {len(df_merged)} rows for {machine_id}")
            else:
                print(f"   ⚠️ No matching dates found for {machine_id}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
        print("-" * 30)

    # 3. Save Final Dataset
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Save as Parquet (Fast & Small)
        save_path = os.path.join(PROCESSED_DIR, "final_ml_dataset.parquet")
        final_df.to_parquet(save_path)
        
        # Also save a small CSV sample so you can see it works
        csv_path = os.path.join(PROCESSED_DIR, "debug_sample.csv")
        final_df.head(100).to_csv(csv_path, index=False)
        
        print(f"\n🎉 DONE! Saved {len(final_df)} rows to: {save_path}")
        print(f"📊 Saved a sample CSV to: {csv_path}")
    else:
        print("\n⚠️ Finished, but no data was generated.")

if __name__ == "__main__":
    main()