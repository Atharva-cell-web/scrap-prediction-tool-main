import pandas as pd
import os

def load_hydra_data(file_path, machine_id):
    """
    Loads the Hydra (Target) data and filters for a specific machine.
    """
    if not os.path.exists(file_path):
        print(f"   ❌ Error: Hydra file not found at {file_path}")
        return pd.DataFrame()

    try:
        # Load the CSV
        df = pd.read_csv(file_path)
        
        # Standardize Machine ID for matching
        # Hydra might say 'M231', our code says 'M-231'. We strip dashes to compare.
        target_id_clean = machine_id.replace("-", "").replace(" ", "") # e.g. "M231"
        
        if 'machine_nr' in df.columns:
            # Create a temporary clean column to match against
            df['id_clean'] = df['machine_nr'].astype(str).str.replace("-", "").str.replace(" ", "")
            
            # Filter
            df_filtered = df[df['id_clean'] == target_id_clean].copy()
            df_filtered = df_filtered.drop(columns=['id_clean'])
            
            return df_filtered
        else:
            print(f"   ⚠️ Warning: 'machine_nr' column missing in Hydra file. Returning empty.")
            return pd.DataFrame()

    except Exception as e:
        print(f"   ❌ Error loading Hydra data: {e}")
        return pd.DataFrame()

# Note: load_machine_data is removed because we now load files directly in the main script.