import pandas as pd
import os

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")
INPUT_FILE = os.path.join(DATA_DIR, "hydra_data.csv")

def check_scrap_columns():
    print(f"🕵️ DETECTIVE MODE: Finding the REAL Scrap Column...")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ File not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip() # Clean names
    
    print("-" * 60)
    print(f"📊 COMPARING COLUMNS (First 5 Rows):")
    print("-" * 60)
    
    # We will look at 'shift_scrap' vs 'actual_scrap_qty'
    cols_to_check = ['shift_scrap', 'actual_scrap_qty', 'actual_qty']
    
    # Only keep columns that actually exist in the file
    existing_cols = [c for c in cols_to_check if c in df.columns]
    
    print(df[existing_cols].head(10))
    
    print("-" * 60)
    print("📈 STATISTICS (Averages):")
    for col in existing_cols:
        avg_val = df[col].mean()
        max_val = df[col].max()
        print(f"   Column '{col}':")
        print(f"      Average: {avg_val:,.1f}")
        print(f"      Maximum: {max_val:,.0f}")
        if avg_val > 10000:
             print("      ⚠️ VERDICT: Too huge. Likely 'Cumulative Total'.")
        else:
             print("      ✅ VERDICT: Looks like real daily scrap.")
        print("")

if __name__ == "__main__":
    check_scrap_columns()