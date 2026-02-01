import pandas as pd
import os
import glob

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def check_files():
    print("🕵️ INSPECTING MACHINE DATA...")
    
    # Look for the M231Jan.csv file specifically
    target_path = os.path.join(DATA_DIR, "M231Jan.csv")
    
    if not os.path.exists(target_path):
        print(f"❌ File not found: {target_path}")
        print("   Please move the files into the 'data' folder first!")
        return

    print(f"   👉 Analyzing: M231Jan.csv")
    
    # Read first 5 rows
    df = pd.read_csv(target_path, nrows=5)
        
    print("-" * 50)
    print("✅ COLUMNS FOUND:")
    for col in df.columns:
        print(f"   [ ] {col}")
    print("-" * 50)
    print("👉 Copy and paste this list in the chat!")

if __name__ == "__main__":
    check_files()