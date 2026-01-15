import os
from pathlib import Path

# --- PROJECT DIRECTORIES ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# New Folder Structure
RAW_PARAM_DIR = DATA_DIR / "raw_param"
RAW_HYDRA_DIR = DATA_DIR / "raw_hydra"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure processed directory exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- AUTOMATIC FILE DETECTION ---
# Find the Hydra file automatically (grabs the first CSV it finds in raw_hydra)
hydra_files = list(RAW_HYDRA_DIR.glob("*.csv"))
if hydra_files:
    HYDRA_FILE = hydra_files[0]
    print(f"✅ Found Hydra File: {HYDRA_FILE.name}")
else:
    HYDRA_FILE = None
    print("⚠️ WARNING: No Hydra file found in data/raw_hydra/")

# We don't list machine files here anymore. The script will scan RAW_PARAM_DIR.