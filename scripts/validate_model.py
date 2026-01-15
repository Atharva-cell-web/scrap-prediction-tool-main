import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_ml_dataset.parquet")

def validate_model():
    print("🚀 Starting Rigorous Cross-Validation (Sanity Check)...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print("❌ Dataset not found.")
        return
    df = pd.read_parquet(DATA_PATH)
    
    # 2. Prepare Data (Same filtering as training)
    # Create Target
    threshold = df['actual_scrap_qty'].quantile(0.75)
    df['target'] = (df['actual_scrap_qty'] > threshold).astype(int)
    
    # Filter Features
    forbidden = ['target', 'actual_scrap_qty', 'actual_qty', 'machine_id', 'date', 
                 'order_start', 'order_stop', 'shift_production_time', 'shift_scrap']
    
    # Drop cheats
    cols_to_drop = [c for c in df.columns if c in forbidden or 'actual' in c or 'shift' in c]
    features = df.drop(columns=cols_to_drop)
    
    # Force Numeric
    features = features.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').fillna(0)
    X = features.select_dtypes(include=[np.number])
    y = df['target']
    
    print(f"📊 Testing on {len(X)} rows and {X.shape[1]} features.")
    
    # 3. Define the Validator (5-Fold)
    # Stratified means it ensures each fold has a fair mix of Good vs Bad parts
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 4. Define the Model (Same settings as before)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    
    # 5. Run Validation
    print("\n🔄 Running 5 distinct tests (this takes a moment)...")
    
    # Calculate scores
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    
    # 6. Report Results
    print("\n" + "="*40)
    print("       🔬 SANITY CHECK RESULTS       ")
    print("="*40)
    
    for i, score in enumerate(acc_scores):
        print(f"  Run #{i+1}: Accuracy = {score:.4f}  (F1: {f1_scores[i]:.4f})")
    
    avg_acc = acc_scores.mean()
    print("-" * 40)
    print(f"🏆 AVERAGE ACCURACY: {avg_acc:.4f} ({(avg_acc*100):.2f}%)")
    print("-" * 40)
    
    if avg_acc > 0.98:
        print("✅ VERDICT: The signal is REAL. It's a 'Broken Leg' physical law.")
    elif avg_acc > 0.85:
        print("⚠️ VERDICT: Model is strong, but the 100% was slightly lucky.")
    else:
        print("❌ VERDICT: The model was overfitting. We need to tune it.")

if __name__ == "__main__":
    validate_model()