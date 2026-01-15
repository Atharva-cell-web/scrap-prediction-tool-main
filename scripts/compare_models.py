import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_ml_dataset.parquet")

def compare_algorithms():
    print(" Starting.....")
    
    # 1. Load & Clean Data (Same as before)
    if not os.path.exists(DATA_PATH):
        print(" Dataset not found.")
        return

    df = pd.read_parquet(DATA_PATH)
    
    # Create Target
    threshold = df['actual_scrap_qty'].quantile(0.75)
    df['target'] = (df['actual_scrap_qty'] > threshold).astype(int)
    
    # Filter Features
    forbidden = ['target', 'actual_scrap_qty', 'actual_qty', 'machine_id', 'date', 
                 'order_start', 'order_stop', 'shift_production_time', 'shift_scrap']
    cols_to_drop = [c for c in df.columns if c in forbidden or 'actual' in c or 'shift' in c]
    features = df.drop(columns=cols_to_drop)
    
    # Force Numeric
    features = features.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').fillna(0)
    X = features.select_dtypes(include=[np.number])
    y = df['target']
    
    print(f"Data Ready: {len(X)} rows, {X.shape[1]} features.")

    # 2. Define the Contenders
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # 3. The Tournament (Cross-Validation)
    results = {}
    
    # Standard Scaler (Needed for Logistic Regression to work well)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n... Training & Cross-Validation (5-Folds)...")
    print("-" * 60)
    print(f"{'Model Name':<25} | {'Avg Accuracy':<15} | {'Std Dev'}")
    print("-" * 60)

    for name, model in models.items():
       
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        
        results[name] = scores.mean()
        print(f"{name:<25} | {scores.mean():.4f}          | +/- {scores.std():.4f}")

    # Winner
    winner = max(results, key=results.get)
    print("-" * 60)
    print(f"most accurate model: {winner} (Accuracy: {results[winner]:.4f})")
    print("-" * 60)

if __name__ == "__main__":
    compare_algorithms()