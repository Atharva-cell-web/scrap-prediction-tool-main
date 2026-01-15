import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os

# Use the existing function from the project to load the data
from src.data_joining import create_ml_dataset_all_machines

warnings.filterwarnings('ignore')

# --- STEP 1: LOAD DATA ---
print("Loading final ML dataset...")
df, _ = create_ml_dataset_all_machines()

# --- STEP 2: TARGET ENGINEERING ---
# Define High Risk as top 25% of scrap quantity
threshold = df['actual_scrap_qty'].quantile(0.75)
df['high_scrap_risk'] = (df['actual_scrap_qty'] > threshold).astype(int)

print(f"Scrap Risk Threshold (75th percentile): {threshold}")
print(f"Class Balance:\n{df['high_scrap_risk'].value_counts()}")

# --- STEP 3: PREPARE DATA FOR TRAINING ---
print("Preparing data for training...")

# 1. Define the Target (y)
y = df['high_scrap_risk']

# 2. Define Features (X)
# NUCLEAR OPTION: Drop anything that smells like a target or a future value
# ALLOW 'pressure' back in, but keep the leaks (production, qty, scrap) out!
forbidden_words = ['scrap', 'risk', 'qty', 'duration', 'runtime', 'counter', 'production']

# Identify columns to drop
cols_to_drop = [c for c in df.columns if any(word in c.lower() for word in forbidden_words)]
print(f"Dropping {len(cols_to_drop)} suspicious columns: {cols_to_drop}")

# Drop them
X_raw = df.drop(columns=cols_to_drop)

# 3. Select ONLY numeric columns
X = X_raw.select_dtypes(include=['number'])

# 4. ONE FINAL CHECK: Drop 'date' if it slipped through (it's not numeric usually, but safe to check)
if 'date' in X.columns:
    X = X.drop(columns=['date'])

print(f"Training with {X.shape[1]} clean sensor features.")

# --- STEP 4: TRAIN MODEL ---
print("Splitting data and training model...")

# Use GroupShuffleSplit to prevent data leakage (split by DATE)
split_key = df['date'] if 'date' in df.columns else df.index

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=split_key))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# CRITICAL: Added class_weight='balanced' to handle the 0.00 recall issue
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- STEP 5: EVALUATION ---
print("\nModel Training Complete! Evaluating...")
y_pred = model.predict(X_test)

# Print the Recall specifically
report = classification_report(y_test, y_pred, output_dict=True)
print("-" * 30)
print(f"RECALL FOR HIGH RISK (Class 1): {report['1']['recall']:.2f}")
print("-" * 30)

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- STEP 6: SAVE FEATURE IMPORTANCE PLOT ---
os.makedirs('outputs', exist_ok=True)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'].head(10)[::-1], feature_importance['importance'].head(10)[::-1])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300)
print("\nFeature importance plot saved to outputs/feature_importance.png")