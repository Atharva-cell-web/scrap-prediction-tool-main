"""
Baseline Model Training Script for Predictive Scrap AI

What this script does:
- Loads the joined ML dataset (HydraData + aggregated ParamData)
- Splits data into train/test sets using GROUP-BASED split to prevent leakage
- Trains a RandomForestRegressor baseline model
- Evaluates performance using MAE and RMSE
- Compares against a naive baseline (predict mean scrap)
- Reports top 15 most important features

Why RandomForest as baseline:
- Works well out-of-the-box without hyperparameter tuning
- Handles non-linear relationships naturally
- Robust to outliers and missing values
- Provides feature importance for interpretability
- No need for feature scaling or normalization

CRITICAL: Data Leakage Prevention
- ParamData is aggregated at machine+date level
- HydraData has many rows per date (orders/shifts)
- All rows on same date share IDENTICAL sensor features
- Random split would put duplicates in train AND test = leakage!
- Solution: Use GroupShuffleSplit with (machine_id, date_key) as groups
- This ensures all rows from a date go to EITHER train OR test, not both

Modeling Decisions Explained:
- Group-based split: Prevents temporal/duplicate leakage
- 80/20 split by groups (not rows): Some variance in actual row counts
- Default hyperparameters: Establishes baseline before tuning
- Exclude leaky columns: actual_qty, target_qty, shift_qty, etc.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_joining import create_ml_dataset_all_machines


# =============================================================================
# Configuration
# =============================================================================

# Target variable to predict
TARGET_COLUMN = "actual_scrap_qty"

# Random seed for reproducibility
# Why 42: Commonly used seed, makes results reproducible across runs
RANDOM_STATE = 42

# Train/test split ratio
# Why 80/20: Standard split that provides enough training data while
# maintaining a reasonable test set size for reliable evaluation
TEST_SIZE = 0.20

# Columns to exclude from features
# These are identifiers, keys, or target-related columns that shouldn't be used as features
EXCLUDE_COLUMNS = [
    # Target and target-related (LEAKAGE - directly correlated with scrap)
    "actual_scrap_qty",      # Target variable - cannot use to predict itself
    "shift_scrap",           # Directly related to target (potential leakage)
    "actual_qty",            # LEAKAGE: actual production qty is outcome data
    "target_qty",            # LEAKAGE: highly correlated with scrap amount
    "shift_qty",             # LEAKAGE: shift-level quantity is outcome data
    "Scrap_counter_first",   # LEAKAGE: scrap counter directly measures scrap
    "Scrap_counter_last",    # LEAKAGE: scrap counter directly measures scrap
    "Scrap_counter_delta",   # LEAKAGE: scrap counter directly measures scrap
    "Scrap_counter_mean",    # LEAKAGE: scrap counter directly measures scrap
    "Scrap_counter_max",     # LEAKAGE: scrap counter directly measures scrap
    "Scrap_counter_std",     # LEAKAGE: scrap counter directly measures scrap
    "Scrap_counter_count",   # LEAKAGE: scrap counter directly measures scrap
    
    # Post-hoc features (known only AFTER production completes)
    "rest_duration",         # LEAKAGE: rest time is outcome data
    "actual_cycle_time_per_1000",  # LEAKAGE: actual cycle time is outcome
    "shift_production_time", # LEAKAGE: production time is outcome data
    "remaining_runtime",     # LEAKAGE: known after production
    
    # Identifiers and keys
    "machine_id",            # Categorical identifier (would need encoding)
    "machine_key",           # Join key
    "date_key",              # Join key (temporal info)
    "agg_date",              # Aggregation date
    "machine_def",           # ParamData machine identifier
    
    # HydraData columns that are identifiers or non-predictive
    "datetime_last_load",
    "local_datetime_last_load",
    "machine_nr",
    "machine_group_nr",
    "order_operation",
    "article",
    "article_description",
    "tool_number",
    "order_planned_start",
    "order_start",
    "order_planned_stop",
    "order_stop",
    "actual_order_start",
    "shift_start_date",
    "shift_stop_date",
    "stoer_text",
    "parts_produced_last_updated",
    "scrap_reason_codes",
    "local_time_zone",
    "date",
    "year",
    "month",
    
    # Categorical columns that need special handling (skip for baseline)
    "Machine_status_mode",   # Categorical - would need encoding
    "actual_machine_status", # Categorical status
    "regrind_flag",          # Binary flag
]


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Identify numeric feature columns suitable for modeling.
    
    Why this approach:
    - Automatically detects numeric columns
    - Excludes known non-feature columns
    - More maintainable than hardcoding feature list
    
    Args:
        df: DataFrame with all columns
        
    Returns:
        List of feature column names
    """
    # Start with all columns
    all_cols = set(df.columns)
    
    # Remove excluded columns
    feature_cols = all_cols - set(EXCLUDE_COLUMNS)
    
    # Keep only numeric columns
    # Why: RandomForest requires numeric input; categorical needs encoding
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = feature_cols.intersection(set(numeric_cols))
    
    return sorted(list(feature_cols))


def compute_naive_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute naive baseline metrics using mean prediction.
    
    Why mean baseline:
    - Simplest possible predictor
    - Any useful model must beat this
    - Provides context for interpreting model metrics
    - Mean minimizes squared error, so it's the optimal "no-information" predictor
    
    Args:
        y_train: Training target values
        y_test: Test target values
        
    Returns:
        Dictionary with baseline metrics
    """
    # Predict the training set mean for all test samples
    # Why training mean: Simulates real-world where we don't know test values
    mean_prediction = y_train.mean()
    
    # Create array of predictions
    y_pred_baseline = np.full_like(y_test, fill_value=mean_prediction, dtype=float)
    
    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred_baseline)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    
    return {
        "mean_prediction": mean_prediction,
        "mae": mae,
        "rmse": rmse,
    }


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor with default hyperparameters.
    
    Why default hyperparameters:
    - Establishes baseline performance
    - RandomForest is relatively robust with defaults
    - Hyperparameter tuning comes later once baseline is established
    
    Default settings (sklearn defaults):
    - n_estimators=100: Reasonable number of trees for baseline
    - max_depth=None: Trees grow until leaves are pure or min_samples
    - min_samples_split=2: Standard splitting criterion
    - min_samples_leaf=1: Allows for detailed leaf nodes
    - n_jobs=-1: Use all available CPU cores for parallelism
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        
    Returns:
        Trained RandomForestRegressor model
    """
    model = RandomForestRegressor(
        n_estimators=100,     # 100 trees (sklearn default)
        max_depth=None,       # No maximum depth (trees grow fully)
        min_samples_split=2,  # Minimum samples to split a node
        min_samples_leaf=1,   # Minimum samples in leaf node
        n_jobs=-1,            # Use all CPU cores for faster training
        random_state=random_state,  # Reproducibility
        verbose=0,            # Suppress training output
    )
    
    print("Training RandomForestRegressor...")
    print(f"  - n_estimators: 100")
    print(f"  - max_depth: None (unlimited)")
    print(f"  - n_jobs: -1 (all cores)")
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate model performance on test set.
    
    Metrics explained:
    - MAE (Mean Absolute Error): Average absolute difference between predictions
      and actual values. Intuitive interpretation in original units.
    - RMSE (Root Mean Squared Error): Square root of average squared errors.
      Penalizes large errors more heavily than MAE.
    
    Why both metrics:
    - MAE: Easy to interpret ("on average, predictions are off by X units")
    - RMSE: More sensitive to outliers, useful if large errors are costly
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Additional context metrics
    y_range = y_test.max() - y_test.min()
    mae_pct = (mae / y_test.mean()) * 100  # MAE as percentage of mean
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mae_pct_of_mean": mae_pct,
        "y_test_mean": y_test.mean(),
        "y_test_std": y_test.std(),
        "y_test_range": y_range,
    }


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract and rank feature importances from trained model.
    
    What feature importance means in RandomForest:
    - Based on mean decrease in impurity (Gini importance)
    - Higher value = feature contributes more to reducing prediction error
    - Computed as average reduction in variance across all trees
    
    Caveats:
    - Can be biased toward high-cardinality features
    - Correlated features may share importance
    - Should be validated with permutation importance later
    
    Args:
        model: Trained RandomForestRegressor
        feature_names: List of feature column names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importances ranked by importance
    """
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })
    
    # Sort by importance descending
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    # Add cumulative importance
    importance_df["cumulative_importance"] = importance_df["importance"].cumsum()
    
    # Add rank
    importance_df["rank"] = range(1, len(importance_df) + 1)
    
    return importance_df.head(top_n).reset_index(drop=True)


def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("BASELINE MODEL TRAINING: RandomForestRegressor")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Load ML Dataset
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: LOADING ML DATASET")
    print("=" * 70)
    
    # Load joined dataset (only matched rows with features)
    ml_df, reports = create_ml_dataset_all_machines(keep_unmatched=False)
    
    print(f"\nDataset loaded:")
    print(f"  - Total samples: {len(ml_df):,}")
    print(f"  - Total columns: {len(ml_df.columns)}")
    
    # =========================================================================
    # Step 2: Prepare Features and Target
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: PREPARING FEATURES AND TARGET")
    print("=" * 70)
    
    # Get feature columns
    feature_cols = get_feature_columns(ml_df)
    print(f"\nFeature columns identified: {len(feature_cols)}")
    
    # Prepare X and y
    X = ml_df[feature_cols].copy()
    y = ml_df[TARGET_COLUMN].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target statistics:")
    print(f"  - Mean: {y.mean():,.2f}")
    print(f"  - Std: {y.std():,.2f}")
    print(f"  - Range: [{y.min():,.0f}, {y.max():,.0f}]")
    
    # Check for missing values in features
    missing_pct = (X.isna().sum().sum() / X.size) * 100
    print(f"\nMissing values in features: {missing_pct:.2f}%")
    
    # Handle any remaining missing values by filling with column median
    # Why median: More robust to outliers than mean
    if missing_pct > 0:
        print("Filling missing values with column median...")
        X = X.fillna(X.median())
    
    # =========================================================================
    # Step 3: Train/Test Split (GROUP-BASED to prevent leakage)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN/TEST SPLIT (Group-Based)")
    print("=" * 70)
    
    # Create group identifier from machine_id + date_key
    # Why: All rows with same (machine, date) share identical features
    # Random split would leak these duplicates between train/test
    groups = ml_df["machine_id"].astype(str) + "_" + ml_df["date_key"].astype(str)
    
    print(f"\nData leakage prevention:")
    print(f"  - Unique groups (machine+date): {groups.nunique()}")
    print(f"  - Total rows: {len(groups):,}")
    print(f"  - Avg rows per group: {len(groups) / groups.nunique():.0f}")
    
    # Use GroupShuffleSplit to ensure no group appears in both train and test
    # Why GroupShuffleSplit: Randomly selects groups, not rows
    # This prevents data leakage from duplicate features
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Get train/test indices
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    # Verify no group overlap
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    
    print(f"\nSplit configuration:")
    print(f"  - Test size: {TEST_SIZE:.0%} of groups")
    print(f"  - Random state: {RANDOM_STATE}")
    print(f"\nResulting sizes:")
    print(f"  - Training groups: {len(train_groups)}")
    print(f"  - Test groups: {len(test_groups)}")
    print(f"  - Group overlap: {len(overlap)} (should be 0)")
    print(f"  - Training samples: {len(X_train):,} ({len(X_train)/len(X):.1%})")
    print(f"  - Test samples: {len(X_test):,} ({len(X_test)/len(X):.1%})")
    print(f"\nTarget distribution:")
    print(f"  - Train mean: {y_train.mean():,.2f}, std: {y_train.std():,.2f}")
    print(f"  - Test mean: {y_test.mean():,.2f}, std: {y_test.std():,.2f}")
    
    # =========================================================================
    # Step 4: Compute Naive Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: NAIVE BASELINE (Predict Mean)")
    print("=" * 70)
    
    baseline_metrics = compute_naive_baseline(y_train, y_test)
    
    print(f"\nNaive baseline strategy: Predict training mean for all samples")
    print(f"  - Predicted value: {baseline_metrics['mean_prediction']:,.2f}")
    print(f"\nBaseline metrics on test set:")
    print(f"  - MAE: {baseline_metrics['mae']:,.2f}")
    print(f"  - RMSE: {baseline_metrics['rmse']:,.2f}")
    
    # =========================================================================
    # Step 5: Train RandomForest Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: TRAINING RANDOMFOREST MODEL")
    print("=" * 70)
    
    model = train_random_forest(X_train, y_train, random_state=RANDOM_STATE)
    
    print("\nTraining complete!")
    
    # =========================================================================
    # Step 6: Evaluate Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 70)
    
    model_metrics = evaluate_model(model, X_test, y_test)
    
    print(f"\nRandomForest metrics on test set:")
    print(f"  - MAE: {model_metrics['mae']:,.2f}")
    print(f"  - RMSE: {model_metrics['rmse']:,.2f}")
    print(f"  - MAE as % of mean: {model_metrics['mae_pct_of_mean']:.1f}%")
    
    # =========================================================================
    # Step 7: Compare to Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: MODEL vs BASELINE COMPARISON")
    print("=" * 70)
    
    mae_improvement = (baseline_metrics['mae'] - model_metrics['mae']) / baseline_metrics['mae'] * 100
    rmse_improvement = (baseline_metrics['rmse'] - model_metrics['rmse']) / baseline_metrics['rmse'] * 100
    
    print(f"\n{'Metric':<20} {'Naive Baseline':>15} {'RandomForest':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'MAE':<20} {baseline_metrics['mae']:>15,.2f} {model_metrics['mae']:>15,.2f} {mae_improvement:>14.1f}%")
    print(f"{'RMSE':<20} {baseline_metrics['rmse']:>15,.2f} {model_metrics['rmse']:>15,.2f} {rmse_improvement:>14.1f}%")
    
    if mae_improvement > 0:
        print(f"\n✓ Model beats naive baseline by {mae_improvement:.1f}% (MAE)")
    else:
        print(f"\n✗ Model does NOT beat naive baseline - need to investigate")
    
    # =========================================================================
    # Step 8: Feature Importance
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: FEATURE IMPORTANCE (Top 15)")
    print("=" * 70)
    
    importance_df = get_feature_importance(model, feature_cols, top_n=15)
    
    print(f"\n{'Rank':<6} {'Feature':<40} {'Importance':>12} {'Cumulative':>12}")
    print("-" * 70)
    for _, row in importance_df.iterrows():
        print(f"{int(row['rank']):<6} {row['feature']:<40} {row['importance']:>12.4f} {row['cumulative_importance']:>11.1%}")
    
    # Summary of top features
    print(f"\nTop 5 features account for {importance_df.head(5)['importance'].sum():.1%} of total importance")
    print(f"Top 15 features account for {importance_df['importance'].sum():.1%} of total importance")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    print(f"""
Model: RandomForestRegressor (default hyperparameters)
Dataset: {len(ml_df):,} samples, {len(feature_cols)} features
Split: {1-TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test (random)

Performance on Test Set:
  ┌─────────────────────────────────────────┐
  │  MAE:  {model_metrics['mae']:>10,.2f}  (baseline: {baseline_metrics['mae']:,.2f})
  │  RMSE: {model_metrics['rmse']:>10,.2f}  (baseline: {baseline_metrics['rmse']:,.2f})
  │  Improvement: {mae_improvement:>6.1f}% vs naive baseline
  └─────────────────────────────────────────┘

Top 3 Features:
  1. {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.4f})
  2. {importance_df.iloc[1]['feature']} ({importance_df.iloc[1]['importance']:.4f})
  3. {importance_df.iloc[2]['feature']} ({importance_df.iloc[2]['importance']:.4f})

Next Steps:
  - Hyperparameter tuning (n_estimators, max_depth, etc.)
  - Time-based train/test split for temporal validation
  - Cross-validation for more robust performance estimate
  - Feature engineering based on domain knowledge
  - Try other models (XGBoost, LightGBM, Neural Networks)
""")
    
    return model, model_metrics, baseline_metrics, importance_df


if __name__ == "__main__":
    model, model_metrics, baseline_metrics, importance_df = main()
