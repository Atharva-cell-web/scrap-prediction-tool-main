"""
Data Aggregation Module for Predictive Scrap AI

What this module does:
- Aggregates long-format ParamData (time-series) to machine + date level
- Transforms data from long format to wide format for ML modeling
- Handles special variable types (time durations, categorical status codes)
- Computes statistical aggregations (mean, max, std, count) per variable
- Drops zero-variance and uninformative variables

Why it is needed:
- ParamData is in long format with one row per sensor reading
- HydraData is at shift level with production/scrap outcomes
- To build predictive models, we need to align time-series features with outcomes
- Aggregation reduces dimensionality while preserving signal

Assumptions:
- ParamData timestamps can be aligned to HydraData dates
- Shift boundaries are defined by shift_start_date and shift_stop_date in HydraData
- For initial implementation, we aggregate by machine + date (can refine to shift later)
- Zero-variance variables provide no predictive value and should be dropped

Aggregation Decisions:
- Numeric variables: mean (central tendency), max (peak values), std (variability), count (data availability)
- Time_on_machine: Convert HH:MM:SS to seconds, then aggregate numerically
- Machine_status: Take mode (most frequent status) per aggregation period
- Zero-variance variables are identified and excluded automatically
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from .config import (
    DATA_DIR,
    MachineConfig,
    ParamDataSchema,
    PARAM_SCHEMA,
    get_all_machine_configs,
)
from .data_loading import load_csv_robust, standardize_column_names

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Constants: Variables to Exclude
# =============================================================================

# Zero-variance variables identified during profiling
# These provide no predictive value as they never change
ZERO_VARIANCE_VARIABLES = [
    "Cyl_tmp_z2",      # Always 0 - unused heating zone
    "Cyl_tmp_z6",      # Always 0 - unused heating zone
    "Cyl_tmp_z7",      # Always 0 - unused heating zone
    "Switch_position", # Constant at 8.0 - no variation
]

# Variables requiring special handling (not standard numeric aggregation)
SPECIAL_VARIABLES = {
    "Time_on_machine": "time_duration",    # HH:MM:SS format -> seconds
    "Machine_status": "categorical",        # Status codes -> mode
}

# Variables that are cumulative counters (require different treatment)
# For counters, we might want first, last, or delta instead of mean
COUNTER_VARIABLES = [
    "Shot_counter",
    "Scrap_counter",
]


# =============================================================================
# Time Duration Conversion
# =============================================================================

def time_string_to_seconds(time_str: str) -> Optional[float]:
    """
    Convert HH:MM:SS time string to total seconds.
    
    What it does:
    - Parses time strings in HH:MM:SS format
    - Returns total seconds as float
    - Handles malformed strings gracefully
    
    Why needed:
    - Time_on_machine is stored as string, but we need numeric for aggregation
    - Seconds is a natural unit for time-based calculations
    
    Args:
        time_str: Time string in HH:MM:SS format
        
    Returns:
        Total seconds as float, or None if parsing fails
        
    Examples:
        >>> time_string_to_seconds("02:43:19")
        9799.0
        >>> time_string_to_seconds("00:01:30")
        90.0
    """
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
    
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:
            return None
    except (ValueError, AttributeError):
        return None


def convert_time_column(df: pd.DataFrame, column: str = "value") -> pd.Series:
    """
    Convert a column of time strings to seconds.
    
    Args:
        df: DataFrame containing time strings
        column: Column name with time values
        
    Returns:
        Series with time values in seconds
    """
    return df[column].apply(time_string_to_seconds)


# =============================================================================
# Variable Classification
# =============================================================================

def classify_variables(df: pd.DataFrame, variable_column: str = "variable_name") -> Dict[str, List[str]]:
    """
    Classify variables into categories for appropriate handling.
    
    What it does:
    - Identifies numeric vs special variables
    - Separates zero-variance variables
    - Groups counter variables separately
    
    Args:
        df: ParamData DataFrame
        variable_column: Column containing variable names
        
    Returns:
        Dictionary with variable categories:
        - 'numeric': Standard numeric variables for aggregation
        - 'time_duration': Time string variables
        - 'categorical': Categorical variables
        - 'counters': Cumulative counter variables
        - 'excluded': Zero-variance and excluded variables
    """
    all_variables = df[variable_column].unique().tolist()
    
    classified = {
        "numeric": [],
        "time_duration": [],
        "categorical": [],
        "counters": [],
        "excluded": [],
    }
    
    for var in all_variables:
        # Check if excluded (zero-variance)
        if var in ZERO_VARIANCE_VARIABLES:
            classified["excluded"].append(var)
        # Check if special handling needed
        elif var in SPECIAL_VARIABLES:
            var_type = SPECIAL_VARIABLES[var]
            classified[var_type].append(var)
        # Check if counter
        elif var in COUNTER_VARIABLES:
            classified["counters"].append(var)
        # Default to numeric
        else:
            classified["numeric"].append(var)
    
    return classified


# =============================================================================
# Data Preprocessing
# =============================================================================

def preprocess_param_data(
    df: pd.DataFrame,
    schema: ParamDataSchema = PARAM_SCHEMA,
    drop_zero_variance: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Preprocess ParamData for aggregation.
    
    What it does:
    - Parses timestamps
    - Extracts date for aggregation grouping
    - Converts Time_on_machine to seconds
    - Converts numeric values to float
    - Optionally drops zero-variance variables
    
    Why needed:
    - Raw data has string timestamps and mixed value types
    - Need clean numeric values for aggregation
    - Need date column for grouping
    
    Args:
        df: Raw ParamData DataFrame
        schema: ParamData schema definition
        drop_zero_variance: Whether to exclude zero-variance variables
        
    Returns:
        Tuple of (preprocessed DataFrame, variable classification dict)
    """
    df = df.copy()
    
    # Parse timestamp
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Extract date for aggregation (will be used to match with HydraData)
    df["agg_date"] = df["timestamp_parsed"].dt.date.astype(str)
    
    # Extract hour for potential shift assignment
    df["hour"] = df["timestamp_parsed"].dt.hour
    
    # Classify variables
    var_classes = classify_variables(df, schema.variable_column)
    
    # Drop zero-variance variables if requested
    if drop_zero_variance:
        excluded = var_classes["excluded"]
        if excluded:
            logger.info(f"Dropping {len(excluded)} zero-variance variables: {excluded}")
            df = df[~df[schema.variable_column].isin(excluded)]
    
    # Convert Time_on_machine values to seconds
    time_vars = var_classes["time_duration"]
    if time_vars:
        mask = df[schema.variable_column].isin(time_vars)
        df.loc[mask, "value_seconds"] = convert_time_column(df.loc[mask], "value")
        logger.info(f"Converted {mask.sum()} time values to seconds")
    
    # Convert numeric values to float
    # First, handle non-time variables
    non_time_mask = ~df[schema.variable_column].isin(time_vars + var_classes["categorical"])
    df.loc[non_time_mask, "value_numeric"] = pd.to_numeric(
        df.loc[non_time_mask, "value"], errors="coerce"
    )
    
    # For time variables, use the converted seconds
    if time_vars:
        time_mask = df[schema.variable_column].isin(time_vars)
        df.loc[time_mask, "value_numeric"] = df.loc[time_mask, "value_seconds"]
    
    return df, var_classes


# =============================================================================
# Aggregation Functions
# =============================================================================

def aggregate_numeric_variables(
    df: pd.DataFrame,
    group_cols: List[str],
    variable_column: str = "variable_name",
    value_column: str = "value_numeric",
    agg_funcs: List[str] = ["mean", "max", "std", "count"],
) -> pd.DataFrame:
    """
    Aggregate numeric variables with statistical functions.
    
    What it does:
    - Groups data by specified columns (machine + date)
    - Computes mean, max, std, count for each variable
    - Pivots from long to wide format
    
    Why these aggregations:
    - mean: Central tendency of sensor readings
    - max: Peak values which may indicate stress/anomalies
    - std: Variability/stability of the process
    - count: Data availability (useful for detecting gaps)
    
    Args:
        df: Preprocessed DataFrame with numeric values
        group_cols: Columns to group by (e.g., ['machine_def', 'agg_date'])
        variable_column: Column with variable names
        value_column: Column with numeric values
        agg_funcs: Aggregation functions to apply
        
    Returns:
        Wide-format DataFrame with one row per group
    """
    # Filter to rows with numeric values
    numeric_df = df[df[value_column].notna()].copy()
    
    if numeric_df.empty:
        logger.warning("No numeric values to aggregate")
        return pd.DataFrame()
    
    # Group and aggregate
    agg_result = (
        numeric_df
        .groupby(group_cols + [variable_column])[value_column]
        .agg(agg_funcs)
        .reset_index()
    )
    
    # Flatten column names after aggregation
    # Result has multi-level columns if agg returns multiple stats
    if isinstance(agg_result.columns, pd.MultiIndex):
        agg_result.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in agg_result.columns
        ]
    
    # Pivot to wide format
    # Each variable becomes multiple columns (var_mean, var_max, var_std, var_count)
    pivot_dfs = []
    
    for func in agg_funcs:
        pivot = agg_result.pivot_table(
            index=group_cols,
            columns=variable_column,
            values=value_column if len(agg_funcs) == 1 else func,
            aggfunc="first"  # Should be unique after groupby
        )
        # Rename columns to include aggregation function
        pivot.columns = [f"{col}_{func}" for col in pivot.columns]
        pivot_dfs.append(pivot)
    
    # Combine all pivoted aggregations
    result = pd.concat(pivot_dfs, axis=1).reset_index()
    
    return result


def aggregate_categorical_variables(
    df: pd.DataFrame,
    group_cols: List[str],
    variable_column: str = "variable_name",
    value_column: str = "value",
    categorical_vars: List[str] = None,
) -> pd.DataFrame:
    """
    Aggregate categorical variables by taking mode (most frequent value).
    
    What it does:
    - For each categorical variable, finds the most common value per group
    - Returns wide-format with one column per categorical variable
    
    Why mode:
    - Machine_status codes indicate operational state
    - Most frequent status represents dominant behavior in the period
    
    Args:
        df: DataFrame with categorical values
        group_cols: Columns to group by
        variable_column: Column with variable names
        value_column: Column with categorical values
        categorical_vars: List of categorical variable names
        
    Returns:
        Wide-format DataFrame with mode of each categorical variable
    """
    if not categorical_vars:
        return pd.DataFrame()
    
    cat_df = df[df[variable_column].isin(categorical_vars)].copy()
    
    if cat_df.empty:
        return pd.DataFrame()
    
    # Custom mode function that handles ties by taking first
    def safe_mode(x):
        modes = x.mode()
        return modes.iloc[0] if len(modes) > 0 else None
    
    # Group and compute mode
    mode_result = (
        cat_df
        .groupby(group_cols + [variable_column])[value_column]
        .agg(safe_mode)
        .reset_index()
    )
    
    # Pivot to wide format
    pivot = mode_result.pivot_table(
        index=group_cols,
        columns=variable_column,
        values=value_column,
        aggfunc="first"
    )
    
    # Rename columns to indicate these are mode values
    pivot.columns = [f"{col}_mode" for col in pivot.columns]
    
    return pivot.reset_index()


def aggregate_counter_variables(
    df: pd.DataFrame,
    group_cols: List[str],
    variable_column: str = "variable_name",
    value_column: str = "value_numeric",
    counter_vars: List[str] = None,
) -> pd.DataFrame:
    """
    Aggregate counter variables with appropriate functions.
    
    What it does:
    - For counters, computes first, last, and delta (last - first)
    - Delta represents the change during the aggregation period
    
    Why delta for counters:
    - Shot_counter and Scrap_counter are cumulative
    - The delta tells us how many shots/scraps occurred in the period
    - This is more meaningful than mean for cumulative counters
    
    Args:
        df: DataFrame with counter values
        group_cols: Columns to group by
        variable_column: Column with variable names
        value_column: Column with numeric values
        counter_vars: List of counter variable names
        
    Returns:
        Wide-format DataFrame with counter aggregations
    """
    if not counter_vars:
        return pd.DataFrame()
    
    counter_df = df[
        (df[variable_column].isin(counter_vars)) & 
        (df[value_column].notna())
    ].copy()
    
    if counter_df.empty:
        return pd.DataFrame()
    
    # Sort by timestamp to ensure first/last are time-ordered
    counter_df = counter_df.sort_values("timestamp_parsed")
    
    # Aggregate with first, last, and compute delta
    agg_result = (
        counter_df
        .groupby(group_cols + [variable_column])[value_column]
        .agg(["first", "last", "count"])
        .reset_index()
    )
    
    # Compute delta
    agg_result["delta"] = agg_result["last"] - agg_result["first"]
    
    # Pivot each statistic
    result_dfs = []
    
    for stat in ["first", "last", "delta", "count"]:
        pivot = agg_result.pivot_table(
            index=group_cols,
            columns=variable_column,
            values=stat,
            aggfunc="first"
        )
        pivot.columns = [f"{col}_{stat}" for col in pivot.columns]
        result_dfs.append(pivot)
    
    result = pd.concat(result_dfs, axis=1).reset_index()
    
    return result


# =============================================================================
# Main Aggregation Function
# =============================================================================

def aggregate_param_data(
    df: pd.DataFrame,
    machine_column: str = "machine_def",
    group_by_date: bool = True,
    schema: ParamDataSchema = PARAM_SCHEMA,
    drop_zero_variance: bool = True,
    numeric_agg_funcs: List[str] = ["mean", "max", "std", "count"],
) -> pd.DataFrame:
    """
    Aggregate ParamData from long format to wide format at machine + date level.
    
    What it does:
    1. Preprocesses data (parse timestamps, convert types)
    2. Drops zero-variance variables
    3. Aggregates numeric variables (mean, max, std, count)
    4. Aggregates categorical variables (mode)
    5. Aggregates counter variables (first, last, delta)
    6. Combines all into a single wide-format DataFrame
    
    Why this approach:
    - Separates concerns for different variable types
    - Preserves meaningful information from each type
    - Creates a clean feature matrix ready for ML
    
    Args:
        df: Raw ParamData DataFrame (long format)
        machine_column: Column identifying the machine
        group_by_date: Whether to group by date (vs. other periods)
        schema: ParamData schema definition
        drop_zero_variance: Whether to exclude zero-variance variables
        numeric_agg_funcs: Aggregation functions for numeric variables
        
    Returns:
        Wide-format DataFrame with aggregated features
    """
    logger.info(f"Starting ParamData aggregation: {len(df):,} rows")
    
    # Step 1: Preprocess
    df_processed, var_classes = preprocess_param_data(
        df, schema, drop_zero_variance
    )
    
    logger.info(f"Variable classification:")
    for cat, vars in var_classes.items():
        logger.info(f"  {cat}: {len(vars)} variables")
    
    # Define grouping columns
    group_cols = [machine_column]
    if group_by_date:
        group_cols.append("agg_date")
    
    # Step 2: Aggregate numeric variables (including converted time)
    numeric_vars = var_classes["numeric"] + var_classes["time_duration"]
    numeric_df = df_processed[df_processed[schema.variable_column].isin(numeric_vars)]
    
    logger.info(f"Aggregating {len(numeric_vars)} numeric variables...")
    numeric_agg = aggregate_numeric_variables(
        numeric_df,
        group_cols,
        schema.variable_column,
        "value_numeric",
        numeric_agg_funcs,
    )
    logger.info(f"  Created {len(numeric_agg.columns) - len(group_cols)} numeric features")
    
    # Step 3: Aggregate categorical variables
    logger.info(f"Aggregating {len(var_classes['categorical'])} categorical variables...")
    categorical_agg = aggregate_categorical_variables(
        df_processed,
        group_cols,
        schema.variable_column,
        "value",
        var_classes["categorical"],
    )
    if not categorical_agg.empty:
        logger.info(f"  Created {len(categorical_agg.columns) - len(group_cols)} categorical features")
    
    # Step 4: Aggregate counter variables
    logger.info(f"Aggregating {len(var_classes['counters'])} counter variables...")
    counter_agg = aggregate_counter_variables(
        df_processed,
        group_cols,
        schema.variable_column,
        "value_numeric",
        var_classes["counters"],
    )
    if not counter_agg.empty:
        logger.info(f"  Created {len(counter_agg.columns) - len(group_cols)} counter features")
    
    # Step 5: Combine all aggregations
    result = numeric_agg
    
    if not categorical_agg.empty:
        result = result.merge(categorical_agg, on=group_cols, how="outer")
    
    if not counter_agg.empty:
        result = result.merge(counter_agg, on=group_cols, how="outer")
    
    # Step 6: Clean up column names and sort
    result = result.sort_values(group_cols).reset_index(drop=True)
    
    logger.info(f"Aggregation complete: {len(result):,} rows, {len(result.columns)} columns")
    
    return result


# =============================================================================
# Convenience Functions
# =============================================================================

def aggregate_param_data_for_machine(
    machine_config: MachineConfig,
    schema: ParamDataSchema = PARAM_SCHEMA,
    **kwargs,
) -> pd.DataFrame:
    """
    Load and aggregate ParamData for a single machine.
    
    Args:
        machine_config: Machine configuration
        schema: ParamData schema
        **kwargs: Additional arguments for aggregate_param_data
        
    Returns:
        Aggregated DataFrame
    """
    # Load raw data
    df = load_csv_robust(machine_config.param_file)
    df = standardize_column_names(df)
    
    # Add machine_id for identification
    df["machine_id"] = machine_config.machine_id
    
    # Aggregate
    return aggregate_param_data(df, schema=schema, **kwargs)


def aggregate_all_param_data(
    data_dir: Path = DATA_DIR,
    schema: ParamDataSchema = PARAM_SCHEMA,
    **kwargs,
) -> pd.DataFrame:
    """
    Load and aggregate ParamData for all machines.
    
    Args:
        data_dir: Base data directory
        schema: ParamData schema
        **kwargs: Additional arguments for aggregate_param_data
        
    Returns:
        Combined aggregated DataFrame for all machines
    """
    configs = get_all_machine_configs(data_dir)
    
    dfs = []
    for machine_id, config in configs.items():
        logger.info(f"\nProcessing machine: {machine_id}")
        try:
            agg_df = aggregate_param_data_for_machine(config, schema, **kwargs)
            agg_df["machine_id"] = machine_id
            dfs.append(agg_df)
        except Exception as e:
            logger.error(f"Failed to aggregate ParamData for {machine_id}: {e}")
    
    if not dfs:
        raise ValueError("No ParamData could be aggregated")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"\nCombined aggregation: {len(combined):,} rows from {len(dfs)} machines")
    
    return combined


def get_aggregation_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a summary of the aggregated data.
    
    Args:
        df: Aggregated DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "feature_types": {
            "mean_features": len([c for c in df.columns if c.endswith("_mean")]),
            "max_features": len([c for c in df.columns if c.endswith("_max")]),
            "std_features": len([c for c in df.columns if c.endswith("_std")]),
            "count_features": len([c for c in df.columns if c.endswith("_count")]),
            "mode_features": len([c for c in df.columns if c.endswith("_mode")]),
            "delta_features": len([c for c in df.columns if c.endswith("_delta")]),
        },
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
    
    return summary


def print_aggregation_summary(df: pd.DataFrame, name: str = "Aggregated ParamData") -> None:
    """Print a formatted summary of aggregated data."""
    summary = get_aggregation_summary(df)
    
    print(f"\n{'='*60}")
    print(f"Aggregation Summary: {name}")
    print(f"{'='*60}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Total Columns: {summary['columns']}")
    
    print(f"\nFeature Types:")
    for ftype, count in summary["feature_types"].items():
        if count > 0:
            print(f"  {ftype}: {count}")
    
    print(f"\nSample Columns (first 20):")
    for col in summary["column_names"][:20]:
        print(f"  - {col}")
    if len(summary["column_names"]) > 20:
        print(f"  ... and {len(summary['column_names']) - 20} more")


# =============================================================================
# Main Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Data Aggregation Module")
    print("=" * 60)
    
    # Test with single machine
    configs = get_all_machine_configs()
    
    if configs:
        machine_id = list(configs.keys())[0]
        config = configs[machine_id]
        
        print(f"\nAggregating ParamData for {machine_id}...")
        
        agg_df = aggregate_param_data_for_machine(config)
        
        print_aggregation_summary(agg_df, f"ParamData - {machine_id}")
        
        print(f"\nSample of aggregated data:")
        print(agg_df.head().to_string())
        
        # Test with all machines
        print("\n" + "=" * 60)
        print("Aggregating all machines...")
        all_agg = aggregate_all_param_data()
        print_aggregation_summary(all_agg, "All Machines")
    else:
        print("No machines found.")
