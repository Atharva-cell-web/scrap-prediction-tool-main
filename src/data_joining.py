"""
Data Joining Module for Predictive Scrap AI

What this module does:
- Joins aggregated ParamData features with HydraData
- Creates a unified ML-ready dataset with features and target
- Handles key standardization (date formats, machine identifiers)
- Detects and reports join quality issues (unmatched rows, duplicates)

Why it is needed:
- HydraData contains the target variable (actual_scrap_qty) at order+shift level
- Aggregated ParamData contains sensor features at machine+date level
- Need to combine these for predictive modeling
- Join quality directly impacts model validity

Assumptions:
- HydraData dates are in DD-MM-YYYY format
- ParamData dates are in YYYY-MM-DD format (after aggregation)
- Machine identifiers need mapping (e.g., 'M-221' -> 'M221-10')
- Multiple HydraData rows per date is expected (multiple orders/shifts)
- Each HydraData row should map to at most one ParamData row

Join Strategy:
- Join on (machine_id, date) as composite key
- Left join from HydraData to preserve all production records
- ParamData features are replicated for all HydraData rows on same date
- This is valid because sensor aggregations represent daily machine state
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .config import (
    DATA_DIR,
    MachineConfig,
    HYDRA_SCHEMA,
    get_all_machine_configs,
)
from .data_loading import load_hydra_data, standardize_column_names
from .data_aggregation import aggregate_param_data_for_machine

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Join Report Data Structure
# =============================================================================

@dataclass
class JoinReport:
    """Report detailing the results of a data join operation."""
    
    # Input statistics
    hydra_rows_before: int = 0
    param_rows_before: int = 0
    hydra_unique_dates: int = 0
    param_unique_dates: int = 0
    
    # Join statistics
    joined_rows: int = 0
    matched_hydra_rows: int = 0
    unmatched_hydra_rows: int = 0
    overlapping_dates: int = 0
    
    # Quality metrics
    match_rate: float = 0.0
    duplicate_joins: int = 0  # ParamData rows matched multiple times (expected)
    
    # Detailed information
    unmatched_dates: List[str] = field(default_factory=list)
    matched_dates: List[str] = field(default_factory=list)
    
    # Target variable preservation
    target_non_null_before: int = 0
    target_non_null_after: int = 0
    
    def summary(self) -> str:
        """Generate a formatted summary of the join report."""
        lines = [
            "",
            "=" * 70,
            "DATA JOIN REPORT",
            "=" * 70,
            "",
            "INPUT STATISTICS:",
            f"  HydraData rows:        {self.hydra_rows_before:,}",
            f"  ParamData rows:        {self.param_rows_before:,}",
            f"  HydraData unique dates: {self.hydra_unique_dates}",
            f"  ParamData unique dates: {self.param_unique_dates}",
            "",
            "JOIN RESULTS:",
            f"  Output rows:           {self.joined_rows:,}",
            f"  Matched HydraData rows: {self.matched_hydra_rows:,}",
            f"  Unmatched HydraData rows: {self.unmatched_hydra_rows:,}",
            f"  Match rate:            {self.match_rate:.1%}",
            f"  Overlapping dates:     {self.overlapping_dates}",
            "",
            "TARGET VARIABLE (actual_scrap_qty):",
            f"  Non-null before join:  {self.target_non_null_before:,}",
            f"  Non-null after join:   {self.target_non_null_after:,}",
            "",
        ]
        
        if self.unmatched_dates:
            lines.append("UNMATCHED DATES (HydraData without ParamData):")
            for date in sorted(self.unmatched_dates)[:10]:
                lines.append(f"  - {date}")
            if len(self.unmatched_dates) > 10:
                lines.append(f"  ... and {len(self.unmatched_dates) - 10} more")
            lines.append("")
        
        if self.matched_dates:
            lines.append("MATCHED DATES:")
            for date in sorted(self.matched_dates):
                lines.append(f"  - {date}")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# Key Standardization Functions
# =============================================================================

def standardize_date_column(
    df: pd.DataFrame,
    date_column: str,
    input_format: Optional[str] = None,
    output_column: str = "date_key",
) -> pd.DataFrame:
    """
    Standardize date column to YYYY-MM-DD format for joining.
    
    What it does:
    - Parses dates from various formats
    - Converts to standard YYYY-MM-DD string format
    - Handles parsing errors gracefully
    
    Why needed:
    - HydraData uses DD-MM-YYYY format
    - ParamData uses YYYY-MM-DD format
    - Need consistent format for joining
    
    Args:
        df: DataFrame with date column
        date_column: Name of the date column to standardize
        input_format: Optional strptime format string
        output_column: Name for the standardized date column
        
    Returns:
        DataFrame with new standardized date column
    """
    df = df.copy()
    
    if input_format:
        # Parse with specific format
        df[output_column] = pd.to_datetime(
            df[date_column], format=input_format, errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    else:
        # Intelligent auto-detection
        # First, check the format by sampling
        sample = df[date_column].dropna().astype(str).head(10)
        
        # Detect if already in YYYY-MM-DD format
        is_iso_format = sample.str.match(r'^\d{4}-\d{2}-\d{2}').all() if len(sample) > 0 else False
        # Detect if in DD-MM-YYYY format
        is_dmy_format = sample.str.match(r'^\d{2}-\d{2}-\d{4}').all() if len(sample) > 0 else False
        
        if is_iso_format:
            # Already YYYY-MM-DD format
            df[output_column] = pd.to_datetime(
                df[date_column], format="%Y-%m-%d", errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        elif is_dmy_format:
            # DD-MM-YYYY format (European)
            df[output_column] = pd.to_datetime(
                df[date_column], format="%d-%m-%Y", errors="coerce"
            ).dt.strftime("%Y-%m-%d")
        else:
            # Fallback: auto-detect with dayfirst=True for European dates
            df[output_column] = pd.to_datetime(
                df[date_column], dayfirst=True, errors="coerce"
            ).dt.strftime("%Y-%m-%d")
    
    # Log parsing issues
    null_count = df[output_column].isna().sum()
    if null_count > 0:
        logger.warning(f"Failed to parse {null_count} dates in column '{date_column}'")
    
    return df


def create_machine_mapping(
    hydra_machines: List[str],
    param_machines: List[str],
) -> Dict[str, str]:
    """
    Create a mapping between HydraData and ParamData machine identifiers.
    
    What it does:
    - Maps HydraData machine_id (e.g., 'M-221') to ParamData machine_def (e.g., 'M221-10')
    - Uses pattern matching to find corresponding machines
    
    Why needed:
    - Machine identifiers differ between data sources
    - HydraData: 'M-221' format
    - ParamData: 'M221-10' format (includes line number)
    
    Strategy:
    - Extract numeric part from both identifiers
    - Match on the core machine number
    - Handle cases where mapping is ambiguous
    
    Args:
        hydra_machines: List of machine IDs from HydraData
        param_machines: List of machine definitions from ParamData
        
    Returns:
        Dictionary mapping hydra machine ID to param machine def
    """
    mapping = {}
    
    for hydra_id in hydra_machines:
        # Extract numeric part (e.g., 'M-221' -> '221')
        hydra_num = hydra_id.replace("M-", "").replace("m-", "")
        
        # Find matching param machine
        for param_def in param_machines:
            # Extract numeric part (e.g., 'M221-10' -> '221')
            param_num = param_def.split("-")[0].replace("M", "").replace("m", "")
            
            if hydra_num == param_num:
                mapping[hydra_id] = param_def
                break
    
    logger.info(f"Created machine mapping: {mapping}")
    return mapping


def standardize_machine_column(
    df: pd.DataFrame,
    machine_column: str,
    mapping: Optional[Dict[str, str]] = None,
    output_column: str = "machine_key",
) -> pd.DataFrame:
    """
    Standardize machine identifier for joining.
    
    Args:
        df: DataFrame with machine column
        machine_column: Name of the machine column
        mapping: Optional mapping dictionary
        output_column: Name for the standardized machine column
        
    Returns:
        DataFrame with standardized machine column
    """
    df = df.copy()
    
    if mapping:
        df[output_column] = df[machine_column].map(mapping)
    else:
        # If no mapping, use as-is
        df[output_column] = df[machine_column]
    
    return df


# =============================================================================
# Main Join Function
# =============================================================================

def join_hydra_and_param_data(
    hydra_df: pd.DataFrame,
    param_df: pd.DataFrame,
    hydra_date_col: str = "date",
    hydra_machine_col: str = "machine_id",
    param_date_col: str = "agg_date",
    param_machine_col: str = "machine_def",
    target_column: str = "actual_scrap_qty",
    keep_unmatched: bool = True,
) -> Tuple[pd.DataFrame, JoinReport]:
    """
    Join HydraData with aggregated ParamData to create ML dataset.
    
    What it does:
    1. Standardizes date formats for both datasets
    2. Creates machine identifier mapping
    3. Performs left join from HydraData to ParamData
    4. Generates detailed join report
    5. Validates target variable preservation
    
    Join Logic:
    - Uses (machine, date) as composite join key
    - Left join preserves all HydraData rows (production records)
    - ParamData features are added where dates match
    - Unmatched rows get NaN for ParamData features
    
    Conflict Resolution:
    - If multiple ParamData rows match (shouldn't happen with proper aggregation),
      the first match is used and a warning is logged
    - Multiple HydraData rows matching one ParamData row is expected
      (multiple orders/shifts per day share same sensor state)
    
    Args:
        hydra_df: HydraData DataFrame (order/shift level)
        param_df: Aggregated ParamData DataFrame (machine/date level)
        hydra_date_col: Date column name in HydraData
        hydra_machine_col: Machine column name in HydraData
        param_date_col: Date column name in ParamData
        param_machine_col: Machine column name in ParamData
        target_column: Target variable column name
        keep_unmatched: Whether to keep HydraData rows without ParamData match
        
    Returns:
        Tuple of (joined DataFrame, JoinReport)
    """
    report = JoinReport()
    
    # Record input statistics
    report.hydra_rows_before = len(hydra_df)
    report.param_rows_before = len(param_df)
    report.target_non_null_before = hydra_df[target_column].notna().sum()
    
    logger.info(f"Starting join: {report.hydra_rows_before:,} HydraData rows, "
                f"{report.param_rows_before:,} ParamData rows")
    
    # ==========================================================================
    # Step 1: Standardize date columns
    # ==========================================================================
    # Detect HydraData date format and standardize to YYYY-MM-DD
    # Some machines use DD-MM-YYYY, others use YYYY-MM-DD
    hydra_df = standardize_date_column(
        hydra_df, 
        hydra_date_col, 
        input_format=None,  # Auto-detect format
        output_column="date_key"
    )
    
    # ParamData dates are already in YYYY-MM-DD format
    param_df = param_df.copy()
    param_df["date_key"] = param_df[param_date_col]
    
    # Record unique dates
    hydra_dates = set(hydra_df["date_key"].dropna().unique())
    param_dates = set(param_df["date_key"].dropna().unique())
    report.hydra_unique_dates = len(hydra_dates)
    report.param_unique_dates = len(param_dates)
    
    # ==========================================================================
    # Step 2: Standardize machine identifiers
    # ==========================================================================
    # Create mapping between machine identifiers
    hydra_machines = hydra_df[hydra_machine_col].unique().tolist()
    param_machines = param_df[param_machine_col].unique().tolist()
    
    machine_mapping = create_machine_mapping(hydra_machines, param_machines)
    
    # Apply mapping to HydraData
    hydra_df = standardize_machine_column(
        hydra_df,
        hydra_machine_col,
        mapping=machine_mapping,
        output_column="machine_key"
    )
    
    # ParamData already has the target format
    param_df["machine_key"] = param_df[param_machine_col]
    
    # ==========================================================================
    # Step 3: Check for duplicate keys in ParamData (should not exist)
    # ==========================================================================
    param_key_counts = param_df.groupby(["machine_key", "date_key"]).size()
    duplicates = param_key_counts[param_key_counts > 1]
    
    if len(duplicates) > 0:
        logger.warning(f"Found {len(duplicates)} duplicate keys in ParamData!")
        logger.warning("Using first occurrence for each duplicate key.")
        # Keep first occurrence
        param_df = param_df.drop_duplicates(subset=["machine_key", "date_key"], keep="first")
    
    # ==========================================================================
    # Step 4: Analyze date overlap
    # ==========================================================================
    overlapping_dates = hydra_dates.intersection(param_dates)
    report.overlapping_dates = len(overlapping_dates)
    report.matched_dates = list(overlapping_dates)
    report.unmatched_dates = list(hydra_dates - param_dates)
    
    logger.info(f"Date overlap: {report.overlapping_dates} dates match "
                f"out of {report.hydra_unique_dates} HydraData dates")
    
    # ==========================================================================
    # Step 5: Prepare ParamData columns for join
    # ==========================================================================
    # Identify ParamData feature columns (exclude key columns)
    param_key_cols = ["machine_key", "date_key", param_machine_col, param_date_col]
    param_feature_cols = [c for c in param_df.columns if c not in param_key_cols]
    
    # Select only necessary columns for join
    param_for_join = param_df[["machine_key", "date_key"] + param_feature_cols].copy()
    
    # ==========================================================================
    # Step 6: Perform the join
    # ==========================================================================
    # Left join: keep all HydraData rows, add ParamData features where available
    joined_df = hydra_df.merge(
        param_for_join,
        on=["machine_key", "date_key"],
        how="left" if keep_unmatched else "inner",
        indicator=True  # Adds '_merge' column showing join result
    )
    
    # ==========================================================================
    # Step 7: Analyze join results
    # ==========================================================================
    merge_counts = joined_df["_merge"].value_counts()
    
    report.joined_rows = len(joined_df)
    report.matched_hydra_rows = merge_counts.get("both", 0)
    report.unmatched_hydra_rows = merge_counts.get("left_only", 0)
    report.match_rate = report.matched_hydra_rows / report.hydra_rows_before
    
    # Count how many times each ParamData row was used (expected to be > 1)
    if "both" in merge_counts.index:
        param_usage = joined_df[joined_df["_merge"] == "both"].groupby(
            ["machine_key", "date_key"]
        ).size()
        report.duplicate_joins = (param_usage > 1).sum()  # Dates with multiple matches
    
    logger.info(f"Join complete: {report.matched_hydra_rows:,} rows matched, "
                f"{report.unmatched_hydra_rows:,} rows unmatched")
    
    # ==========================================================================
    # Step 8: Clean up and validate
    # ==========================================================================
    # Remove merge indicator column
    joined_df = joined_df.drop(columns=["_merge"])
    
    # Validate target variable preservation
    report.target_non_null_after = joined_df[target_column].notna().sum()
    
    if report.target_non_null_after != report.target_non_null_before:
        logger.warning(f"Target variable count changed: "
                      f"{report.target_non_null_before} -> {report.target_non_null_after}")
    
    return joined_df, report


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ml_dataset_for_machine(
    machine_config: MachineConfig,
    keep_unmatched: bool = False,
) -> Tuple[pd.DataFrame, JoinReport]:
    """
    Create complete ML dataset for a single machine.
    
    What it does:
    - Loads HydraData for the machine
    - Aggregates ParamData for the machine
    - Joins them to create feature matrix with target
    
    Args:
        machine_config: Machine configuration
        keep_unmatched: Whether to keep HydraData rows without ParamData
        
    Returns:
        Tuple of (ML dataset DataFrame, JoinReport)
    """
    # Load HydraData
    hydra_df = load_hydra_data(machine_config)
    
    # Aggregate ParamData
    param_df = aggregate_param_data_for_machine(machine_config)
    
    # Join
    return join_hydra_and_param_data(
        hydra_df, 
        param_df,
        keep_unmatched=keep_unmatched,
    )


def create_ml_dataset_all_machines(
    data_dir: Path = DATA_DIR,
    keep_unmatched: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, JoinReport]]:
    """
    Create complete ML dataset for all machines.
    
    Args:
        data_dir: Base data directory
        keep_unmatched: Whether to keep HydraData rows without ParamData
        
    Returns:
        Tuple of (combined ML dataset, dict of JoinReports by machine)
    """
    configs = get_all_machine_configs(data_dir)
    
    dfs = []
    reports = {}
    
    for machine_id, config in configs.items():
        logger.info(f"\nProcessing machine: {machine_id}")
        try:
            ml_df, report = create_ml_dataset_for_machine(config, keep_unmatched)
            dfs.append(ml_df)
            reports[machine_id] = report
        except Exception as e:
            logger.error(f"Failed to create ML dataset for {machine_id}: {e}")
    
    if not dfs:
        raise ValueError("No ML datasets could be created")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    return combined, reports


def get_ml_dataset_summary(df: pd.DataFrame, target_column: str = "actual_scrap_qty") -> Dict:
    """
    Generate summary statistics for the ML dataset.
    
    Args:
        df: ML dataset DataFrame
        target_column: Target variable column name
        
    Returns:
        Dictionary with summary statistics
    """
    # Identify feature columns (ParamData aggregations)
    feature_patterns = ["_mean", "_max", "_std", "_count", "_mode", "_delta", "_first", "_last"]
    feature_cols = [c for c in df.columns if any(p in c for p in feature_patterns)]
    
    # Identify key columns
    key_cols = ["machine_key", "date_key", "machine_id", "machine_nr", "machine_def", "agg_date"]
    key_cols = [c for c in key_cols if c in df.columns]
    
    # Calculate feature statistics
    feature_df = df[feature_cols]
    
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "feature_columns": len(feature_cols),
        "target_column": target_column,
        "target_stats": {
            "count": int(df[target_column].notna().sum()),
            "mean": float(df[target_column].mean()),
            "std": float(df[target_column].std()),
            "min": float(df[target_column].min()),
            "max": float(df[target_column].max()),
            "zeros": int((df[target_column] == 0).sum()),
            "non_zeros": int((df[target_column] > 0).sum()),
        },
        "feature_missing_pct": (feature_df.isnull().sum() / len(df) * 100).mean(),
        "rows_with_features": int(feature_df.notna().any(axis=1).sum()),
        "rows_without_features": int(feature_df.isna().all(axis=1).sum()),
        "unique_machines": df["machine_id"].nunique() if "machine_id" in df.columns else 0,
        "unique_dates": df["date_key"].nunique() if "date_key" in df.columns else 0,
    }
    
    return summary


def print_ml_dataset_summary(df: pd.DataFrame, name: str = "ML Dataset") -> None:
    """Print formatted summary of ML dataset."""
    summary = get_ml_dataset_summary(df)
    
    print(f"\n{'=' * 70}")
    print(f"ML DATASET SUMMARY: {name}")
    print("=" * 70)
    
    print(f"\nDATASET SHAPE:")
    print(f"  Total rows:          {summary['total_rows']:,}")
    print(f"  Total columns:       {summary['total_columns']}")
    print(f"  Feature columns:     {summary['feature_columns']}")
    print(f"  Unique machines:     {summary['unique_machines']}")
    print(f"  Unique dates:        {summary['unique_dates']}")
    
    print(f"\nTARGET VARIABLE ({summary['target_column']}):")
    ts = summary["target_stats"]
    print(f"  Count:               {ts['count']:,}")
    print(f"  Mean:                {ts['mean']:.2f}")
    print(f"  Std:                 {ts['std']:.2f}")
    print(f"  Range:               [{ts['min']:.0f}, {ts['max']:.0f}]")
    print(f"  Zero scrap rows:     {ts['zeros']:,} ({ts['zeros']/summary['total_rows']*100:.1f}%)")
    print(f"  Non-zero scrap rows: {ts['non_zeros']:,} ({ts['non_zeros']/summary['total_rows']*100:.1f}%)")
    
    print(f"\nFEATURE COVERAGE:")
    print(f"  Rows WITH features:  {summary['rows_with_features']:,}")
    print(f"  Rows WITHOUT features: {summary['rows_without_features']:,}")
    print(f"  Avg feature missing %: {summary['feature_missing_pct']:.1f}%")


# =============================================================================
# Main Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Testing Data Joining Module")
    print("=" * 70)
    
    # Test with single machine
    configs = get_all_machine_configs()
    
    if configs:
        machine_id = list(configs.keys())[0]
        config = configs[machine_id]
        
        print(f"\nCreating ML dataset for {machine_id}...")
        
        # With unmatched rows (for analysis)
        ml_df, report = create_ml_dataset_for_machine(config, keep_unmatched=True)
        
        print(report.summary())
        print_ml_dataset_summary(ml_df, f"ML Dataset - {machine_id}")
        
        # Show sample
        print(f"\nSample of joined data (first 5 rows, key columns):")
        key_cols = ["machine_id", "date_key", "article", "actual_scrap_qty", 
                    "Cushion_mean", "Cycle_time_mean", "Injection_pressure_mean"]
        available_cols = [c for c in key_cols if c in ml_df.columns]
        print(ml_df[available_cols].head().to_string(index=False))
    else:
        print("No machines found.")
