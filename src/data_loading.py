"""
Data Loading Module for Predictive Scrap AI

What this module does:
- Loads HydraData (order/shift level tabular data) from CSV files
- Loads ParamData (time-series sensor data) from CSV files
- Handles multiple machines with consistent interface
- Provides memory-efficient loading options for large datasets
- Standardizes column names and basic type conversions

Why it is needed:
- Industrial data often comes in inconsistent formats
- Centralized loading ensures consistent preprocessing
- Enables efficient handling of large sensor datasets
- Provides a clean API for downstream processing

Assumptions:
- CSV files use standard delimiters (comma, semicolon, or tab auto-detected)
- Files may have varying encodings (utf-8, latin-1 common in industrial systems)
- Column names may have inconsistent casing or whitespace
- Timestamps may be in various formats
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import pandas as pd
import numpy as np

from .config import (
    DATA_DIR,
    MachineConfig,
    HydraDataSchema,
    ParamDataSchema,
    HYDRA_SCHEMA,
    PARAM_SCHEMA,
    discover_machines,
    get_all_machine_configs,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Column Name Standardization
# =============================================================================

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for consistency.
    
    What it does:
    - Converts to lowercase
    - Replaces spaces with underscores
    - Strips leading/trailing whitespace
    - Removes special characters
    
    Args:
        df: DataFrame with potentially messy column names
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace(r'[^\w\s]', '', regex=True)
    )
    return df


# =============================================================================
# CSV Loading with Auto-Detection
# =============================================================================

def load_csv_robust(
    filepath: Path,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict] = None,
    parse_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load CSV with automatic encoding and delimiter detection.
    
    What it does:
    - Tries multiple encodings if not specified
    - Auto-detects delimiter if not specified
    - Handles common industrial data quirks
    
    Why it is needed:
    - Industrial systems often export data with varying encodings
    - Delimiter conventions vary between systems
    
    Args:
        filepath: Path to CSV file
        encoding: Character encoding (auto-detected if None)
        delimiter: Column delimiter (auto-detected if None)
        nrows: Number of rows to read (None for all)
        usecols: Specific columns to load
        dtype: Column data types
        parse_dates: Columns to parse as dates
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be parsed
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Encodings to try in order of likelihood for industrial systems
    encodings_to_try = [encoding] if encoding else ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    # Delimiters to try
    delimiters_to_try = [delimiter] if delimiter else [',', ';', '\t', '|']
    
    last_error = None
    
    for enc in encodings_to_try:
        for delim in delimiters_to_try:
            try:
                df = pd.read_csv(
                    filepath,
                    encoding=enc,
                    delimiter=delim,
                    nrows=nrows,
                    usecols=usecols,
                    dtype=dtype,
                    parse_dates=parse_dates,
                    low_memory=False,  # Avoid mixed type warnings
                )
                
                # Check if parsing was successful (more than one column usually)
                if len(df.columns) > 1 or delimiter is not None:
                    logger.debug(f"Successfully loaded {filepath} with encoding={enc}, delimiter={repr(delim)}")
                    return df
                    
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_error = e
                continue
    
    raise ValueError(f"Could not parse {filepath}. Last error: {last_error}")


# =============================================================================
# HydraData Loading
# =============================================================================

def load_hydra_data(
    machine_config: MachineConfig,
    schema: HydraDataSchema = HYDRA_SCHEMA,
    standardize_columns: bool = True,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load HydraData for a specific machine.
    
    What it does:
    - Loads order/shift level data from CSV
    - Standardizes column names
    - Adds machine_id column for multi-machine analysis
    - Performs basic type inference
    
    Args:
        machine_config: Configuration for the machine
        schema: Schema definition for validation reference
        standardize_columns: Whether to standardize column names
        nrows: Number of rows to load (None for all)
        
    Returns:
        DataFrame with HydraData
    """
    logger.info(f"Loading HydraData for machine: {machine_config.machine_id}")
    
    df = load_csv_robust(machine_config.hydra_file, nrows=nrows)
    
    if standardize_columns:
        df = standardize_column_names(df)
    
    # Add machine_id if not present
    if 'machine_id' not in df.columns:
        df['machine_id'] = machine_config.machine_id
    
    # Log basic info
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Check for target column
    if schema.target_column in df.columns:
        non_null = df[schema.target_column].notna().sum()
        logger.info(f"  Target column '{schema.target_column}': {non_null:,} non-null values")
    else:
        logger.warning(f"  Target column '{schema.target_column}' not found in data")
    
    return df


def load_all_hydra_data(
    data_dir: Path = DATA_DIR,
    schema: HydraDataSchema = HYDRA_SCHEMA,
    standardize_columns: bool = True,
) -> pd.DataFrame:
    """
    Load and combine HydraData from all machines.
    
    What it does:
    - Discovers all machines in data directory
    - Loads HydraData for each machine
    - Combines into single DataFrame with machine_id column
    
    Args:
        data_dir: Base data directory
        schema: Schema definition
        standardize_columns: Whether to standardize column names
        
    Returns:
        Combined DataFrame with all machines' HydraData
    """
    configs = get_all_machine_configs(data_dir)
    
    if not configs:
        raise ValueError(f"No machine data found in {data_dir}")
    
    dfs = []
    for machine_id, config in configs.items():
        try:
            df = load_hydra_data(config, schema, standardize_columns)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load HydraData for {machine_id}: {e}")
    
    if not dfs:
        raise ValueError("No HydraData could be loaded from any machine")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined HydraData: {len(combined):,} total rows from {len(dfs)} machines")
    
    return combined


# =============================================================================
# ParamData Loading
# =============================================================================

def load_param_data(
    machine_config: MachineConfig,
    schema: ParamDataSchema = PARAM_SCHEMA,
    standardize_columns: bool = True,
    nrows: Optional[int] = None,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load ParamData (time-series sensor data) for a specific machine.
    
    What it does:
    - Loads long-format sensor/parameter data
    - Optionally filters to specific variables
    - Standardizes column names
    - Adds machine_id for multi-machine analysis
    
    Why long format:
    - Industrial sensor data often has hundreds of parameters
    - Long format is more storage-efficient for sparse data
    - Easier to add new sensors without schema changes
    
    Args:
        machine_config: Configuration for the machine
        schema: Schema definition
        standardize_columns: Whether to standardize column names
        nrows: Number of rows to load (None for all)
        variables: Specific variable names to filter (None for all)
        
    Returns:
        DataFrame with ParamData
    """
    logger.info(f"Loading ParamData for machine: {machine_config.machine_id}")
    
    df = load_csv_robust(machine_config.param_file, nrows=nrows)
    
    if standardize_columns:
        df = standardize_column_names(df)
    
    # Add machine_id if not present
    if 'machine_id' not in df.columns:
        df['machine_id'] = machine_config.machine_id
    
    # Filter to specific variables if requested
    if variables is not None and schema.variable_column in df.columns:
        # Standardize variable names for comparison
        df_var_col = df[schema.variable_column].str.strip().str.lower()
        variables_lower = [v.strip().lower() for v in variables]
        df = df[df_var_col.isin(variables_lower)]
        logger.info(f"  Filtered to {len(variables)} variables: {len(df):,} rows remain")
    
    # Log basic info
    logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    if schema.variable_column in df.columns:
        unique_vars = df[schema.variable_column].nunique()
        logger.info(f"  Unique variables: {unique_vars}")
    
    return df


def load_all_param_data(
    data_dir: Path = DATA_DIR,
    schema: ParamDataSchema = PARAM_SCHEMA,
    standardize_columns: bool = True,
    variables: Optional[List[str]] = None,
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load and combine ParamData from all machines.
    
    What it does:
    - Discovers all machines in data directory
    - Loads ParamData for each machine
    - Combines into single DataFrame
    - Optionally samples for memory efficiency
    
    Args:
        data_dir: Base data directory
        schema: Schema definition
        standardize_columns: Whether to standardize column names
        variables: Specific variable names to filter
        sample_frac: Fraction of data to sample (None for all)
        
    Returns:
        Combined DataFrame with all machines' ParamData
    """
    configs = get_all_machine_configs(data_dir)
    
    if not configs:
        raise ValueError(f"No machine data found in {data_dir}")
    
    dfs = []
    for machine_id, config in configs.items():
        try:
            df = load_param_data(config, schema, standardize_columns, variables=variables)
            
            if sample_frac is not None and 0 < sample_frac < 1:
                df = df.sample(frac=sample_frac, random_state=42)
                logger.info(f"  Sampled to {len(df):,} rows ({sample_frac*100:.1f}%)")
            
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to load ParamData for {machine_id}: {e}")
    
    if not dfs:
        raise ValueError("No ParamData could be loaded from any machine")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined ParamData: {len(combined):,} total rows from {len(dfs)} machines")
    
    return combined


# =============================================================================
# Data Summary Functions
# =============================================================================

def get_data_summary(df: pd.DataFrame, name: str = "DataFrame") -> Dict:
    """
    Generate a summary of loaded data.
    
    Args:
        df: DataFrame to summarize
        name: Name for reporting
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
    
    # Add numeric column stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    return summary


def print_data_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print a formatted summary of the data."""
    summary = get_data_summary(df, name)
    
    print(f"\n{'='*60}")
    print(f"Data Summary: {summary['name']}")
    print(f"{'='*60}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Columns: {summary['columns']}")
    print(f"Memory: {summary['memory_mb']:.2f} MB")
    
    print(f"\nColumn Details:")
    print("-" * 40)
    for col in summary['column_names']:
        dtype = summary['dtypes'][col]
        missing = summary['missing_counts'][col]
        missing_pct = summary['missing_pct'][col]
        print(f"  {col}: {dtype} ({missing:,} missing, {missing_pct:.1f}%)")


# =============================================================================
# Main Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    # Test data loading
    print("Testing Data Loading Module")
    print("=" * 60)
    
    # Discover machines
    machines = discover_machines()
    print(f"\nDiscovered machines: {machines}")
    
    if machines:
        # Load data for first machine
        config = MachineConfig.from_machine_id(machines[0])
        
        print(f"\nLoading data for {machines[0]}...")
        
        # Load HydraData
        try:
            hydra_df = load_hydra_data(config)
            print_data_summary(hydra_df, f"HydraData - {machines[0]}")
        except Exception as e:
            print(f"Error loading HydraData: {e}")
        
        # Load ParamData
        try:
            param_df = load_param_data(config, nrows=10000)  # Limit for testing
            print_data_summary(param_df, f"ParamData - {machines[0]}")
        except Exception as e:
            print(f"Error loading ParamData: {e}")
    else:
        print("No machines found. Please check data directory.")
