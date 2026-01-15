"""
Configuration Module for Predictive Scrap AI

What this module does:
- Centralizes all configuration settings for the project
- Defines file paths, column names, and data schemas
- Provides machine-specific configurations
- Makes it easy to adapt to new machines or data changes

Why it is needed:
- Avoids hardcoding paths and column names throughout the codebase
- Single source of truth for project configuration
- Makes the code maintainable and adaptable to industrial data changes
- Enables easy extension to new machines without code changes

Assumptions:
- Data folder structure follows: data/{MACHINE_ID}/HydraData.csv, ParamData.csv
- Machine IDs are prefixed with 'M-' (e.g., M-221, M-601)
- HydraData contains order/shift level tabular data
- ParamData contains time-series sensor data in long format
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# Path Configuration
# =============================================================================

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directory
DATA_DIR = PROJECT_ROOT / "data"

# Output directories (created on demand)
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"


# =============================================================================
# Data File Naming Convention
# =============================================================================

HYDRA_DATA_SUFFIX = "HydraData.csv"
PARAM_DATA_SUFFIX = "ParamData.csv"


# =============================================================================
# HydraData Schema Definition
# =============================================================================

@dataclass
class HydraDataSchema:
    """
    Schema definition for HydraData (Order/Shift level tabular data).
    
    This schema defines expected columns and their data types.
    Actual columns may vary by machine - this serves as a reference.
    """
    # Target variable - the value we want to predict
    target_column: str = "actual_scrap_qty"
    
    # Expected key columns for grouping/joining
    key_columns: List[str] = field(default_factory=lambda: [
        "order_id",      # Unique order identifier
        "shift",         # Shift identifier (e.g., 1, 2, 3 or A, B, C)
        "machine_id",    # Machine identifier
    ])
    
    # Common numeric columns (production metrics)
    numeric_columns: List[str] = field(default_factory=lambda: [
        "production_qty",
        "actual_scrap_qty",
        "planned_qty",
    ])
    
    # Date/time columns
    datetime_columns: List[str] = field(default_factory=lambda: [
        "order_date",
        "start_time",
        "end_time",
    ])


@dataclass
class ParamDataSchema:
    """
    Schema definition for ParamData (Time-series sensor data in long format).
    
    Long format means each row is a single observation:
    - variable_name: the sensor/parameter name
    - value: the measured value
    - timestamp: when the measurement was taken
    """
    # Column containing the parameter/sensor name
    variable_column: str = "variable_name"
    
    # Column containing the measured value
    value_column: str = "value"
    
    # Column containing the timestamp
    timestamp_column: str = "timestamp"
    
    # Optional columns for linking to HydraData
    linkage_columns: List[str] = field(default_factory=lambda: [
        "order_id",
        "machine_id",
    ])


# =============================================================================
# Machine Configuration
# =============================================================================

@dataclass
class MachineConfig:
    """Configuration for a specific injection molding machine."""
    machine_id: str
    data_path: Path
    hydra_file: Path
    param_file: Path
    description: Optional[str] = None
    
    @classmethod
    def from_machine_id(cls, machine_id: str, data_dir: Path = DATA_DIR) -> "MachineConfig":
        """
        Create MachineConfig from machine ID.
        
        Args:
            machine_id: Machine identifier (e.g., 'M-221')
            data_dir: Base data directory
            
        Returns:
            MachineConfig instance
        """
        machine_path = data_dir / machine_id
        return cls(
            machine_id=machine_id,
            data_path=machine_path,
            hydra_file=machine_path / f"{machine_id}{HYDRA_DATA_SUFFIX}",
            param_file=machine_path / f"{machine_id}{PARAM_DATA_SUFFIX}",
        )


def discover_machines(data_dir: Path = DATA_DIR) -> List[str]:
    """
    Automatically discover machine IDs from the data directory.
    
    Assumptions:
    - Each subdirectory in data_dir represents a machine
    - Machine directories contain both HydraData and ParamData files
    
    Args:
        data_dir: Base data directory to scan
        
    Returns:
        List of machine IDs found
    """
    if not data_dir.exists():
        return []
    
    machines = []
    for path in data_dir.iterdir():
        if path.is_dir() and not path.name.startswith('.'):
            # Check if required files exist
            hydra_file = path / f"{path.name}{HYDRA_DATA_SUFFIX}"
            param_file = path / f"{path.name}{PARAM_DATA_SUFFIX}"
            
            if hydra_file.exists() and param_file.exists():
                machines.append(path.name)
    
    return sorted(machines)


def get_all_machine_configs(data_dir: Path = DATA_DIR) -> Dict[str, MachineConfig]:
    """
    Get configurations for all discovered machines.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Dictionary mapping machine_id to MachineConfig
    """
    machines = discover_machines(data_dir)
    return {m: MachineConfig.from_machine_id(m, data_dir) for m in machines}


# =============================================================================
# Default Instances
# =============================================================================

# Schema instances with default settings
HYDRA_SCHEMA = HydraDataSchema()
PARAM_SCHEMA = ParamDataSchema()


# =============================================================================
# Validation Settings
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for data validation rules."""
    
    # Maximum allowed missing value percentage before warning
    max_missing_pct_warn: float = 5.0
    
    # Maximum allowed missing value percentage before error
    max_missing_pct_error: float = 20.0
    
    # Minimum required rows for a valid dataset
    min_rows: int = 10
    
    # Maximum allowed duplicate percentage
    max_duplicate_pct: float = 5.0
    
    # Target column constraints
    target_min_value: float = 0.0  # Scrap cannot be negative
    target_max_value: Optional[float] = None  # No upper bound by default


# Default validation config
VALIDATION_CONFIG = ValidationConfig()


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    # Quick test of configuration
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Data Directory Exists: {DATA_DIR.exists()}")
    print(f"\nDiscovered Machines: {discover_machines()}")
    
    for machine_id, config in get_all_machine_configs().items():
        print(f"\n{machine_id}:")
        print(f"  HydraData: {config.hydra_file} (exists: {config.hydra_file.exists()})")
        print(f"  ParamData: {config.param_file} (exists: {config.param_file.exists()})")
