"""
Predictive Scrap AI - Source Package

This package contains modules for data loading, validation, aggregation,
joining, and processing for the Predictive Scrap AI tool used in injection molding machines.

Modules:
    config: Configuration settings, schemas, and path definitions
    data_loading: Functions for loading HydraData and ParamData
    data_validation: Data quality checks and validation reports
    data_aggregation: Time-series aggregation from long to wide format
    data_joining: Joining HydraData with aggregated ParamData for ML
"""

__version__ = "0.1.0"

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    discover_machines,
    get_all_machine_configs,
    MachineConfig,
)

from .data_loading import (
    load_hydra_data,
    load_param_data,
    load_all_hydra_data,
    load_all_param_data,
)

from .data_validation import (
    validate_hydra_data,
    validate_param_data,
    validate_and_report,
    ValidationReport,
)

from .data_aggregation import (
    aggregate_param_data,
    aggregate_param_data_for_machine,
    aggregate_all_param_data,
)

from .data_joining import (
    join_hydra_and_param_data,
    create_ml_dataset_for_machine,
    create_ml_dataset_all_machines,
    JoinReport,
)
