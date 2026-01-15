"""
Data Validation Module for Predictive Scrap AI

What this module does:
- Validates loaded data against expected schemas
- Checks for data quality issues (missing values, duplicates, outliers)
- Generates validation reports
- Provides actionable insights for data cleaning

Why it is needed:
- Industrial data is inherently messy and inconsistent
- Early detection of data issues prevents downstream failures
- Ensures model training uses clean, reliable data
- Documents data quality for audit purposes

Assumptions:
- Data has already been loaded using data_loading module
- Column names have been standardized (lowercase, underscores)
- Some missing data is expected in industrial settings
- Outliers should be flagged but not automatically removed
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import pandas as pd
import numpy as np

from .config import (
    HydraDataSchema,
    ParamDataSchema,
    ValidationConfig,
    HYDRA_SCHEMA,
    PARAM_SCHEMA,
    VALIDATION_CONFIG,
)

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Validation Result Types
# =============================================================================

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in the data."""
    column: Optional[str]
    issue_type: str
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        col_str = f"[{self.column}]" if self.column else "[GLOBAL]"
        return f"{self.severity.value.upper()} {col_str}: {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    dataset_name: str
    row_count: int
    column_count: int
    issues: List[ValidationIssue] = field(default_factory=list)
    column_stats: Dict[str, Dict] = field(default_factory=dict)
    is_valid: bool = True
    
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
        if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            self.is_valid = False
    
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL)
    
    def summary(self) -> str:
        """Generate a text summary of the validation report."""
        lines = [
            f"\n{'='*60}",
            f"Validation Report: {self.dataset_name}",
            f"{'='*60}",
            f"Rows: {self.row_count:,}",
            f"Columns: {self.column_count}",
            f"Status: {'VALID' if self.is_valid else 'INVALID'}",
            f"Issues: {self.critical_count} critical, {self.error_count} errors, {self.warning_count} warnings",
            f"{'-'*60}",
        ]
        
        if self.issues:
            lines.append("Issues Found:")
            for issue in self.issues:
                lines.append(f"  • {issue}")
        else:
            lines.append("No issues found. Data looks clean!")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary for serialization."""
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "is_valid": self.is_valid,
            "issue_counts": {
                "critical": self.critical_count,
                "error": self.error_count,
                "warning": self.warning_count,
            },
            "issues": [
                {
                    "column": i.column,
                    "type": i.issue_type,
                    "severity": i.severity.value,
                    "message": i.message,
                    "details": i.details,
                }
                for i in self.issues
            ],
            "column_stats": self.column_stats,
        }


# =============================================================================
# Core Validation Functions
# =============================================================================

def check_missing_values(
    df: pd.DataFrame,
    config: ValidationConfig = VALIDATION_CONFIG,
) -> List[ValidationIssue]:
    """
    Check for missing values in each column.
    
    What it does:
    - Calculates missing value percentage for each column
    - Flags columns exceeding warning/error thresholds
    
    Args:
        df: DataFrame to check
        config: Validation configuration
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        if missing_pct > config.max_missing_pct_error:
            issues.append(ValidationIssue(
                column=col,
                issue_type="missing_values",
                severity=ValidationSeverity.ERROR,
                message=f"High missing rate: {missing_pct:.1f}% ({missing_count:,} values)",
                details={"missing_count": int(missing_count), "missing_pct": float(missing_pct)},
            ))
        elif missing_pct > config.max_missing_pct_warn:
            issues.append(ValidationIssue(
                column=col,
                issue_type="missing_values",
                severity=ValidationSeverity.WARNING,
                message=f"Moderate missing rate: {missing_pct:.1f}% ({missing_count:,} values)",
                details={"missing_count": int(missing_count), "missing_pct": float(missing_pct)},
            ))
        elif missing_count > 0:
            issues.append(ValidationIssue(
                column=col,
                issue_type="missing_values",
                severity=ValidationSeverity.INFO,
                message=f"Some missing values: {missing_pct:.1f}% ({missing_count:,} values)",
                details={"missing_count": int(missing_count), "missing_pct": float(missing_pct)},
            ))
    
    return issues


def check_duplicates(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
    config: ValidationConfig = VALIDATION_CONFIG,
) -> List[ValidationIssue]:
    """
    Check for duplicate rows.
    
    What it does:
    - Checks for exact duplicate rows
    - If key columns provided, checks for duplicate keys
    
    Args:
        df: DataFrame to check
        key_columns: Columns that should form unique keys
        config: Validation configuration
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    # Check exact duplicates
    exact_dupes = df.duplicated().sum()
    exact_dupe_pct = (exact_dupes / len(df)) * 100
    
    if exact_dupe_pct > config.max_duplicate_pct:
        issues.append(ValidationIssue(
            column=None,
            issue_type="duplicates",
            severity=ValidationSeverity.ERROR,
            message=f"High duplicate row rate: {exact_dupe_pct:.1f}% ({exact_dupes:,} rows)",
            details={"duplicate_count": int(exact_dupes), "duplicate_pct": float(exact_dupe_pct)},
        ))
    elif exact_dupes > 0:
        issues.append(ValidationIssue(
            column=None,
            issue_type="duplicates",
            severity=ValidationSeverity.WARNING,
            message=f"Duplicate rows found: {exact_dupe_pct:.1f}% ({exact_dupes:,} rows)",
            details={"duplicate_count": int(exact_dupes), "duplicate_pct": float(exact_dupe_pct)},
        ))
    
    # Check key column duplicates if specified
    if key_columns:
        available_keys = [k for k in key_columns if k in df.columns]
        if available_keys:
            key_dupes = df.duplicated(subset=available_keys).sum()
            if key_dupes > 0:
                key_dupe_pct = (key_dupes / len(df)) * 100
                issues.append(ValidationIssue(
                    column=None,
                    issue_type="duplicate_keys",
                    severity=ValidationSeverity.WARNING,
                    message=f"Duplicate keys [{', '.join(available_keys)}]: {key_dupe_pct:.1f}% ({key_dupes:,} rows)",
                    details={
                        "key_columns": available_keys,
                        "duplicate_count": int(key_dupes),
                    },
                ))
    
    return issues


def check_numeric_ranges(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> List[ValidationIssue]:
    """
    Check if numeric values fall within expected ranges.
    
    What it does:
    - Validates values against min/max bounds
    - Flags out-of-range values
    
    Args:
        df: DataFrame to check
        column: Column name to check
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    if column not in df.columns:
        return issues
    
    col_data = df[column].dropna()
    
    if not pd.api.types.is_numeric_dtype(col_data):
        issues.append(ValidationIssue(
            column=column,
            issue_type="type_mismatch",
            severity=ValidationSeverity.WARNING,
            message=f"Expected numeric type, got {col_data.dtype}",
        ))
        return issues
    
    if min_value is not None:
        below_min = (col_data < min_value).sum()
        if below_min > 0:
            issues.append(ValidationIssue(
                column=column,
                issue_type="out_of_range",
                severity=ValidationSeverity.ERROR,
                message=f"{below_min:,} values below minimum ({min_value})",
                details={
                    "violation_count": int(below_min),
                    "min_allowed": min_value,
                    "actual_min": float(col_data.min()),
                },
            ))
    
    if max_value is not None:
        above_max = (col_data > max_value).sum()
        if above_max > 0:
            issues.append(ValidationIssue(
                column=column,
                issue_type="out_of_range",
                severity=ValidationSeverity.ERROR,
                message=f"{above_max:,} values above maximum ({max_value})",
                details={
                    "violation_count": int(above_max),
                    "max_allowed": max_value,
                    "actual_max": float(col_data.max()),
                },
            ))
    
    return issues


def check_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5,
) -> List[ValidationIssue]:
    """
    Check for outliers using the IQR method.
    
    What it does:
    - Calculates Q1, Q3, and IQR
    - Flags values outside Q1 - multiplier*IQR and Q3 + multiplier*IQR
    
    Why IQR method:
    - Robust to existing outliers
    - Industry standard for initial outlier detection
    - Does not assume normal distribution
    
    Args:
        df: DataFrame to check
        column: Column name to check
        multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme outliers)
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    if column not in df.columns:
        return issues
    
    col_data = df[column].dropna()
    
    if not pd.api.types.is_numeric_dtype(col_data) or len(col_data) < 4:
        return issues
    
    q1 = col_data.quantile(0.25)
    q3 = col_data.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
    outlier_count = len(outliers)
    
    if outlier_count > 0:
        outlier_pct = (outlier_count / len(col_data)) * 100
        severity = ValidationSeverity.WARNING if outlier_pct < 5 else ValidationSeverity.ERROR
        
        issues.append(ValidationIssue(
            column=column,
            issue_type="outliers",
            severity=severity,
            message=f"{outlier_count:,} potential outliers ({outlier_pct:.1f}%) using IQR method",
            details={
                "outlier_count": int(outlier_count),
                "outlier_pct": float(outlier_pct),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "q1": float(q1),
                "q3": float(q3),
            },
        ))
    
    return issues


def check_column_existence(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None,
) -> List[ValidationIssue]:
    """
    Check if required and optional columns exist.
    
    Args:
        df: DataFrame to check
        required_columns: Columns that must exist
        optional_columns: Columns that are nice to have
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    for col in required_columns:
        if col not in df.columns:
            issues.append(ValidationIssue(
                column=col,
                issue_type="missing_column",
                severity=ValidationSeverity.ERROR,
                message=f"Required column '{col}' not found",
                details={"available_columns": list(df.columns)},
            ))
    
    if optional_columns:
        for col in optional_columns:
            if col not in df.columns:
                issues.append(ValidationIssue(
                    column=col,
                    issue_type="missing_column",
                    severity=ValidationSeverity.INFO,
                    message=f"Optional column '{col}' not found",
                ))
    
    return issues


def check_data_types(
    df: pd.DataFrame,
    expected_types: Dict[str, str],
) -> List[ValidationIssue]:
    """
    Check if columns have expected data types.
    
    Args:
        df: DataFrame to check
        expected_types: Dict mapping column names to expected type strings
                       ('numeric', 'datetime', 'string', 'categorical')
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    type_checks = {
        'numeric': lambda x: pd.api.types.is_numeric_dtype(x),
        'datetime': lambda x: pd.api.types.is_datetime64_any_dtype(x),
        'string': lambda x: pd.api.types.is_string_dtype(x) or pd.api.types.is_object_dtype(x),
        'categorical': lambda x: pd.api.types.is_categorical_dtype(x) or pd.api.types.is_object_dtype(x),
    }
    
    for col, expected in expected_types.items():
        if col not in df.columns:
            continue
        
        check_func = type_checks.get(expected.lower())
        if check_func and not check_func(df[col]):
            issues.append(ValidationIssue(
                column=col,
                issue_type="type_mismatch",
                severity=ValidationSeverity.WARNING,
                message=f"Expected {expected}, got {df[col].dtype}",
                details={"expected": expected, "actual": str(df[col].dtype)},
            ))
    
    return issues


# =============================================================================
# Schema-Specific Validation
# =============================================================================

def validate_hydra_data(
    df: pd.DataFrame,
    schema: HydraDataSchema = HYDRA_SCHEMA,
    config: ValidationConfig = VALIDATION_CONFIG,
    dataset_name: str = "HydraData",
) -> ValidationReport:
    """
    Validate HydraData against expected schema.
    
    What it does:
    - Checks for required columns
    - Validates target column (actual_scrap_qty)
    - Checks for missing values and duplicates
    - Identifies potential outliers in numeric columns
    
    Args:
        df: HydraData DataFrame
        schema: Expected schema
        config: Validation configuration
        dataset_name: Name for the report
        
    Returns:
        ValidationReport with all findings
    """
    report = ValidationReport(
        dataset_name=dataset_name,
        row_count=len(df),
        column_count=len(df.columns),
    )
    
    # Check minimum rows
    if len(df) < config.min_rows:
        report.add_issue(ValidationIssue(
            column=None,
            issue_type="insufficient_data",
            severity=ValidationSeverity.CRITICAL,
            message=f"Dataset has only {len(df)} rows, minimum required is {config.min_rows}",
        ))
    
    # Check for target column
    if schema.target_column not in df.columns:
        report.add_issue(ValidationIssue(
            column=schema.target_column,
            issue_type="missing_column",
            severity=ValidationSeverity.CRITICAL,
            message=f"Target column '{schema.target_column}' not found",
            details={"available_columns": list(df.columns)},
        ))
    else:
        # Validate target column
        report.issues.extend(check_numeric_ranges(
            df, schema.target_column,
            min_value=config.target_min_value,
            max_value=config.target_max_value,
        ))
        report.issues.extend(check_outliers_iqr(df, schema.target_column))
    
    # Check missing values
    report.issues.extend(check_missing_values(df, config))
    
    # Check duplicates
    report.issues.extend(check_duplicates(df, schema.key_columns, config))
    
    # Calculate column statistics
    for col in df.columns:
        stats = {"dtype": str(df[col].dtype)}
        
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            stats.update({
                "mean": float(desc.get("mean", 0)),
                "std": float(desc.get("std", 0)),
                "min": float(desc.get("min", 0)),
                "max": float(desc.get("max", 0)),
            })
        else:
            stats["unique_count"] = int(df[col].nunique())
            stats["top_values"] = df[col].value_counts().head(5).to_dict()
        
        report.column_stats[col] = stats
    
    # Update validity based on critical/error issues
    if any(i.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR) for i in report.issues):
        report.is_valid = False
    
    return report


def validate_param_data(
    df: pd.DataFrame,
    schema: ParamDataSchema = PARAM_SCHEMA,
    config: ValidationConfig = VALIDATION_CONFIG,
    dataset_name: str = "ParamData",
) -> ValidationReport:
    """
    Validate ParamData (time-series sensor data) against expected schema.
    
    What it does:
    - Checks for required columns (variable_name, value, timestamp)
    - Validates data types
    - Checks for anomalous sensor readings
    - Reports on data coverage
    
    Args:
        df: ParamData DataFrame
        schema: Expected schema
        config: Validation configuration
        dataset_name: Name for the report
        
    Returns:
        ValidationReport with all findings
    """
    report = ValidationReport(
        dataset_name=dataset_name,
        row_count=len(df),
        column_count=len(df.columns),
    )
    
    # Check minimum rows
    if len(df) < config.min_rows:
        report.add_issue(ValidationIssue(
            column=None,
            issue_type="insufficient_data",
            severity=ValidationSeverity.CRITICAL,
            message=f"Dataset has only {len(df)} rows, minimum required is {config.min_rows}",
        ))
    
    # Check required columns for long-format data
    required_cols = [schema.variable_column, schema.value_column]
    report.issues.extend(check_column_existence(df, required_cols, [schema.timestamp_column]))
    
    # Check missing values
    report.issues.extend(check_missing_values(df, config))
    
    # Check duplicates
    report.issues.extend(check_duplicates(df, config=config))
    
    # Validate value column if present
    if schema.value_column in df.columns:
        report.issues.extend(check_outliers_iqr(df, schema.value_column, multiplier=3.0))
    
    # Report on variable coverage
    if schema.variable_column in df.columns:
        unique_vars = df[schema.variable_column].nunique()
        top_vars = df[schema.variable_column].value_counts().head(10)
        
        report.column_stats["_variable_summary"] = {
            "unique_variables": int(unique_vars),
            "top_variables": top_vars.to_dict(),
        }
        
        # Check for variables with very few readings
        var_counts = df[schema.variable_column].value_counts()
        sparse_vars = var_counts[var_counts < 10]
        if len(sparse_vars) > 0:
            report.add_issue(ValidationIssue(
                column=schema.variable_column,
                issue_type="sparse_data",
                severity=ValidationSeverity.INFO,
                message=f"{len(sparse_vars)} variables have fewer than 10 readings",
                details={"sparse_variables": sparse_vars.to_dict()},
            ))
    
    # Update validity
    if any(i.severity in (ValidationSeverity.CRITICAL, ValidationSeverity.ERROR) for i in report.issues):
        report.is_valid = False
    
    return report


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_and_report(
    df: pd.DataFrame,
    data_type: str = "hydra",
    print_report: bool = True,
) -> ValidationReport:
    """
    Validate data and optionally print report.
    
    Args:
        df: DataFrame to validate
        data_type: 'hydra' or 'param'
        print_report: Whether to print the report
        
    Returns:
        ValidationReport
    """
    if data_type.lower() == "hydra":
        report = validate_hydra_data(df)
    elif data_type.lower() == "param":
        report = validate_param_data(df)
    else:
        raise ValueError(f"Unknown data type: {data_type}. Use 'hydra' or 'param'.")
    
    if print_report:
        print(report.summary())
    
    return report


# =============================================================================
# Main Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    # Test with sample data
    print("Testing Data Validation Module")
    print("=" * 60)
    
    # Create sample HydraData for testing
    sample_hydra = pd.DataFrame({
        "order_id": ["O001", "O002", "O003", "O004", "O005"],
        "machine_id": ["M-221"] * 5,
        "shift": [1, 2, 3, 1, 2],
        "production_qty": [100, 150, 120, 130, 110],
        "actual_scrap_qty": [5, 8, 3, 12, 6],
    })
    
    print("\nValidating sample HydraData:")
    report = validate_and_report(sample_hydra, data_type="hydra")
    
    # Create sample ParamData for testing
    sample_param = pd.DataFrame({
        "variable_name": ["temp", "pressure", "temp", "pressure", "temp"] * 10,
        "value": [250.5, 120.3, 251.2, 119.8, 249.9] * 10,
        "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
        "machine_id": ["M-221"] * 50,
    })
    
    print("\nValidating sample ParamData:")
    report = validate_and_report(sample_param, data_type="param")
