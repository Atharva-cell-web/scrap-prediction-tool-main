# Predictive Scrap AI Tool

A Python-based predictive analytics tool for injection molding machines that predicts `actual_scrap_qty` at order + shift level using historical data.

## 🎯 Business Objective

Reduce scrap proactively by predicting scrap quantities before production runs, enabling operators to take preventive actions.

## 📁 Project Structure

```
predictive-scrap-ai/
├── data/                      # Machine data (CSV files)
│   ├── M-221/
│   │   ├── M-221HydraData.csv    # Order/Shift level data
│   │   └── M-221ParamData.csv    # Time-series sensor data
│   ├── M-601/
│   │   ├── M-601HydraData.csv
│   │   └── M-601ParamData.csv
├── src/                       # Source code modules
│   ├── __init__.py
│   ├── config.py              # Configuration and schemas
│   ├── data_loading.py        # Data loading utilities
│   └── data_validation.py     # Data validation and quality checks
├── notebooks/                 # Jupyter notebooks for exploration
├── output/                    # Generated outputs (models, reports)
├── .venv/                     # Virtual environment (created by uv)
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## 📊 Data Overview

### HydraData (Tabular, Order/Shift Level)
Contains order-level production information:
- Order details
- Shift information
- Production quantity
- **`actual_scrap_qty`** (TARGET variable)

### ParamData (Time-series, Machine Parameters)
Long-format sensor/process data:
- `variable_name`: Sensor/parameter identifier
- `value`: Measured value
- `timestamp`: Measurement time

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone/Navigate to the project:**
   ```bash
   cd predictive-scrap-ai
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create venv with uv
   uv venv
   
   # Activate (Windows PowerShell)
   .venv\Scripts\Activate.ps1
   
   # Activate (Windows CMD)
   .venv\Scripts\activate.bat
   
   # Activate (Linux/Mac)
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv sync
   ```

### Basic Usage

```python
from src.config import discover_machines, get_all_machine_configs
from src.data_loading import load_hydra_data, load_param_data
from src.data_validation import validate_and_report

# Discover available machines
machines = discover_machines()
print(f"Available machines: {machines}")

# Get configuration for a specific machine
configs = get_all_machine_configs()
machine_config = configs["M-221"]

# Load HydraData
hydra_df = load_hydra_data(machine_config)
print(f"Loaded {len(hydra_df)} rows of HydraData")

# Load ParamData
param_df = load_param_data(machine_config)
print(f"Loaded {len(param_df)} rows of ParamData")

# Validate data
report = validate_and_report(hydra_df, data_type="hydra")
```

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation and analysis |
| numpy | Numerical computing |
| scikit-learn | Machine learning algorithms |
| matplotlib | Static visualizations |
| seaborn | Statistical visualizations |
| pyarrow | Efficient data serialization |

## 🔧 Development

### Running Tests
```bash
# Install dev dependencies
uv sync --dev

# Run tests
pytest
```

### Code Style
The project uses:
- **black** for code formatting
- **ruff** for linting

```bash
# Format code
black src/

# Lint code
ruff check src/
```

## 📋 Module Documentation

### config.py
Centralizes all configuration:
- Path definitions
- Data schemas (HydraDataSchema, ParamDataSchema)
- Machine configurations
- Validation settings

### data_loading.py
Handles data ingestion:
- Robust CSV loading with auto-detection
- Column standardization
- Multi-machine data combination
- Memory-efficient loading options

### data_validation.py
Ensures data quality:
- Missing value checks
- Duplicate detection
- Outlier identification (IQR method)
- Schema validation
- Validation reports

## ⚠️ Assumptions

1. Data follows the naming convention: `{MACHINE_ID}HydraData.csv`, `{MACHINE_ID}ParamData.csv`
2. Machine IDs are folder names in the `data/` directory
3. HydraData contains a target column named `actual_scrap_qty`
4. ParamData is in long format with `variable_name`, `value`, `timestamp` columns
5. Industrial data may have quality issues (handled gracefully)

## 🗺️ Roadmap

- [x] Data loading module
- [x] Data validation module
- [ ] Feature engineering
- [ ] Time-series aggregation
- [ ] Baseline model development
- [ ] Model evaluation framework
- [ ] Hyperparameter tuning
- [ ] Production deployment

## 📄 License

Internal use only.
