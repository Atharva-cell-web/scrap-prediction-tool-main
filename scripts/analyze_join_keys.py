"""
Script to analyze join keys between HydraData and ParamData.
"""

import pandas as pd
from src.config import get_all_machine_configs
from src.data_loading import load_hydra_data
from src.data_aggregation import aggregate_param_data_for_machine


def main():
    configs = get_all_machine_configs()
    config = configs["M-221"]

    # Load HydraData
    hydra = load_hydra_data(config)

    # Load aggregated ParamData
    param_agg = aggregate_param_data_for_machine(config)

    print("=" * 70)
    print("KEY COLUMNS FOR JOINING")
    print("=" * 70)

    print("\nHYDRADATA:")
    print(f"  machine_nr samples: {list(hydra['machine_nr'].unique()[:3])}")
    print(f"  machine_id samples: {list(hydra['machine_id'].unique()[:3])}")
    print(f"  date format samples: {list(hydra['date'].unique()[:5])}")
    print(f"  date dtype: {hydra['date'].dtype}")

    print("\nAGGREGATED PARAMDATA:")
    print(f"  machine_def samples: {list(param_agg['machine_def'].unique()[:3])}")
    print(f"  agg_date format samples: {list(param_agg['agg_date'].unique()[:5])}")
    print(f"  agg_date dtype: {param_agg['agg_date'].dtype}")

    print("\nDATE OVERLAP CHECK:")
    hydra_dates = set(hydra["date"].unique())
    param_dates = set(param_agg["agg_date"].unique())
    print(f"  HydraData unique dates: {len(hydra_dates)}")
    print(f"  ParamData unique dates: {len(param_dates)}")

    # Convert hydra dates to same format (DD-MM-YYYY -> YYYY-MM-DD)
    hydra["date_std"] = pd.to_datetime(
        hydra["date"], format="%d-%m-%Y", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    hydra_dates_std = set(hydra["date_std"].dropna().unique())

    print(f"\n  HydraData dates (standardized): {sorted(list(hydra_dates_std))[:5]}")
    print(f"  ParamData dates: {sorted(list(param_dates))[:5]}")

    overlap = hydra_dates_std.intersection(param_dates)
    print(f"\n  Overlapping dates: {len(overlap)}")
    print(f"  Overlapping date list: {sorted(list(overlap))}")

    # Machine mapping
    print("\n" + "=" * 70)
    print("MACHINE IDENTIFIER MAPPING")
    print("=" * 70)
    print(f"\n  HydraData machine_nr: {hydra['machine_nr'].unique()}")
    print(f"  HydraData machine_id: {hydra['machine_id'].unique()}")
    print(f"  ParamData machine_def: {param_agg['machine_def'].unique()}")

    # The machine_def in ParamData is like 'M221-10', machine_nr in HydraData is 'M-221'
    # Need to create a mapping or standardize

    print("\n" + "=" * 70)
    print("HYDRA ROWS PER DATE (after standardization)")
    print("=" * 70)
    date_counts = hydra.groupby("date_std").size().sort_index()
    print(f"\nRows per date (sample):")
    for date, count in list(date_counts.items())[:10]:
        print(f"  {date}: {count} rows")
    
    print(f"\n  Total unique dates in HydraData: {len(date_counts)}")
    print(f"  Total rows in HydraData: {len(hydra)}")
    print(f"  Mean rows per date: {len(hydra) / len(date_counts):.1f}")


if __name__ == "__main__":
    main()
