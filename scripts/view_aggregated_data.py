"""
Script to display detailed view of aggregated ParamData.
"""

import pandas as pd
from src.config import get_all_machine_configs
from src.data_aggregation import aggregate_param_data_for_machine


def main():
    configs = get_all_machine_configs()

    print("=" * 80)
    print("AGGREGATED PARAMDATA - DETAILED VIEW")
    print("=" * 80)

    for machine_id, config in configs.items():
        print(f"\n{'=' * 80}")
        print(f"MACHINE: {machine_id}")
        print("=" * 80)
        
        agg_df = aggregate_param_data_for_machine(config)
        
        # Basic shape
        print(f"\nSHAPE: {agg_df.shape[0]} rows x {agg_df.shape[1]} columns")
        
        # Grouping columns
        print(f"\nGROUPING COLUMNS:")
        print(f"  machine_def: {list(agg_df['machine_def'].unique())}")
        print(f"  agg_date: {sorted(agg_df['agg_date'].unique())}")
        
        # Feature columns by type
        print(f"\nFEATURE COLUMNS BY TYPE:")
        
        # Mean features
        mean_cols = [c for c in agg_df.columns if c.endswith("_mean")]
        print(f"\n  MEAN features ({len(mean_cols)}):")
        for c in mean_cols[:6]:
            print(f"    {c}: min={agg_df[c].min():.2f}, max={agg_df[c].max():.2f}")
        print(f"    ... and {len(mean_cols) - 6} more")
        
        # Max features
        max_cols = [c for c in agg_df.columns if c.endswith("_max")]
        print(f"\n  MAX features ({len(max_cols)}):")
        for c in max_cols[:6]:
            print(f"    {c}: min={agg_df[c].min():.2f}, max={agg_df[c].max():.2f}")
        
        # Std features
        std_cols = [c for c in agg_df.columns if c.endswith("_std")]
        print(f"\n  STD features ({len(std_cols)}):")
        for c in std_cols[:6]:
            print(f"    {c}: min={agg_df[c].min():.4f}, max={agg_df[c].max():.4f}")
        
        # Count features
        count_cols = [c for c in agg_df.columns if c.endswith("_count")]
        print(f"\n  COUNT features ({len(count_cols)}):")
        for c in count_cols[:4]:
            print(f"    {c}: min={int(agg_df[c].min())}, max={int(agg_df[c].max())}")
        
        # Mode (categorical) features
        mode_cols = [c for c in agg_df.columns if c.endswith("_mode")]
        print(f"\n  MODE (categorical) features ({len(mode_cols)}):")
        for c in mode_cols:
            print(f"    {c}: {agg_df[c].value_counts().to_dict()}")
        
        # Delta (counter) features
        delta_cols = [c for c in agg_df.columns if c.endswith("_delta")]
        print(f"\n  DELTA (counter change) features ({len(delta_cols)}):")
        for c in delta_cols:
            print(f"    {c}: min={agg_df[c].min():.0f}, max={agg_df[c].max():.0f}")
        
        # Time_on_machine verification
        print(f"\n  TIME_ON_MACHINE (converted from HH:MM:SS to seconds):")
        mean_min = agg_df["Time_on_machine_mean"].min()
        mean_max = agg_df["Time_on_machine_mean"].max()
        max_val = agg_df["Time_on_machine_max"].max()
        print(f"    mean range: {mean_min:.0f} - {mean_max:.0f} seconds")
        print(f"    max value: {max_val:.0f} seconds ({max_val/3600:.1f} hours)")
        
        # Missing values check
        missing = agg_df.isnull().sum()
        cols_with_missing = missing[missing > 0]
        if len(cols_with_missing) > 0:
            print(f"\n  COLUMNS WITH MISSING VALUES:")
            for col_name, cnt in cols_with_missing.items():
                print(f"    {col_name}: {cnt} missing")
        else:
            print(f"\n  NO MISSING VALUES - Data is complete!")
        
        # Sample output
        print(f"\n  SAMPLE OUTPUT (first 3 rows, key columns):")
        key_cols = ["machine_def", "agg_date", "Cushion_mean", "Cycle_time_mean", 
                    "Injection_pressure_mean", "Machine_status_mode", "Scrap_counter_delta"]
        available_cols = [c for c in key_cols if c in agg_df.columns]
        print(agg_df[available_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
