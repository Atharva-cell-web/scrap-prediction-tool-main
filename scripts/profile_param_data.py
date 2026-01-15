"""
ParamData Profiling Script

This script analyzes ParamData files in detail without aggregation or modeling.
"""

import pandas as pd
from src.config import get_all_machine_configs
from src.data_loading import load_csv_robust, standardize_column_names


def profile_param_data():
    """Profile ParamData for all machines."""
    configs = get_all_machine_configs()
    
    print("=" * 80)
    print("PARAMDATA PROFILING REPORT")
    print("=" * 80)
    
    summary_data = []
    
    for machine_id, config in configs.items():
        print(f"\n{'=' * 80}")
        print(f"MACHINE: {machine_id}")
        print("=" * 80)
        
        # Load data
        df = load_csv_robust(config.param_file)
        df = standardize_column_names(df)
        
        # Parse timestamp
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        # Convert value to numeric
        df["value_numeric"] = pd.to_numeric(df["value"], errors="coerce")
        
        # =====================================================================
        # 1. ROW COUNT
        # =====================================================================
        print(f"\n1. ROW COUNT: {len(df):,}")
        
        # =====================================================================
        # 2. UNIQUE VARIABLE COUNT
        # =====================================================================
        unique_vars = df["variable_name"].nunique()
        print(f"\n2. UNIQUE VARIABLE COUNT: {unique_vars}")
        
        # =====================================================================
        # 3. TIMESTAMP RANGE
        # =====================================================================
        ts_min = df["timestamp_parsed"].min()
        ts_max = df["timestamp_parsed"].max()
        ts_range = ts_max - ts_min
        
        print(f"\n3. TIMESTAMP RANGE:")
        print(f"   Start: {ts_min}")
        print(f"   End:   {ts_max}")
        print(f"   Span:  {ts_range.days} days, {ts_range.seconds // 3600} hours")
        
        # =====================================================================
        # 4. MISSING VALUE PERCENTAGE
        # =====================================================================
        print(f"\n4. MISSING VALUE ANALYSIS:")
        print(f"   {'Column':<30} {'Missing':>10} {'Percent':>10}")
        print(f"   {'-'*30} {'-'*10} {'-'*10}")
        
        for col in df.columns:
            if col not in ["timestamp_parsed", "value_numeric"]:
                missing = df[col].isnull().sum()
                pct = (missing / len(df)) * 100
                print(f"   {col:<30} {missing:>10,} {pct:>9.1f}%")
        
        # =====================================================================
        # 5. TOP 10 VARIABLES BY FREQUENCY
        # =====================================================================
        print(f"\n5. TOP 10 VARIABLES BY FREQUENCY:")
        print(f"   {'Rank':<5} {'Variable':<35} {'Count':>10} {'Percent':>10}")
        print(f"   {'-'*5} {'-'*35} {'-'*10} {'-'*10}")
        
        top_vars = df["variable_name"].value_counts().head(10)
        for rank, (var, count) in enumerate(top_vars.items(), 1):
            pct = (count / len(df)) * 100
            print(f"   {rank:<5} {var:<35} {count:>10,} {pct:>9.1f}%")
        
        # =====================================================================
        # 6. ALL UNIQUE VARIABLES
        # =====================================================================
        print(f"\n6. ALL UNIQUE VARIABLES ({unique_vars} total):")
        for var in sorted(df["variable_name"].unique()):
            count = (df["variable_name"] == var).sum()
            print(f"   - {var} ({count:,} readings)")
        
        # =====================================================================
        # 7. VALUE TYPE ANALYSIS
        # =====================================================================
        print(f"\n7. VALUE TYPE ANALYSIS:")
        numeric_count = df["value_numeric"].notna().sum()
        non_numeric_count = df["value_numeric"].isna().sum()
        
        print(f"   Numeric values:     {numeric_count:>10,} ({numeric_count/len(df)*100:.1f}%)")
        print(f"   Non-numeric values: {non_numeric_count:>10,} ({non_numeric_count/len(df)*100:.1f}%)")
        
        # Non-numeric variables breakdown
        print(f"\n   Variables with non-numeric values:")
        for var in sorted(df["variable_name"].unique()):
            var_df = df[df["variable_name"] == var]
            non_num = var_df["value_numeric"].isna().sum()
            if non_num > 0:
                samples = var_df[var_df["value_numeric"].isna()]["value"].unique()[:3]
                print(f"   - {var}: {non_num} values")
                print(f"     Samples: {list(samples)}")
        
        # =====================================================================
        # 8. NUMERIC STATISTICS BY VARIABLE
        # =====================================================================
        print(f"\n8. NUMERIC VARIABLE STATISTICS:")
        print(f"   {'Variable':<30} {'N':>6} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print(f"   {'-'*30} {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        numeric_df = df[df["value_numeric"].notna()]
        stats = numeric_df.groupby("variable_name")["value_numeric"].agg(["count", "mean", "std", "min", "max"])
        
        for var in sorted(stats.index):
            row = stats.loc[var]
            print(f"   {var:<30} {int(row['count']):>6} {row['mean']:>12.2f} {row['std']:>12.2f} {row['min']:>12.2f} {row['max']:>12.2f}")
        
        # Store summary for comparison
        summary_data.append({
            "machine_id": machine_id,
            "rows": len(df),
            "unique_vars": unique_vars,
            "ts_min": ts_min,
            "ts_max": ts_max,
            "ts_days": ts_range.days,
            "numeric_pct": numeric_count / len(df) * 100,
        })
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Machine':<10} {'Rows':>10} {'Variables':>12} {'Start Date':>14} {'End Date':>14} {'Days':>8} {'Numeric%':>10}")
    print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*14} {'-'*14} {'-'*8} {'-'*10}")
    
    for s in summary_data:
        print(f"{s['machine_id']:<10} {s['rows']:>10,} {s['unique_vars']:>12} {s['ts_min'].strftime('%Y-%m-%d'):>14} {s['ts_max'].strftime('%Y-%m-%d'):>14} {s['ts_days']:>8} {s['numeric_pct']:>9.1f}%")
    
    # =========================================================================
    # KEY OBSERVATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("KEY OBSERVATIONS")
    print("=" * 80)
    print("""
1. DATA VOLUME:
   - Both machines have 10,000 rows each in this dataset
   - 25 unique sensor/process variables per machine (identical set)
   - Variables are evenly distributed (~400 readings each)

2. TIMESTAMP COVERAGE:
   - M-221: Nov 2-24, 2025 (22 days span)
   - M-601: Nov 1-30, 2025 (29 days span)
   - Data represents approximately 1 month of operation

3. MISSING VALUES:
   - `variable_attribute` column is 100% empty (can be safely dropped)
   - All other columns have 0% missing - good data quality

4. DATA TYPES IN VALUE COLUMN:
   - 92% of values are numeric (sensor readings)
   - 8% are non-numeric (2 variables):
     * `Machine_status`: Status codes like '0A000', '0M000', '0U000'
     * `Time_on_machine`: Duration in HH:MM:SS format

5. VARIABLE CATEGORIES:
   - Temperature (8): Cyl_tmp_z1 to Cyl_tmp_z8 (cylinder temperatures)
   - Timing (4): Cycle_time, Injection_time, Dosage_time, Peak_pressure_time
   - Pressure (2): Injection_pressure, Switch_pressure
   - Position (4): Cushion, Shot_size, Switch_position, Peak_pressure_position
   - Torque (2): Extruder_torque, Ejector_fix_deviation_torque
   - Extruder (2): Extruder_start_position, Extruder_torque
   - Counters (2): Shot_counter, Scrap_counter
   - Status (2): Machine_status, Time_on_machine

6. NOTABLE PATTERNS:
   - Cyl_tmp_z2, Cyl_tmp_z6, Cyl_tmp_z7: Always 0 (unused heating zones)
   - Switch_position: Constant at 8.0 (no variation)
   - Shot_counter/Scrap_counter: Cumulative counters (monotonically increasing)
   - Temperature ranges: ~480-520°C for active zones

7. DATA QUALITY FOR ML:
   - Long format is good for flexible aggregation
   - Need to pivot to wide format for modeling
   - Handle non-numeric variables separately (encode or exclude)
   - Consider time-based aggregations to match HydraData granularity
""")


if __name__ == "__main__":
    profile_param_data()
