"""
Script to create and analyze the final ML dataset.
"""

import pandas as pd
import logging
from src.config import get_all_machine_configs
from src.data_joining import (
    create_ml_dataset_for_machine,
    create_ml_dataset_all_machines,
    print_ml_dataset_summary,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    print("=" * 80)
    print("CREATING FINAL ML DATASET")
    print("=" * 80)
    
    configs = get_all_machine_configs()
    
    all_dfs = []
    all_reports = []
    
    for machine_id, config in configs.items():
        print(f"\n{'=' * 80}")
        print(f"MACHINE: {machine_id}")
        print("=" * 80)
        
        # Create ML dataset with unmatched rows to see full picture
        ml_df_full, report_full = create_ml_dataset_for_machine(config, keep_unmatched=True)
        
        # Also create matched-only version
        ml_df_matched, report_matched = create_ml_dataset_for_machine(config, keep_unmatched=False)
        
        print(report_full.summary())
        
        # Show summary for matched-only dataset
        print("\n" + "-" * 70)
        print("MATCHED ROWS ONLY (for modeling):")
        print("-" * 70)
        print(f"  Rows: {len(ml_df_matched):,}")
        
        if len(ml_df_matched) > 0:
            print_ml_dataset_summary(ml_df_matched, f"ML Dataset (matched) - {machine_id}")
            
            # Show sample of matched data
            print(f"\nSample of MATCHED data (first 5 rows):")
            key_cols = [
                "machine_id", "date_key", "article", 
                "actual_scrap_qty", "shift_scrap",
                "Cushion_mean", "Cycle_time_mean", 
                "Injection_pressure_mean", "Machine_status_mode"
            ]
            available_cols = [c for c in key_cols if c in ml_df_matched.columns]
            print(ml_df_matched[available_cols].head().to_string(index=False))
            
            # Feature column sample
            print(f"\nFeature columns available ({len([c for c in ml_df_matched.columns if '_mean' in c or '_max' in c or '_std' in c])} aggregations):")
            feature_cols = [c for c in ml_df_matched.columns if '_mean' in c][:10]
            for col in feature_cols:
                print(f"  {col}: min={ml_df_matched[col].min():.2f}, max={ml_df_matched[col].max():.2f}")
            
            all_dfs.append(ml_df_matched)
            all_reports.append(report_matched)
    
    # Combined summary
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        print("\n" + "=" * 80)
        print("COMBINED ML DATASET (ALL MACHINES, MATCHED ONLY)")
        print("=" * 80)
        print_ml_dataset_summary(combined_df, "Combined ML Dataset")
        
        # Target distribution by machine
        print("\nTARGET (actual_scrap_qty) BY MACHINE:")
        for machine_id in combined_df["machine_id"].unique():
            machine_data = combined_df[combined_df["machine_id"] == machine_id]
            target = machine_data["actual_scrap_qty"]
            print(f"  {machine_id}: n={len(machine_data)}, mean={target.mean():.1f}, "
                  f"std={target.std():.1f}, range=[{target.min():.0f}, {target.max():.0f}]")
        
        # Feature completeness
        print("\nFEATURE COMPLETENESS:")
        feature_cols = [c for c in combined_df.columns if "_mean" in c or "_max" in c]
        missing_pct = combined_df[feature_cols].isnull().sum() / len(combined_df) * 100
        
        if missing_pct.max() > 0:
            print("  Features with missing values:")
            for col, pct in missing_pct[missing_pct > 0].items():
                print(f"    {col}: {pct:.1f}% missing")
        else:
            print("  All features are complete - no missing values!")
        
        # Ready for modeling message
        print("\n" + "=" * 80)
        print("ML DATASET READY FOR MODELING")
        print("=" * 80)
        print(f"""
Dataset is ready for model training:
  - Total samples: {len(combined_df):,}
  - Features: {len([c for c in combined_df.columns if any(p in c for p in ['_mean', '_max', '_std', '_count', '_mode', '_delta'])])}
  - Target: actual_scrap_qty (continuous)
  - Machines: {combined_df['machine_id'].nunique()}
  
Note: Only rows with matching ParamData are included.
Unmatched rows ({sum(r.unmatched_hydra_rows for r in all_reports):,}) were excluded.
        """)


if __name__ == "__main__":
    main()
