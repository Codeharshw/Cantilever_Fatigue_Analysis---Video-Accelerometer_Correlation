"""
ROC NaN DIAGNOSTIC AND FIX SCRIPT
==================================

This script diagnoses and fixes NaN issues in the ROC (Rate of Change) data
from the cantilever fatigue analysis.

Author: Research-grade implementation
Date: 2026-02-17
"""

import pandas as pd
import numpy as np
import os
import glob

def diagnose_and_fix_roc():
    """Diagnose and fix ROC NaN issues in timeseries CSV files."""
    
    print("="*70)
    print("ROC NaN DIAGNOSTIC AND FIX")
    print("="*70)
    
    # Find all timeseries CSV files
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  NOTE: This script automatically finds all files matching the      │
    # │  pattern "indent_*_timeseries.csv" in the current directory.      │
    # │  Run this from the same folder where those files were generated.  │
    # └─────────────────────────────────────────────────────────────────────┘
    csv_files = glob.glob("indent_*_timeseries.csv")
    
    if not csv_files:
        print("\n⚠ No timeseries CSV files found!")
        print("Please run the video analysis script first.")
        return
    
    print(f"\nFound {len(csv_files)} timeseries files\n")
    
    stats_summary = []
    
    for csv_file in sorted(csv_files):
        # Extract indent number
        indent_id = csv_file.split('_')[1]
        
        # Load data
        df = pd.read_csv(csv_file)
        
        # Diagnose ROC issues
        n_total = len(df)
        n_nan = df['roc'].isna().sum()
        n_inf = np.isinf(df['roc']).sum()
        n_valid = n_total - n_nan - n_inf
        
        roc_std = df['roc'].std()
        roc_mean = df['roc'].mean()
        
        print(f"Indent {indent_id}:")
        print(f"  Total points: {n_total}")
        print(f"  NaN values: {n_nan} ({n_nan/n_total*100:.1f}%)")
        print(f"  Inf values: {n_inf} ({n_inf/n_total*100:.1f}%)")
        print(f"  Valid points: {n_valid} ({n_valid/n_total*100:.1f}%)")
        print(f"  ROC mean: {roc_mean:.6f}")
        print(f"  ROC std: {roc_std:.6f}")
        
        # Determine the issue
        issue = "OK"
        if n_valid == 0:
            issue = "All NaN/Inf"
        elif roc_std == 0 or np.isnan(roc_std):
            issue = "Zero variance"
        elif n_nan > n_total * 0.5:
            issue = "Too many NaN"
        elif n_valid < 100:
            issue = "Too few valid points"
        
        print(f"  Status: {issue}")
        
        # Apply fixes
        if issue != "OK":
            print(f"  Applying fix...")
            
            # Strategy: Recalculate ROC from intensity using central difference
            # avoiding the first and last points
            
            times = df['time_hours'].values
            intensities = df['intensity'].values
            
            # Calculate ROC using numpy gradient (handles boundaries properly)
            roc_fixed = np.gradient(intensities, times)
            
            # Optional: Apply light smoothing to reduce noise
            from scipy.ndimage import gaussian_filter1d
            roc_fixed = gaussian_filter1d(roc_fixed, sigma=1.0)
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │  USER CHANGE REQUIRED (if applicable): Set boundary_times   │
            # │  to the experiment hours where video segments join.         │
            # │                                                             │
            # │  The gradient across a segment boundary is meaningless      │
            # │  (it spans a time gap) so those points are set to NaN.     │
            # │                                                             │
            # │  72h experiment with 3 daily segments: [24.0, 48.0]        │
            # │  48h experiment with 2 daily segments: [24.0]              │
            # │  Single continuous recording: [] (empty list)              │
            # └─────────────────────────────────────────────────────────────┘
            boundary_times = [24.0, 48.0]
            for boundary in boundary_times:
                # Find indices near boundary (within 0.1 hours = 6 minutes)
                boundary_mask = np.abs(times - boundary) < 0.1
                if np.any(boundary_mask):
                    boundary_idx = np.where(boundary_mask)[0]
                    # Set ROC to NaN at boundaries (per user's Q6 answer)
                    roc_fixed[boundary_idx] = np.nan
                    print(f"    Set ROC=NaN at {boundary}h boundary (n={len(boundary_idx)})")
            
            # Update dataframe
            df['roc'] = roc_fixed
            
            # Save fixed version
            output_file = csv_file.replace('.csv', '_fixed.csv')
            df.to_csv(output_file, index=False)
            
            # Re-check
            n_nan_fixed = df['roc'].isna().sum()
            n_valid_fixed = len(df) - n_nan_fixed - np.isinf(df['roc']).sum()
            roc_std_fixed = df['roc'].std()
            
            print(f"    After fix: {n_valid_fixed} valid points, std={roc_std_fixed:.6f}")
            print(f"    Saved to: {output_file}")
        
        stats_summary.append({
            'indent': indent_id,
            'total': n_total,
            'nan': n_nan,
            'inf': n_inf,
            'valid': n_valid,
            'mean': roc_mean,
            'std': roc_std,
            'issue': issue
        })
        
        print()
    
    # Summary report
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    df_summary = pd.DataFrame(stats_summary)
    
    print("\nIssues by category:")
    for issue_type in df_summary['issue'].unique():
        count = (df_summary['issue'] == issue_type).sum()
        print(f"  {issue_type}: {count} indents")
    
    print(f"\nFixed files saved as: indent_*_timeseries_fixed.csv")
    print(f"\nTo use fixed files in correlation analysis:")
    print(f"  1. Backup original files")
    print(f"  2. Rename _fixed.csv files to replace originals")
    print(f"  3. Re-run accelerometer_video_correlation.py")
    
    # Save summary
    df_summary.to_csv('roc_diagnostic_summary.csv', index=False)
    print(f"\nDiagnostic summary saved to: roc_diagnostic_summary.csv")

if __name__ == "__main__":
    diagnose_and_fix_roc()
