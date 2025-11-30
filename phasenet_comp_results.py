#!/usr/bin/env python3
"""
PhaseNet Comprehensive Results Analysis - FIXED FOR YOUR FILENAME FORMAT
Handles: STATION_STATION_DATETIME.npz format
"""

import pandas as pd
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted_picks', required=True)
    parser.add_argument('--manual_picks', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sampling_rate', type=float, default=100)
    parser.add_argument('--tolerance_ms', type=float, default=500)
    return parser.parse_args()

def extract_station_datetime(filename):
    """
    Extract station and datetime from filename
    Format: DDWD_DDWD_2021009071642.npz -> station=DDWD, datetime=2021009071642
    """
    if pd.isna(filename):
        return None, None
    
    # Remove .npz extension
    basename = filename.replace('.npz', '')
    
    # Split by underscore
    parts = basename.split('_')
    
    if len(parts) >= 3:
        # Format: STATION_STATION_DATETIME
        station = parts[0]  # First STATION
        datetime_str = parts[2]  # DATETIME part
        return station, datetime_str
    elif len(parts) == 2:
        # Format: STATION_DATETIME
        station = parts[0]
        datetime_str = parts[1]
        return station, datetime_str
    else:
        # Fallback
        return basename, basename

def load_manual_picks(manual_file):
    """Load manual picks and create lookup key"""
    print(f"\nLoading manual picks: {manual_file}")
    
    df = pd.read_csv(manual_file, sep='\t')
    print(f"  Loaded {len(df)} manual records")
    
    # Extract station and datetime from fname
    df[['station', 'datetime']] = df['fname'].apply(
        lambda x: pd.Series(extract_station_datetime(x))
    )
    
    # Create unique key: station + datetime
    df['key'] = df['station'] + '|' + df['datetime']
    
    # Count picks
    n_p = df['p_idx'].notna().sum()
    n_s = df['s_idx'].notna().sum()
    
    print(f"  Manual P picks: {n_p}")
    print(f"  Manual S picks: {n_s}")
    print(f"  Unique station-events: {df['key'].nunique()}")
    print(f"  Sample keys: {list(df['key'].head(3))}")
    
    return df

def load_predicted_picks(pred_file, chunk_size=10000):
    """Load predicted picks in chunks"""
    print(f"\nLoading predicted picks: {pred_file}")
    
    # Count total
    with open(pred_file, 'r') as f:
        total = sum(1 for _ in f) - 1
    print(f"  Total predicted picks: {total}")
    
    # Load in chunks
    chunks = []
    for i, chunk in enumerate(pd.read_csv(pred_file, chunksize=chunk_size)):
        # Extract station and datetime
        chunk[['station', 'datetime']] = chunk['file_name'].apply(
            lambda x: pd.Series(extract_station_datetime(x))
        )
        chunk['key'] = chunk['station'] + '|' + chunk['datetime']
        
        # Keep only needed columns
        chunk = chunk[['key', 'station', 'datetime', 'phase_type', 
                      'phase_index', 'phase_score']].copy()
        chunks.append(chunk)
        
        if (i + 1) % 50 == 0:
            print(f"    Processed {(i+1) * chunk_size} picks...")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df)} predicted picks")
    print(f"  Unique station-events: {df['key'].nunique()}")
    print(f"  Sample keys: {list(df['key'].head(3))}")
    
    return df

def match_picks(manual_df, pred_df, sampling_rate):
    """Match manual and predicted picks by station-event"""
    print("\n" + "="*80)
    print("MATCHING PICKS")
    print("="*80)
    
    # Find common keys
    manual_keys = set(manual_df['key'].unique())
    pred_keys = set(pred_df['key'].unique())
    common_keys = manual_keys & pred_keys
    
    print(f"  Manual station-events: {len(manual_keys)}")
    print(f"  Predicted station-events: {len(pred_keys)}")
    print(f"  Common station-events: {len(common_keys)}")
    
    if len(common_keys) == 0:
        print("\n⚠️  WARNING: No common station-events found!")
        print(f"  Sample manual keys: {list(manual_keys)[:3]}")
        print(f"  Sample pred keys: {list(pred_keys)[:3]}")
        return None, None
    
    # Match picks
    p_diffs = []
    s_diffs = []
    
    for i, key in enumerate(common_keys):
        # Get manual pick for this station-event
        manual_event = manual_df[manual_df['key'] == key].iloc[0]
        
        # Get predicted picks for this station-event
        pred_event = pred_df[pred_df['key'] == key]
        
        # Match P-phase (find closest)
        if pd.notna(manual_event['p_idx']):
            pred_p = pred_event[pred_event['phase_type'] == 'P']
            if len(pred_p) > 0:
                manual_idx = manual_event['p_idx']
                # Find closest predicted pick
                diffs = (pred_p['phase_index'] - manual_idx).abs()
                closest_idx = diffs.idxmin()
                closest_pred_idx = pred_p.loc[closest_idx, 'phase_index']
                
                # Calculate error in ms
                diff_ms = (closest_pred_idx - manual_idx) / sampling_rate * 1000
                p_diffs.append(diff_ms)
        
        # Match S-phase (find closest)
        if pd.notna(manual_event['s_idx']):
            pred_s = pred_event[pred_event['phase_type'] == 'S']
            if len(pred_s) > 0:
                manual_idx = manual_event['s_idx']
                diffs = (pred_s['phase_index'] - manual_idx).abs()
                closest_idx = diffs.idxmin()
                closest_pred_idx = pred_s.loc[closest_idx, 'phase_index']
                
                diff_ms = (closest_pred_idx - manual_idx) / sampling_rate * 1000
                s_diffs.append(diff_ms)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(common_keys)} station-events...")
    
    print(f"\n  Matched P picks: {len(p_diffs)}")
    print(f"  Matched S picks: {len(s_diffs)}")
    
    # Calculate metrics
    p_metrics = None
    s_metrics = None
    
    if len(p_diffs) > 0:
        p_arr = np.array(p_diffs)
        p_metrics = {
            'n_picks': len(p_diffs),
            'mae_ms': np.mean(np.abs(p_arr)),
            'rmse_ms': np.sqrt(np.mean(p_arr**2)),
            'mean_error_ms': np.mean(p_arr),
            'std_error_ms': np.std(p_arr),
            'median_error_ms': np.median(p_arr),
            'min_error_ms': np.min(p_arr),
            'max_error_ms': np.max(p_arr)
        }
    
    if len(s_diffs) > 0:
        s_arr = np.array(s_diffs)
        s_metrics = {
            'n_picks': len(s_diffs),
            'mae_ms': np.mean(np.abs(s_arr)),
            'rmse_ms': np.sqrt(np.mean(s_arr**2)),
            'mean_error_ms': np.mean(s_arr),
            'std_error_ms': np.std(s_arr),
            'median_error_ms': np.median(s_arr),
            'min_error_ms': np.min(s_arr),
            'max_error_ms': np.max(s_arr)
        }
    
    return p_metrics, s_metrics

def save_results(output_file, p_metrics, s_metrics):
    """Save results to file and print to console"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        def write(text):
            print(text)
            f.write(text + '\n')
        
        write("=" * 100)
        write("PHASENET COMPREHENSIVE PERFORMANCE EVALUATION")
        write("=" * 100)
        write("")
        
        if p_metrics:
            write("P-PHASE RESULTS:")
            write(f"  Matched picks:   {p_metrics['n_picks']}")
            write(f"  MAE:             {p_metrics['mae_ms']:.3f} ms")
            write(f"  RMSE:            {p_metrics['rmse_ms']:.3f} ms")
            write(f"  Mean error:      {p_metrics['mean_error_ms']:.3f} ms")
            write(f"  Std deviation:   {p_metrics['std_error_ms']:.3f} ms")
            write(f"  Median error:    {p_metrics['median_error_ms']:.3f} ms")
            write(f"  Min error:       {p_metrics['min_error_ms']:.3f} ms")
            write(f"  Max error:       {p_metrics['max_error_ms']:.3f} ms")
            write("")
        else:
            write("P-PHASE RESULTS: No matches found")
            write("")
        
        if s_metrics:
            write("S-PHASE RESULTS:")
            write(f"  Matched picks:   {s_metrics['n_picks']}")
            write(f"  MAE:             {s_metrics['mae_ms']:.3f} ms")
            write(f"  RMSE:            {s_metrics['rmse_ms']:.3f} ms")
            write(f"  Mean error:      {s_metrics['mean_error_ms']:.3f} ms")
            write(f"  Std deviation:   {s_metrics['std_error_ms']:.3f} ms")
            write(f"  Median error:    {s_metrics['median_error_ms']:.3f} ms")
            write(f"  Min error:       {s_metrics['min_error_ms']:.3f} ms")
            write(f"  Max error:       {s_metrics['max_error_ms']:.3f} ms")
            write("")
        else:
            write("S-PHASE RESULTS: No matches found")
            write("")
        
        write("=" * 100)

def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("PHASENET EVALUATION - FIXED FOR YOUR DATA FORMAT")
    print("="*80)
    
    # Load data
    manual_df = load_manual_picks(args.manual_picks)
    pred_df = load_predicted_picks(args.predicted_picks)
    
    # Match and calculate
    p_metrics, s_metrics = match_picks(manual_df, pred_df, args.sampling_rate)
    
    # Save results
    save_results(args.output_dir, p_metrics, s_metrics)
    
    print(f"\n✅ Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()