import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, 
    mean_absolute_error, accuracy_score, roc_auc_score, 
    mean_squared_error, matthews_corrcoef
)
from scipy.stats import pearsonr, kstest, norm
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# Create output file
output_filename = 'technical_phasenet_analysis.txt'
print(f"Saving technical analysis to: {output_filename}")

# Redirect stdout to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# Open output file
output_file = open(output_filename, 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, output_file)

print("=" * 100)
print("TECHNICAL PHASENET PERFORMANCE ANALYSIS - SEISMIC PICKING QUALITY")
print("=" * 100)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load files
manual_df = pd.read_csv('npz_data_list_detailed.csv', delimiter='\t')
pred_df = pd.read_csv('picks.csv')

print("Manual picks columns:", manual_df.columns.tolist())
print("Predicted picks columns:", pred_df.columns.tolist())

# ============================================================================
# TECHNICAL DATA VALIDATION
# ============================================================================

def extract_station_event(filename):
    parts = filename.replace('.npz', '').split('.')
    if len(parts) >= 5:
        station = f"{parts[0]}.{parts[1]}"
        event_id = parts[4]
        return station, event_id
    return None, None

# Add station and event_id to both dataframes
manual_df[['station', 'event_id']] = manual_df['fname'].apply(
    lambda x: pd.Series(extract_station_event(x))
)

pred_df[['station', 'event_id']] = pred_df['file_name'].apply(
    lambda x: pd.Series(extract_station_event(x))
)

print("🔍 TECHNICAL DATA VALIDATION")
print(f"Manual entries: {len(manual_df)}")
print(f"Predicted picks: {len(pred_df)}")

# Check data quality
print(f"\n📊 DATA QUALITY ASSESSMENT:")
print(f"Manual P picks: {manual_df['p_idx'].notna().sum()}")
print(f"Manual S picks: {manual_df['s_idx'].notna().sum()}")
print(f"PhaseNet P picks: {len(pred_df[pred_df['phase_type'] == 'P'])}")
print(f"PhaseNet S picks: {len(pred_df[pred_df['phase_type'] == 'S'])}")

# ============================================================================
# CREATE TECHNICAL COMPARISON DATASET
# ============================================================================

manual_picks_list = []
for _, row in manual_df.iterrows():
    if not pd.isna(row['p_idx']):
        manual_picks_list.append({
            'station': row['station'], 'event_id': row['event_id'], 'phase_type': 'P',
            'phase_index': row['p_idx'], 'phase_score': 1.0, 'source': 'manual', 
            'file_name': row['fname']
        })
    if not pd.isna(row['s_idx']):
        manual_picks_list.append({
            'station': row['station'], 'event_id': row['event_id'], 'phase_type': 'S', 
            'phase_index': row['s_idx'], 'phase_score': 1.0, 'source': 'manual',
            'file_name': row['fname']
        })

manual_picks_df = pd.DataFrame(manual_picks_list)
pred_picks_df = pred_df.copy()
pred_picks_df['source'] = 'phasenet'

# Merge for technical comparison
comparison_df = pd.merge(
    manual_picks_df, pred_picks_df,
    on=['station', 'event_id', 'phase_type'],
    suffixes=('_manual', '_pred'),
    how='inner'
)

print(f"\n🔬 TECHNICAL COMPARISON DATASET:")
print(f"Matched picks for analysis: {len(comparison_df)}")

if len(comparison_df) > 0:
    # Calculate technical metrics
    sampling_rate = 100  # Hz
    comparison_df['time_diff_samples'] = comparison_df['phase_index_pred'] - comparison_df['phase_index_manual']
    comparison_df['time_diff_seconds'] = comparison_df['time_diff_samples'] / sampling_rate
    comparison_df['time_diff_ms'] = comparison_df['time_diff_seconds'] * 1000
    comparison_df['abs_time_error_ms'] = np.abs(comparison_df['time_diff_ms'])
    
    # Separate P and S picks
    p_comparison = comparison_df[comparison_df['phase_type'] == 'P'].copy()
    s_comparison = comparison_df[comparison_df['phase_type'] == 'S'].copy()

# ============================================================================
# TECHNICAL TIME ACCURACY ANALYSIS
# ============================================================================

def analyze_time_accuracy_technical(comparison_data, phase_name):
    """Comprehensive technical analysis of time accuracy"""
    if len(comparison_data) == 0:
        print(f"No {phase_name} picks for technical analysis")
        return None
    
    time_diffs_ms = comparison_data['time_diff_ms'].values
    abs_errors_ms = np.abs(time_diffs_ms)
    
    print(f"\n{'='*70}")
    print(f"🕒 {phase_name} PHASE - TECHNICAL TIME ACCURACY ANALYSIS")
    print(f"{'='*70}")
    print(f"Analysis sample size: {len(comparison_data)} picks")
    
    # Basic statistical moments
    mean_error = np.mean(time_diffs_ms)
    std_error = np.std(time_diffs_ms)
    skewness = pd.Series(time_diffs_ms).skew()
    kurtosis = pd.Series(time_diffs_ms).kurtosis()
    
    print(f"\n📈 STATISTICAL MOMENTS:")
    print(f"  Mean Error:       {mean_error:8.3f} ms (1st moment - bias)")
    print(f"  Std Deviation:    {std_error:8.3f} ms (2nd moment - precision)")
    print(f"  Skewness:         {skewness:8.3f} (3rd moment - asymmetry)")
    print(f"  Kurtosis:         {kurtosis:8.3f} (4th moment - tail behavior)")
    
    # Error distribution analysis
    print(f"\n📊 ERROR DISTRIBUTION ANALYSIS:")
    
    # Normality test
    if len(time_diffs_ms) > 3:
        ks_stat, ks_p = kstest(time_diffs_ms, 'norm')
        print(f"  Kolmogorov-Smirnov test: D={ks_stat:.4f}, p={ks_p:.4f}")
        if ks_p > 0.05:
            print("  → Errors appear normally distributed (p > 0.05)")
        else:
            print("  → Errors deviate from normal distribution (p ≤ 0.05)")
    
    # Robust statistics (less sensitive to outliers)
    median_error = np.median(time_diffs_ms)
    mad = np.median(np.abs(time_diffs_ms - median_error))  # Median Absolute Deviation
    iqr = np.percentile(time_diffs_ms, 75) - np.percentile(time_diffs_ms, 25)
    
    print(f"\n🛡️  ROBUST STATISTICS (Outlier-resistant):")
    print(f"  Median Error:     {median_error:8.3f} ms")
    print(f"  MAD:              {mad:8.3f} ms (Median Absolute Deviation)")
    print(f"  IQR:              {iqr:8.3f} ms (Interquartile Range)")
    
    # Error metrics
    mae = np.mean(abs_errors_ms)
    rmse = np.sqrt(np.mean(time_diffs_ms**2))
    
    print(f"\n🎯 ERROR METRICS:")
    print(f"  MAE:              {mae:8.3f} ms (Mean Absolute Error)")
    print(f"  RMSE:             {rmse:8.3f} ms (Root Mean Square Error)")
    if mae > 0:
        print(f"  RMSE/MAE ratio:   {rmse/mae:8.3f} (>1 indicates outlier influence)")
    
    # Precision analysis
    print(f"\n🎯 PRECISION ANALYSIS:")
    for threshold_ms in [5, 10, 20, 50]:
        within_threshold = np.sum(abs_errors_ms <= threshold_ms)
        percentage = (within_threshold / len(abs_errors_ms)) * 100
        print(f"  Within ±{threshold_ms:2d} ms: {within_threshold:3d}/{len(abs_errors_ms):3d} ({percentage:5.1f}%)")
    
    # Outlier analysis
    q1 = np.percentile(abs_errors_ms, 25)
    q3 = np.percentile(abs_errors_ms, 75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    outliers = np.sum(abs_errors_ms > outlier_threshold)
    
    print(f"\n🚨 OUTLIER ANALYSIS:")
    print(f"  Q1 (25%):         {q1:8.1f} ms")
    print(f"  Q3 (75%):         {q3:8.1f} ms")
    print(f"  Outlier threshold:{outlier_threshold:8.1f} ms (Q3 + 1.5×IQR)")
    print(f"  Outliers:         {outliers:3d} picks ({outliers/len(abs_errors_ms)*100:.1f}%)")
    
    # Confidence interval analysis
    confidence_95 = np.percentile(abs_errors_ms, 95)
    confidence_99 = np.percentile(abs_errors_ms, 99)
    
    print(f"\n📏 CONFIDENCE INTERVALS:")
    print(f"  95% of errors ≤   {confidence_95:8.1f} ms")
    print(f"  99% of errors ≤   {confidence_99:8.1f} ms")
    
    stats = {
        'n_picks': len(comparison_data),
        'mean_error_ms': mean_error,
        'std_error_ms': std_error,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'median_error_ms': median_error,
        'mad_ms': mad,
        'mae_ms': mae,
        'rmse_ms': rmse,
        'outlier_count': outliers,
        'confidence_95_ms': confidence_95,
        'confidence_99_ms': confidence_99,
        'time_diffs_ms': time_diffs_ms,
        'abs_errors_ms': abs_errors_ms
    }
    
    return stats

# Perform technical analysis
p_technical = analyze_time_accuracy_technical(p_comparison, "P") if 'p_comparison' in locals() and len(p_comparison) > 0 else None
s_technical = analyze_time_accuracy_technical(s_comparison, "S") if 's_comparison' in locals() and len(s_comparison) > 0 else None

# ============================================================================
# PHASE DETECTION RELIABILITY ANALYSIS - FIXED VERSION
# ============================================================================

def analyze_detection_reliability(manual_df, pred_df, tolerance_samples=50):
    """Technical analysis of phase detection reliability"""
    detection_data = []
    
    for _, manual_row in manual_df.iterrows():
        station = manual_row['station']
        event_id = manual_row['event_id']
        
        preds_for_event = pred_df[
            (pred_df['station'] == station) & 
            (pred_df['event_id'] == event_id)
        ]
        
        # P pick analysis
        p_manual_idx = manual_row['p_idx']
        p_detected = 0
        p_confidence = 0.0
        p_time_error = np.nan
        
        if not pd.isna(p_manual_idx):
            p_preds = preds_for_event[preds_for_event['phase_type'] == 'P']
            for _, pred_row in p_preds.iterrows():
                time_error_samples = abs(pred_row['phase_index'] - p_manual_idx)
                if time_error_samples <= tolerance_samples:
                    p_detected = 1
                    p_confidence = pred_row['phase_score']
                    p_time_error = time_error_samples / 100.0 * 1000  # Convert to ms
                    break
        
        # S pick analysis
        s_manual_idx = manual_row['s_idx']
        s_detected = 0
        s_confidence = 0.0
        s_time_error = np.nan
        
        if not pd.isna(s_manual_idx):
            s_preds = preds_for_event[preds_for_event['phase_type'] == 'S']
            for _, pred_row in s_preds.iterrows():
                time_error_samples = abs(pred_row['phase_index'] - s_manual_idx)
                if time_error_samples <= tolerance_samples:
                    s_detected = 1
                    s_confidence = pred_row['phase_score']
                    s_time_error = time_error_samples / 100.0 * 1000  # Convert to ms
                    break
        
        detection_data.append({
            'station': station, 'event_id': event_id,
            'p_exists': 1 if not pd.isna(p_manual_idx) else 0,
            'p_detected': p_detected, 'p_confidence': p_confidence, 'p_time_error_ms': p_time_error,
            's_exists': 1 if not pd.isna(s_manual_idx) else 0, 
            's_detected': s_detected, 's_confidence': s_confidence, 's_time_error_ms': s_time_error
        })
    
    return pd.DataFrame(detection_data)

detection_df = analyze_detection_reliability(manual_df, pred_df)
print(f"\n🔍 DETECTION RELIABILITY DATASET:")
print(f"Station-events analyzed: {len(detection_df)}")

def analyze_phase_detection_technical(phase_type, detection_df):
    """Technical analysis of phase detection performance"""
    if phase_type == 'P':
        true_exists = detection_df['p_exists']
        pred_detected = detection_df['p_detected']
        pred_confidence = detection_df['p_confidence']
        time_errors = detection_df['p_time_error_ms']
    else:
        true_exists = detection_df['s_exists']
        pred_detected = detection_df['s_detected']
        pred_confidence = detection_df['s_confidence']
        time_errors = detection_df['s_time_error_ms']
    
    # Only evaluate where manual pick exists
    eval_mask = true_exists == 1
    y_true_exists = true_exists[eval_mask]
    y_detected = pred_detected[eval_mask]
    y_confidence = pred_confidence[eval_mask]
    valid_time_errors = time_errors[eval_mask & (pred_detected == 1)]
    
    if len(y_true_exists) == 0:
        print(f"No {phase_type} picks available for detection analysis")
        return None
    
    print(f"\n{'='*70}")
    print(f"🎯 {phase_type} PHASE - DETECTION RELIABILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Evaluation samples: {len(y_true_exists)}")
    
    # Detection metrics
    accuracy = accuracy_score(y_detected, (y_confidence > 0.5).astype(int))
    precision = precision_score(y_detected, (y_confidence > 0.5).astype(int), zero_division=0)
    recall = recall_score(y_detected, (y_confidence > 0.5).astype(int), zero_division=0)
    f1 = f1_score(y_detected, (y_confidence > 0.5).astype(int), zero_division=0)
    
    print(f"\n📊 DETECTION PERFORMANCE:")
    print(f"  Accuracy:         {accuracy:8.3f}")
    print(f"  Precision:        {precision:8.3f} (True Positives / All Positives)")
    print(f"  Recall:           {recall:8.3f} (True Positives / Actual Positives)")
    print(f"  F1-Score:         {f1:8.3f} (Harmonic mean of Precision & Recall)")
    
    # Confidence score analysis - FIXED: Handle empty arrays
    valid_confidences = y_confidence[y_confidence > 0]  # Only non-zero confidences
    
    print(f"\n🎚️  CONFIDENCE SCORE ANALYSIS:")
    if len(valid_confidences) > 0:
        print(f"  Mean confidence:  {np.mean(valid_confidences):8.3f}")
        print(f"  Std confidence:   {np.std(valid_confidences):8.3f}")
        print(f"  Min confidence:   {np.min(valid_confidences):8.3f}")
        print(f"  Max confidence:   {np.max(valid_confidences):8.3f}")
    else:
        print(f"  No confidence scores available for analysis")
    
    # Detection timing for successful detections
    valid_time_errors_clean = valid_time_errors[~np.isnan(valid_time_errors)]
    if len(valid_time_errors_clean) > 0:
        print(f"\n⏱️  DETECTION TIMING (Successful detections):")
        print(f"  Mean time error:  {np.mean(valid_time_errors_clean):8.1f} ms")
        print(f"  Std time error:   {np.std(valid_time_errors_clean):8.1f} ms")
        print(f"  Max time error:   {np.max(valid_time_errors_clean):8.1f} ms")
    
    # Detection rate analysis
    true_positives = np.sum(y_detected == 1)
    false_negatives = np.sum(y_detected == 0)
    detection_rate = true_positives / len(y_detected) if len(y_detected) > 0 else 0
    
    print(f"\n📈 DETECTION RATES:")
    print(f"  True Positives:   {true_positives:4d}")
    print(f"  False Negatives:  {false_negatives:4d}")
    print(f"  Detection Rate:   {detection_rate:8.3f}")
    
    # Confidence threshold analysis - FIXED: Handle edge cases
    print(f"\n⚖️  CONFIDENCE THRESHOLD ANALYSIS:")
    if len(y_confidence) > 0:
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            # Count picks with confidence >= threshold AND that were actually detected
            high_confidence_detections = np.sum((y_confidence >= threshold) & (y_detected == 1))
            total_detections = np.sum(y_detected == 1)
            
            if total_detections > 0:
                percentage = (high_confidence_detections / total_detections) * 100
                print(f"  ≥{threshold}: {high_confidence_detections:3d}/{total_detections:3d} ({percentage:5.1f}%)")
            else:
                print(f"  ≥{threshold}:   0/  0 (  0.0%) - No detections")
    else:
        print("  No confidence scores available for threshold analysis")
    
    metrics = {
        'n_samples': len(y_true_exists),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'detection_rate': detection_rate,
        'mean_confidence': np.mean(valid_confidences) if len(valid_confidences) > 0 else 0,
        'std_confidence': np.std(valid_confidences) if len(valid_confidences) > 0 else 0
    }
    
    return metrics

p_detection_tech = analyze_phase_detection_technical('P', detection_df)
s_detection_tech = analyze_phase_detection_technical('S', detection_df)

# ============================================================================
# TECHNICAL VISUALIZATION - FIXED VERSION
# ============================================================================

def create_technical_plots(p_comparison, s_comparison, p_technical, s_technical):
    """Create technical analysis plots with error handling"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Technical PhaseNet Performance Analysis - Seismic Picking Quality', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Error distribution with statistical annotations
    if p_technical and len(p_comparison) > 0:
        p_errors = p_comparison['time_diff_ms'].values
        axes[0, 0].hist(p_errors, bins=50, alpha=0.7, color='blue', density=True, edgecolor='black')
        
        # Add statistical lines
        axes[0, 0].axvline(p_technical['mean_error_ms'], color='red', linestyle='--', 
                          label=f"Mean: {p_technical['mean_error_ms']:.1f} ms")
        axes[0, 0].axvline(p_technical['median_error_ms'], color='green', linestyle='--',
                          label=f"Median: {p_technical['median_error_ms']:.1f} ms")
        axes[0, 0].axvline(0, color='black', linestyle='-', alpha=0.3)
        
        axes[0, 0].set_xlabel('Time Error (ms)')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('P-Phase: Error Distribution & Statistics')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No P-phase data\navailable', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('P-Phase: No Data')
    
    # Plot 2: S-phase error distribution
    if s_technical and len(s_comparison) > 0:
        s_errors = s_comparison['time_diff_ms'].values
        axes[0, 1].hist(s_errors, bins=50, alpha=0.7, color='red', density=True, edgecolor='black')
        
        axes[0, 1].axvline(s_technical['mean_error_ms'], color='blue', linestyle='--',
                          label=f"Mean: {s_technical['mean_error_ms']:.1f} ms")
        axes[0, 1].axvline(s_technical['median_error_ms'], color='green', linestyle='--',
                          label=f"Median: {s_technical['median_error_ms']:.1f} ms")
        axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.3)
        
        axes[0, 1].set_xlabel('Time Error (ms)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('S-Phase: Error Distribution & Statistics')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No S-phase data\navailable', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('S-Phase: No Data')
    
    # Plot 3: Cumulative error distribution with confidence intervals
    if p_technical or s_technical:
        if p_technical and len(p_comparison) > 0:
            sorted_p_errors = np.sort(np.abs(p_comparison['time_diff_ms']))
            y_p = np.arange(len(sorted_p_errors)) / float(len(sorted_p_errors))
            axes[0, 2].plot(sorted_p_errors, y_p, label=f'P-phase (n={len(p_comparison)})', 
                           color='blue', linewidth=2)
        
        if s_technical and len(s_comparison) > 0:
            sorted_s_errors = np.sort(np.abs(s_comparison['time_diff_ms']))
            y_s = np.arange(len(sorted_s_errors)) / float(len(sorted_s_errors))
            axes[0, 2].plot(sorted_s_errors, y_s, label=f'S-phase (n={len(s_comparison)})', 
                           color='red', linewidth=2)
        
        # Add confidence interval lines
        if p_technical:
            axes[0, 2].axvline(p_technical['confidence_95_ms'], color='blue', linestyle=':', 
                              alpha=0.7, label=f'P 95%: {p_technical["confidence_95_ms"]:.1f} ms')
        if s_technical:
            axes[0, 2].axvline(s_technical['confidence_95_ms'], color='red', linestyle=':', 
                              alpha=0.7, label=f'S 95%: {s_technical["confidence_95_ms"]:.1f} ms')
        
        axes[0, 2].set_xlabel('Absolute Time Error (ms)')
        axes[0, 2].set_ylabel('Cumulative Fraction')
        axes[0, 2].set_title('Cumulative Error Distribution with Confidence Intervals')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'No data for\ncumulative analysis', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Cumulative Error: No Data')
    
    # Plot 4: Error vs Confidence with statistical analysis
    if len(p_comparison) > 0 and len(s_comparison) > 0:
        # Calculate correlation - handle cases with no variation
        try:
            p_corr = np.corrcoef(p_comparison['phase_score_pred'], 
                                np.abs(p_comparison['time_diff_ms']))[0,1]
        except:
            p_corr = 0
            
        try:
            s_corr = np.corrcoef(s_comparison['phase_score_pred'], 
                                np.abs(s_comparison['time_diff_ms']))[0,1]
        except:
            s_corr = 0
        
        axes[1, 0].scatter(p_comparison['phase_score_pred'], np.abs(p_comparison['time_diff_ms']), 
                          alpha=0.6, color='blue', label=f'P-phase (r={p_corr:.3f})', s=30)
        axes[1, 0].scatter(s_comparison['phase_score_pred'], np.abs(s_comparison['time_diff_ms']), 
                          alpha=0.6, color='red', label=f'S-phase (r={s_corr:.3f})', s=30)
        axes[1, 0].set_xlabel('PhaseNet Confidence Score')
        axes[1, 0].set_ylabel('Absolute Time Error (ms)')
        axes[1, 0].set_title('Error vs Confidence (Correlation Analysis)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No data for\ncorrelation analysis', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Error vs Confidence: No Data')
    
    # Plot 5: Statistical moments comparison - FIXED VERSION
    if p_technical and s_technical:
        moments = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis']
        p_moments = [p_technical['mean_error_ms'], p_technical['std_error_ms'], 
                    p_technical['skewness'], p_technical['kurtosis']]
        s_moments = [s_technical['mean_error_ms'], s_technical['std_error_ms'], 
                    s_technical['skewness'], s_technical['kurtosis']]
        
        # FIX: Convert to numpy arrays for proper absolute value calculation
        p_moments_arr = np.array(p_moments)
        s_moments_arr = np.array(s_moments)
        
        # Normalize for comparison - handle division by zero
        p_max = np.max(np.abs(p_moments_arr))
        s_max = np.max(np.abs(s_moments_arr))
        
        p_norm = np.abs(p_moments_arr) / p_max if p_max > 0 else np.zeros_like(p_moments_arr)
        s_norm = np.abs(s_moments_arr) / s_max if s_max > 0 else np.zeros_like(s_moments_arr)
        
        x_pos = np.arange(len(moments))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, p_norm, width, label='P-phase', alpha=0.7, color='blue')
        axes[1, 1].bar(x_pos + width/2, s_norm, width, label='S-phase', alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Statistical Moments')
        axes[1, 1].set_ylabel('Normalized Magnitude')
        axes[1, 1].set_title('Statistical Moments Comparison (Normalized)')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(moments)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No data for\nstatistical comparison', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Statistical Moments: No Data')
    
    # Plot 6: Technical summary
    axes[1, 2].axis('off')
    if p_technical and s_technical:
        summary_text = "TECHNICAL SUMMARY:\n\n"
        summary_text += f"P-Phase (n={p_technical['n_picks']}):\n"
        summary_text += f"• Bias: {p_technical['mean_error_ms']:+.1f} ms\n"
        summary_text += f"• Precision: {p_technical['std_error_ms']:.1f} ms\n"
        summary_text += f"• 95% CI: ≤{p_technical['confidence_95_ms']:.1f} ms\n"
        summary_text += f"• Outliers: {p_technical['outlier_count']} ({p_technical['outlier_count']/p_technical['n_picks']*100:.1f}%)\n\n"
        summary_text += f"S-Phase (n={s_technical['n_picks']}):\n"
        summary_text += f"• Bias: {s_technical['mean_error_ms']:+.1f} ms\n"
        summary_text += f"• Precision: {s_technical['std_error_ms']:.1f} ms\n"
        summary_text += f"• 95% CI: ≤{s_technical['confidence_95_ms']:.1f} ms\n"
        summary_text += f"• Outliers: {s_technical['outlier_count']} ({s_technical['outlier_count']/s_technical['n_picks']*100:.1f}%)"
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    else:
        axes[1, 2].text(0.5, 0.5, 'No technical summary\navailable', 
                       ha='center', va='center', transform=axes[1, 2].transAxes,
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('technical_phasenet_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate technical plots
print(f"\n📊 GENERATING TECHNICAL ANALYSIS PLOTS...")
create_technical_plots(
    p_comparison if 'p_comparison' in locals() else None,
    s_comparison if 's_comparison' in locals() else None,
    p_technical, s_technical
)

# ============================================================================
# TECHNICAL CONCLUSIONS
# ============================================================================

print("\n" + "="*80)
print("🔬 TECHNICAL CONCLUSIONS - PHASENET PERFORMANCE ASSESSMENT")
print("="*80)

print(f"\n📁 TECHNICAL OUTPUT FILES:")
print(f"1. {output_filename} - Comprehensive technical analysis")
print(f"2. technical_phasenet_analysis.png - Technical visualization")

print(f"\n🎯 KEY TECHNICAL INSIGHTS:")
if p_technical:
    print(f"• P-PHASE: Bias={p_technical['mean_error_ms']:+.1f}ms, Precision={p_technical['std_error_ms']:.1f}ms")
    print(f"  → 95% of P-pick errors ≤ {p_technical['confidence_95_ms']:.1f} ms")
    print(f"  → Distribution skewness: {p_technical['skewness']:.3f}")
if s_technical:
    print(f"• S-PHASE: Bias={s_technical['mean_error_ms']:+.1f}ms, Precision={s_technical['std_error_ms']:.1f}ms")
    print(f"  → 95% of S-pick errors ≤ {s_technical['confidence_95_ms']:.1f} ms")
    print(f"  → Distribution skewness: {s_technical['skewness']:.3f}")

print(f"\n✅ TECHNICAL ANALYSIS COMPLETE!")
print("Comprehensive seismic picking quality assessment finished.")

# Close output file and restore stdout
sys.stdout = original_stdout
output_file.close()

print(f"\n📄 Technical analysis saved to: {output_filename}")
print("🖼️  Technical plots saved as: technical_phasenet_analysis.png")