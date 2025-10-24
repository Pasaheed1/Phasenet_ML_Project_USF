import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, 
    mean_absolute_error, accuracy_score, roc_auc_score, 
    mean_squared_error, matthews_corrcoef
)
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# Create output file
output_filename = 'phasenet_comprehensive_results.txt'
print(f"Saving results to: {output_filename}")

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
print("PHASENET COMPREHENSIVE PERFORMANCE EVALUATION")
print("=" * 100)
print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Load files
manual_df = pd.read_csv('npz.csv', delimiter='\t')
pred_df = pd.read_csv('picks.csv')

print("Manual picks columns:", manual_df.columns.tolist())
print("Predicted picks columns:", pred_df.columns.tolist())

# ============================================================================
# PAPER'S RESULTS - TRAVEL TIME RESIDUALS (NOT PICKING ERRORS)
# ============================================================================

paper_results = {
    'P': {
        'travel_time_residual_mean_ms': 2.068,    # Mean travel time residual in ms
        'travel_time_residual_std_ms': 51.530,    # Std dev of travel time residuals in ms
        'precision': 0.939,       # Pick detection precision
        'recall': 0.857,          # Pick detection recall  
        'f1_score': 0.896,        # Pick detection F1-score
        'dataset_size': 3001     # 3-component traces used
    },
    'S': {
        'travel_time_residual_mean_ms': 3.311,    # Mean travel time residual in ms  
        'travel_time_residual_std_ms': 82.858,    # Std dev of travel time residuals in ms
        'precision': 0.853,       # Pick detection precision
        'recall': 0.755,          # Pick detection recall
        'f1_score': 0.801,       # Pick detection F1-score
        'dataset_size': 3001     # 3-component traces used
    }
}

# ============================================================================
# DATA PREPROCESSING
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

# Convert to proper timestamps
manual_df['p_timestamp'] = pd.to_datetime(manual_df['p_time']).astype('int64') // 10**9
manual_df['s_timestamp'] = pd.to_datetime(manual_df['s_time']).astype('int64') // 10**9
pred_df['phase_timestamp'] = pd.to_datetime(pred_df['phase_time']).astype('int64') // 10**9

print(f"Manual entries: {len(manual_df)}")
print(f"Predicted picks: {len(pred_df)}")

# ============================================================================
# PICKING ACCURACY METRICS (YOUR ANALYSIS)
# ============================================================================

# Create manual picks dataframe
manual_picks_list = []
for _, row in manual_df.iterrows():
    if not pd.isna(row['p_idx']) and not pd.isna(row['p_timestamp']):
        manual_picks_list.append({
            'station': row['station'], 'event_id': row['event_id'], 'phase_type': 'P',
            'phase_index': row['p_idx'], 'phase_timestamp': row['p_timestamp'],
            'phase_score': 1.0, 'source': 'manual', 'file_name': row['fname']
        })
    if not pd.isna(row['s_idx']) and not pd.isna(row['s_timestamp']):
        manual_picks_list.append({
            'station': row['station'], 'event_id': row['event_id'], 'phase_type': 'S', 
            'phase_index': row['s_idx'], 'phase_timestamp': row['s_timestamp'],
            'phase_score': 1.0, 'source': 'manual', 'file_name': row['fname']
        })

manual_picks_df = pd.DataFrame(manual_picks_list)
pred_picks_df = pred_df.copy()
pred_picks_df['source'] = 'phasenet'

print(f"\nManual P picks: {len(manual_picks_df[manual_picks_df['phase_type'] == 'P'])}")
print(f"Manual S picks: {len(manual_picks_df[manual_picks_df['phase_type'] == 'S'])}")

# Merge for direct picking accuracy comparison
comparison_df = pd.merge(
    manual_picks_df, pred_picks_df,
    on=['station', 'event_id', 'phase_type'],
    suffixes=('_manual', '_pred'),
    how='inner'
)

print(f"Matched picks for direct comparison: {len(comparison_df)}")

if len(comparison_df) > 0:
    # Calculate picking errors in seconds and milliseconds
    comparison_df['time_diff_seconds'] = (
        comparison_df['phase_timestamp_pred'] - comparison_df['phase_timestamp_manual']
    )
    comparison_df['time_diff_ms'] = comparison_df['time_diff_seconds'] * 1000  # Convert to ms
    
    p_comparison = comparison_df[comparison_df['phase_type'] == 'P'].copy()
    s_comparison = comparison_df[comparison_df['phase_type'] == 'S'].copy()

# ============================================================================
# HISTOGRAM PLOTS FUNCTION
# ============================================================================

def create_histogram_plots(p_comparison, s_comparison, p_picking_accuracy, s_picking_accuracy):
    """Create comprehensive histogram plots for P and S phase picking errors"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PhaseNet Picking Error Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: P-phase histogram
    if p_picking_accuracy and len(p_comparison) > 0:
        p_errors_ms = p_comparison['time_diff_ms'].values
        axes[0, 0].hist(p_errors_ms, bins=50, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
        axes[0, 0].axvline(p_picking_accuracy['mean_picking_error_ms'], color='red', linestyle='--', 
                          linewidth=2, label=f"Mean: {p_picking_accuracy['mean_picking_error_ms']:.2f} ms")
        axes[0, 0].axvline(p_picking_accuracy['mean_picking_error_ms'] + p_picking_accuracy['std_picking_error_ms'], 
                          color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f"±1σ: {p_picking_accuracy['std_picking_error_ms']:.2f} ms")
        axes[0, 0].axvline(p_picking_accuracy['mean_picking_error_ms'] - p_picking_accuracy['std_picking_error_ms'], 
                          color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_xlabel('Picking Error (ms)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title(f'P-Phase Picking Errors\n(n={p_picking_accuracy["n_picks"]} picks)', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f'Mean: {p_picking_accuracy["mean_picking_error_ms"]:.2f} ms\nStd: {p_picking_accuracy["std_picking_error_ms"]:.2f} ms\nMAE: {p_picking_accuracy["mae_ms"]:.2f} ms\nRMSE: {p_picking_accuracy["rmse_ms"]:.2f} ms'
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    # Plot 2: S-phase histogram
    if s_picking_accuracy and len(s_comparison) > 0:
        s_errors_ms = s_comparison['time_diff_ms'].values
        axes[0, 1].hist(s_errors_ms, bins=50, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
        axes[0, 1].axvline(s_picking_accuracy['mean_picking_error_ms'], color='blue', linestyle='--', 
                          linewidth=2, label=f"Mean: {s_picking_accuracy['mean_picking_error_ms']:.2f} ms")
        axes[0, 1].axvline(s_picking_accuracy['mean_picking_error_ms'] + s_picking_accuracy['std_picking_error_ms'], 
                          color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f"±1σ: {s_picking_accuracy['std_picking_error_ms']:.2f} ms")
        axes[0, 1].axvline(s_picking_accuracy['mean_picking_error_ms'] - s_picking_accuracy['std_picking_error_ms'], 
                          color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        axes[0, 1].set_xlabel('Picking Error (ms)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title(f'S-Phase Picking Errors\n(n={s_picking_accuracy["n_picks"]} picks)', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f'Mean: {s_picking_accuracy["mean_picking_error_ms"]:.2f} ms\nStd: {s_picking_accuracy["std_picking_error_ms"]:.2f} ms\nMAE: {s_picking_accuracy["mae_ms"]:.2f} ms\nRMSE: {s_picking_accuracy["rmse_ms"]:.2f} ms'
        axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    # Plot 3: Combined histogram for comparison
    if p_picking_accuracy and s_picking_accuracy and len(p_comparison) > 0 and len(s_comparison) > 0:
        p_errors_ms = p_comparison['time_diff_ms'].values
        s_errors_ms = s_comparison['time_diff_ms'].values
        
        # Determine common x-axis limits
        all_errors = np.concatenate([p_errors_ms, s_errors_ms])
        x_limit = np.percentile(np.abs(all_errors), 95) * 1.2  # Use 95th percentile for x-axis
        
        axes[1, 0].hist(p_errors_ms, bins=50, alpha=0.6, color='blue', label=f'P-phase (n={len(p_errors_ms)})', density=True)
        axes[1, 0].hist(s_errors_ms, bins=50, alpha=0.6, color='red', label=f'S-phase (n={len(s_errors_ms)})', density=True)
        axes[1, 0].set_xlabel('Picking Error (ms)', fontsize=12)
        axes[1, 0].set_ylabel('Density', fontsize=12)
        axes[1, 0].set_title('P vs S Phase Error Distribution (Normalized)', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(-x_limit, x_limit)
    
    # Plot 4: Error vs Confidence Score
    if len(p_comparison) > 0 and len(s_comparison) > 0:
        axes[1, 1].scatter(p_comparison['phase_score_pred'], p_comparison['time_diff_ms'], 
                          alpha=0.6, color='blue', label='P-phase', s=30)
        axes[1, 1].scatter(s_comparison['phase_score_pred'], s_comparison['time_diff_ms'], 
                          alpha=0.6, color='red', label='S-phase', s=30)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('PhaseNet Confidence Score', fontsize=12)
        axes[1, 1].set_ylabel('Picking Error (ms)', fontsize=12)
        axes[1, 1].set_title('Picking Error vs Confidence Score', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficients
        if len(p_comparison) > 1:
            p_corr = np.corrcoef(p_comparison['phase_score_pred'], np.abs(p_comparison['time_diff_ms']))[0,1]
            axes[1, 1].text(0.05, 0.95, f'P-phase |r| = {abs(p_corr):.3f}', 
                           transform=axes[1, 1].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        if len(s_comparison) > 1:
            s_corr = np.corrcoef(s_comparison['phase_score_pred'], np.abs(s_comparison['time_diff_ms']))[0,1]
            axes[1, 1].text(0.05, 0.85, f'S-phase |r| = {abs(s_corr):.3f}', 
                           transform=axes[1, 1].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Hide empty subplot if no data
    if not (p_picking_accuracy and s_picking_accuracy):
        for i in range(2):
            for j in range(2):
                if not axes[i, j].has_data():
                    axes[i, j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('picking_error_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create cumulative distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if p_picking_accuracy and len(p_comparison) > 0:
        sorted_p_errors = np.sort(np.abs(p_comparison['time_diff_ms']))
        y_p = np.arange(len(sorted_p_errors)) / float(len(sorted_p_errors))
        ax.plot(sorted_p_errors, y_p, label=f'P-phase (n={len(p_comparison)})', color='blue', linewidth=2)
    
    if s_picking_accuracy and len(s_comparison) > 0:
        sorted_s_errors = np.sort(np.abs(s_comparison['time_diff_ms']))
        y_s = np.arange(len(sorted_s_errors)) / float(len(sorted_s_errors))
        ax.plot(sorted_s_errors, y_s, label=f'S-phase (n={len(s_comparison)})', color='red', linewidth=2)
    
    ax.set_xlabel('Absolute Picking Error (ms)', fontsize=12)
    ax.set_ylabel('Cumulative Fraction', fontsize=12)
    ax.set_title('Cumulative Distribution of Absolute Picking Errors', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add tolerance lines
    for tolerance in [10, 20, 50, 100]:
        ax.axvline(x=tolerance, color='gray', linestyle='--', alpha=0.5)
        ax.text(tolerance, 0.1, f'{tolerance} ms', rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('cumulative_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# COMPARISON PLOTS FUNCTION
# ============================================================================

def create_comparison_plots(p_picking_accuracy, s_picking_accuracy, p_detection, s_detection):
    """Create comparison plots between your results and paper results"""
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PhaseNet Performance: Your Study vs Paper Reference', fontsize=16, fontweight='bold')
    
    # Plot 1: Time Accuracy Comparison (Mean ± Std)
    if p_picking_accuracy and s_picking_accuracy:
        phases = ['P-phase', 'S-phase']
        your_means = [p_picking_accuracy['mean_picking_error_ms'], s_picking_accuracy['mean_picking_error_ms']]
        your_stds = [p_picking_accuracy['std_picking_error_ms'], s_picking_accuracy['std_picking_error_ms']]
        paper_means = [paper_results['P']['travel_time_residual_mean_ms'], paper_results['S']['travel_time_residual_mean_ms']]
        paper_stds = [paper_results['P']['travel_time_residual_std_ms'], paper_results['S']['travel_time_residual_std_ms']]
        
        x_pos = np.arange(len(phases))
        width = 0.35
        
        # Your results
        axes[0, 0].bar(x_pos - width/2, your_means, width, yerr=your_stds, 
                       capsize=5, label='Your Study (Picking Error)', alpha=0.7, color='blue')
        # Paper results
        axes[0, 0].bar(x_pos + width/2, paper_means, width, yerr=paper_stds, 
                       capsize=5, label='Paper (Travel Time Residual)', alpha=0.7, color='red')
        
        axes[0, 0].set_xlabel('Phase Type', fontsize=12)
        axes[0, 0].set_ylabel('Mean ± Std (ms)', fontsize=12)
        axes[0, 0].set_title('Time Accuracy: Mean ± Standard Deviation', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(phases)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (your_mean, paper_mean) in enumerate(zip(your_means, paper_means)):
            axes[0, 0].text(i - width/2, your_means[i] + your_stds[i] + 5, f'{your_mean:.1f}', 
                           ha='center', va='bottom', fontweight='bold')
            axes[0, 0].text(i + width/2, paper_means[i] + paper_stds[i] + 5, f'{paper_mean:.1f}', 
                           ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Detection Performance Comparison
    if p_detection and s_detection:
        metrics = ['Precision', 'Recall', 'F1-Score']
        p_your = [p_detection['precision'], p_detection['recall'], p_detection['f1_score']]
        s_your = [s_detection['precision'], s_detection['recall'], s_detection['f1_score']]
        p_paper = [paper_results['P']['precision'], paper_results['P']['recall'], paper_results['P']['f1_score']]
        s_paper = [paper_results['S']['precision'], paper_results['S']['recall'], paper_results['S']['f1_score']]
        
        x_pos = np.arange(len(metrics))
        width = 0.2
        
        # P-phase
        axes[0, 1].bar(x_pos - width*1.5, p_your, width, label='Your P-phase', alpha=0.7, color='lightblue')
        axes[0, 1].bar(x_pos - width/2, p_paper, width, label='Paper P-phase', alpha=0.7, color='darkblue')
        # S-phase
        axes[0, 1].bar(x_pos + width/2, s_your, width, label='Your S-phase', alpha=0.7, color='lightcoral')
        axes[0, 1].bar(x_pos + width*1.5, s_paper, width, label='Paper S-phase', alpha=0.7, color='darkred')
        
        axes[0, 1].set_xlabel('Metrics', fontsize=12)
        axes[0, 1].set_ylabel('Score', fontsize=12)
        axes[0, 1].set_title('Detection Performance Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(metrics)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1.0)
    
    # Plot 3: Additional Metrics (Your Study Only)
    if p_picking_accuracy and s_picking_accuracy:
        metrics_add = ['MAE', 'RMSE']
        p_add = [p_picking_accuracy['mae_ms'], p_picking_accuracy['rmse_ms']]
        s_add = [s_picking_accuracy['mae_ms'], s_picking_accuracy['rmse_ms']]
        
        x_pos = np.arange(len(metrics_add))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, p_add, width, label='P-phase', alpha=0.7, color='blue')
        axes[1, 0].bar(x_pos + width/2, s_add, width, label='S-phase', alpha=0.7, color='red')
        
        axes[1, 0].set_xlabel('Metrics', fontsize=12)
        axes[1, 0].set_ylabel('Error (ms)', fontsize=12)
        axes[1, 0].set_title('Additional Error Metrics (Your Study)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(metrics_add)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, (p_val, s_val) in enumerate(zip(p_add, s_add)):
            axes[1, 0].text(i - width/2, p_val + 1, f'{p_val:.1f}', ha='center', va='bottom', fontweight='bold')
            axes[1, 0].text(i + width/2, s_val + 1, f'{s_val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Performance Summary Radar Chart (simplified)
    if p_detection and s_detection:
        # Simplified performance indicators
        categories = ['Time\nAccuracy', 'Detection\nPrecision', 'Detection\nRecall', 'Overall\nF1']
        
        # Normalize scores for radar chart (simplified to bar chart)
        p_performance = [
            min(1.0, 100 / (abs(p_picking_accuracy['mae_ms']) + 1)) if p_picking_accuracy else 0,
            p_detection['precision'],
            p_detection['recall'], 
            p_detection['f1_score']
        ]
        
        s_performance = [
            min(1.0, 100 / (abs(s_picking_accuracy['mae_ms']) + 1)) if s_picking_accuracy else 0,
            s_detection['precision'],
            s_detection['recall'],
            s_detection['f1_score']
        ]
        
        x_pos = np.arange(len(categories))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, p_performance, width, label='P-phase', alpha=0.7, color='blue')
        axes[1, 1].bar(x_pos + width/2, s_performance, width, label='S-phase', alpha=0.7, color='red')
        
        axes[1, 1].set_xlabel('Performance Categories', fontsize=12)
        axes[1, 1].set_ylabel('Normalized Score', fontsize=12)
        axes[1, 1].set_title('Performance Summary (Higher is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# DIRECT PICKING ACCURACY (YOUR METRICS)
# ============================================================================

def calculate_picking_accuracy(comparison_data, phase_name):
    if len(comparison_data) == 0:
        print(f"No {phase_name} picks for direct accuracy comparison")
        return None
    
    time_diffs_ms = comparison_data['time_diff_ms'].values
    
    # Picking accuracy metrics (YOUR ANALYSIS)
    mean_picking_error_ms = np.mean(time_diffs_ms)
    std_picking_error_ms = np.std(time_diffs_ms)
    mae_ms = np.mean(np.abs(time_diffs_ms))
    rmse_ms = np.sqrt(np.mean(time_diffs_ms**2))
    
    # Additional metrics
    pcc, p_value = pearsonr(comparison_data['phase_timestamp_manual'], comparison_data['phase_timestamp_pred'])
    
    stats = {
        'n_picks': len(comparison_data),
        # Picking accuracy (your metrics)
        'mean_picking_error_ms': mean_picking_error_ms,
        'std_picking_error_ms': std_picking_error_ms,
        'mae_ms': mae_ms,
        'rmse_ms': rmse_ms,
        'pearson_r': pcc,
        # Additional statistics
        'min_error_ms': np.min(time_diffs_ms),
        'max_error_ms': np.max(time_diffs_ms),
        'median_error_ms': np.median(time_diffs_ms)
    }
    
    print(f"\n--- {phase_name} PHASE - DIRECT PICKING ACCURACY ---")
    print(f"Number of matched picks: {stats['n_picks']}")
    
    print(f"\nPICKING ERROR METRICS (Your Analysis):")
    print(f"  Mean picking error: {stats['mean_picking_error_ms']:.3f} ms")
    print(f"  Std dev of picking errors: {stats['std_picking_error_ms']:.3f} ms")
    print(f"  MAE: {stats['mae_ms']:.3f} ms")
    print(f"  RMSE: {stats['rmse_ms']:.3f} ms")
    print(f"  Error range: [{stats['min_error_ms']:.3f}, {stats['max_error_ms']:.3f}] ms")
    print(f"  Median error: {stats['median_error_ms']:.3f} ms")
    print(f"  Pearson correlation: {stats['pearson_r']:.3f}")
    
    # Convert paper's travel time residuals to comparable format
    paper_mean_ms = paper_results[phase_name]['travel_time_residual_mean_ms']
    paper_std_ms = paper_results[phase_name]['travel_time_residual_std_ms']
    
    print(f"\nCOMPARISON CONTEXT:")
    print(f"  Your picking error (mean±std): {stats['mean_picking_error_ms']:.3f} ± {stats['std_picking_error_ms']:.3f} ms")
    print(f"  Paper's travel time residual (mean±std): {paper_mean_ms:.3f} ± {paper_std_ms:.3f} ms")
    print(f"  NOTE: Paper reports travel time residuals from earthquake location, not direct picking errors")
    
    # Tolerance analysis in ms
    print(f"\nTOLERANCE ANALYSIS:")
    for tolerance_ms in [10, 20, 50, 100]:
        within_tol = np.sum(np.abs(time_diffs_ms) <= tolerance_ms)
        percentage = (within_tol / len(time_diffs_ms)) * 100
        print(f"  Within ±{tolerance_ms} ms: {within_tol}/{len(time_diffs_ms)} ({percentage:.1f}%)")
    
    return stats

p_picking_accuracy = calculate_picking_accuracy(p_comparison, "P") if 'p_comparison' in locals() and len(p_comparison) > 0 else None
s_picking_accuracy = calculate_picking_accuracy(s_comparison, "S") if 's_comparison' in locals() and len(s_comparison) > 0 else None

# ============================================================================
# CREATE HISTOGRAM PLOTS
# ============================================================================

print(f"\n📊 GENERATING HISTOGRAM PLOTS...")
create_histogram_plots(p_comparison, s_comparison, p_picking_accuracy, s_picking_accuracy)

# ============================================================================
# DETECTION PERFORMANCE (COMPARABLE METRIC)
# ============================================================================

def create_detection_dataset(manual_df, pred_df, tolerance_ms=50):
    detection_data = []
    
    for _, manual_row in manual_df.iterrows():
        station = manual_row['station']
        event_id = manual_row['event_id']
        
        preds_for_event = pred_df[(pred_df['station'] == station) & (pred_df['event_id'] == event_id)]
        
        # P pick detection
        p_manual_time = manual_row['p_timestamp'] if not pd.isna(manual_row['p_timestamp']) else None
        p_detected = 0
        p_confidence = 0.0
        
        if p_manual_time is not None:
            p_preds = preds_for_event[preds_for_event['phase_type'] == 'P']
            for _, pred_row in p_preds.iterrows():
                time_diff_ms = abs(pred_row['phase_timestamp'] - p_manual_time) * 1000
                if time_diff_ms <= tolerance_ms:
                    p_detected = 1
                    p_confidence = pred_row['phase_score']
                    break
        
        # S pick detection
        s_manual_time = manual_row['s_timestamp'] if not pd.isna(manual_row['s_timestamp']) else None
        s_detected = 0
        s_confidence = 0.0
        
        if s_manual_time is not None:
            s_preds = preds_for_event[preds_for_event['phase_type'] == 'S']
            for _, pred_row in s_preds.iterrows():
                time_diff_ms = abs(pred_row['phase_timestamp'] - s_manual_time) * 1000
                if time_diff_ms <= tolerance_ms:
                    s_detected = 1
                    s_confidence = pred_row['phase_score']
                    break
        
        detection_data.append({
            'station': station, 'event_id': event_id,
            'p_exists': 1 if p_manual_time is not None else 0,
            'p_detected': p_detected, 'p_confidence': p_confidence,
            's_exists': 1 if s_manual_time is not None else 0, 
            's_detected': s_detected, 's_confidence': s_confidence
        })
    
    return pd.DataFrame(detection_data)

detection_df = create_detection_dataset(manual_df, pred_df, tolerance_ms=50)
print(f"\nDetection dataset created with {len(detection_df)} station-events")

def calculate_detection_performance(phase_type, detection_df):
    if phase_type == 'P':
        true_exists = detection_df['p_exists']
        pred_detected = detection_df['p_detected']
        pred_confidence = detection_df['p_confidence']
    else:
        true_exists = detection_df['s_exists']
        pred_detected = detection_df['s_detected']
        pred_confidence = detection_df['s_confidence']
    
    eval_mask = true_exists == 1
    y_true = pred_detected[eval_mask]
    y_scores = pred_confidence[eval_mask]
    
    if len(y_true) == 0:
        print(f"No {phase_type} picks available for detection evaluation")
        return None
    
    y_pred = (y_scores > 0.5).astype(int)
    
    # Detection metrics (COMPARABLE TO PAPER)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Your additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'n_samples': len(y_true),
        # Comparable to paper
        'precision': precision,
        'recall': recall, 
        'f1_score': f1,
        # Your additional metrics
        'accuracy': accuracy,
        'mcc': mcc,
        'confusion_matrix': conf_matrix
    }
    
    print(f"\n--- {phase_type} PHASE - DETECTION PERFORMANCE ---")
    print(f"Samples: {metrics['n_samples']}")
    
    print(f"\nDETECTION METRICS (Comparable to Paper):")
    print(f"  Precision: {metrics['precision']:.3f} (Paper: {paper_results[phase_type]['precision']:.3f})")
    print(f"  Recall: {metrics['recall']:.3f} (Paper: {paper_results[phase_type]['recall']:.3f})")
    print(f"  F1-Score: {metrics['f1_score']:.3f} (Paper: {paper_results[phase_type]['f1_score']:.3f})")
    
    print(f"\nADDITIONAL DETECTION METRICS (Your Analysis):")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Matthews CC: {metrics['mcc']:.3f}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    return metrics

p_detection = calculate_detection_performance('P', detection_df)
s_detection = calculate_detection_performance('S', detection_df)

# ============================================================================
# CREATE COMPARISON PLOTS
# ============================================================================

print(f"\n📈 GENERATING COMPARISON PLOTS...")
create_comparison_plots(p_picking_accuracy, s_picking_accuracy, p_detection, s_detection)

# ============================================================================
# COMPREHENSIVE COMPARISON TABLE
# ============================================================================

print("\n" + "="*120)
print("COMPREHENSIVE COMPARISON: DIRECT PICKING ACCURACY vs PAPER'S TRAVEL TIME RESIDUALS")
print("="*120)

comparison_data = []

for phase in ['P', 'S']:
    my_picking = p_picking_accuracy if phase == 'P' else s_picking_accuracy
    my_detection = p_detection if phase == 'P' else s_detection
    paper = paper_results[phase]
    
    if my_picking and my_detection:
        # Your results - Picking Accuracy
        comparison_data.append({
            'Phase': phase,
            'Metric Type': 'Picking Accuracy (Your Study)',
            'Mean (ms)': f"{my_picking['mean_picking_error_ms']:.3f}",
            'Std Dev (ms)': f"{my_picking['std_picking_error_ms']:.3f}",
            'MAE (ms)': f"{my_picking['mae_ms']:.3f}",
            'RMSE (ms)': f"{my_picking['rmse_ms']:.3f}",
            'Precision': f"{my_detection['precision']:.3f}",
            'Recall': f"{my_detection['recall']:.3f}",
            'F1-Score': f"{my_detection['f1_score']:.3f}",
            'Accuracy': f"{my_detection['accuracy']:.3f}",
            'MCC': f"{my_detection['mcc']:.3f}",
            'Samples': f"{my_picking['n_picks']} picks"
        })
    
    # Paper's results - Travel Time Residuals
    comparison_data.append({
        'Phase': phase,
        'Metric Type': 'Travel Time Residuals (Paper)',
        'Mean (ms)': f"{paper['travel_time_residual_mean_ms']:.3f}",
        'Std Dev (ms)': f"{paper['travel_time_residual_std_ms']:.3f}",
        'MAE (ms)': 'N/A',
        'RMSE (ms)': 'N/A',
        'Precision': f"{paper['precision']:.3f}",
        'Recall': f"{paper['recall']:.3f}",
        'F1-Score': f"{paper['f1_score']:.3f}",
        'Accuracy': 'N/A',
        'MCC': 'N/A',
        'Samples': f"{paper['dataset_size']} traces"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n📊 COMPREHENSIVE COMPARISON TABLE")
print("NOTE: Different metrics - Your study measures direct picking accuracy, Paper reports travel time residuals from earthquake location")
print(comparison_df.to_string(index=False))

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"\n📁 OUTPUT FILES GENERATED:")
print(f"1. {output_filename} - Comprehensive results with all metrics")
print(f"2. picking_error_histograms.png - Error distribution analysis")
print(f"3. cumulative_error_distribution.png - Cumulative error plots") 
print(f"4. performance_comparison.png - Your study vs paper comparison")

print(f"\n📈 KEY METRICS COLLECTED:")
print(f"• Regression Metrics: Mean, Std, MAE, RMSE, Pearson correlation")
print(f"• Classification Metrics: Precision, Recall, F1-Score, Accuracy, MCC")
print(f"• Additional Analysis: Tolerance analysis, confidence correlation")

print(f"\n✅ ANALYSIS COMPLETE!")
print("All results saved to file and comparison plots generated.")

# Close output file and restore stdout
sys.stdout = original_stdout
output_file.close()

print(f"\n📄 Results saved to: {output_filename}")
print("🖼️  Plots saved as PNG files")