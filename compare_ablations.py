#!/usr/bin/env python3
"""
Extract metrics from all ablation experiments and create comparison table
Compares PREDICTED PICKS from different PhaseNet models against MANUAL PICKS (ground truth)

Usage: python compare_ablations.py

Input Files Location: /Users/saheeadewaledalli/PhaseNet/test_data/converted_back_to_npz_3comp/
"""

import pandas as pd
import re
import os

# Configuration with your specific paths
MANUAL_PICKS_BASE_DIR = "/Users/saheeadewaledalli/PhaseNet/test_data/converted_back_to_npz_3comp"
PREDICTED_PICKS_BASE_DIR = "results"  # Contains picks from all model variants

def count_manual_picks():
    """Count the actual number of manual picks from the ground truth file"""
    manual_picks_file = f"{MANUAL_PICKS_BASE_DIR}/npz_data_list_detailed.csv"
    if os.path.exists(manual_picks_file):
        try:
            # Read the CSV and count rows (excluding header)
            df = pd.read_csv(manual_picks_file)
            return len(df)
        except Exception as e:
            print(f"Warning: Could not read manual picks file: {e}")
            # Fallback: count lines in file
            with open(manual_picks_file, 'r') as f:
                lines = f.readlines()
            return len(lines) - 1 if len(lines) > 0 else 0  # Subtract header
    return 0

def count_predicted_picks(picks_file):
    """Count the number of predicted picks from a picks.csv file"""
    if os.path.exists(picks_file):
        try:
            # Try to read as CSV first (more reliable)
            df = pd.read_csv(picks_file)
            return len(df)
        except Exception as e:
            print(f"Warning: Could not read predicted picks as CSV {picks_file}: {e}")
            # Fallback: count lines in file
            with open(picks_file, 'r') as f:
                lines = f.readlines()
            return len(lines) - 1 if len(lines) > 0 else 0  # Subtract header
    return 0

def calculate_precision_recall_f1(manual_picks_count, predicted_picks_count, matched_picks):
    """Calculate precision, recall, and F1 score"""
    if manual_picks_count == 0 or predicted_picks_count == 0:
        return None, None, None
    
    precision = matched_picks / predicted_picks_count if predicted_picks_count > 0 else 0
    recall = matched_picks / manual_picks_count if manual_picks_count > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
        
    return precision, recall, f1

def extract_metrics_from_file(filepath):
    """Extract key metrics from phasenet_comprehensive_results.txt - FIXED VERSION"""
    metrics = {}
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        print(f"    📄 Analyzing file: {os.path.basename(filepath)}")
        
        # Initialize all metrics to None
        p_matches = p_mae = p_rmse = p_mean = p_std = p_median = None
        s_matches = s_mae = s_rmse = s_mean = s_std = s_median = None
        
        # Split content into lines for more robust parsing
        lines = content.split('\n')
        
        # Parse each line individually
        for i, line in enumerate(lines):
            line = line.strip()
            
            # P-phase metrics
            if 'P-PHASE RESULTS:' in line:
                # Look ahead in the next lines for P-phase metrics
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    if 'Matched picks:' in next_line:
                        p_matches = int(re.search(r'Matched picks:\s*(\d+)', next_line).group(1))
                    elif 'MAE:' in next_line:
                        p_mae = float(re.search(r'MAE:\s*([\d.]+)', next_line).group(1))
                    elif 'RMSE:' in next_line:
                        p_rmse = float(re.search(r'RMSE:\s*([\d.]+)', next_line).group(1))
                    elif 'Mean error:' in next_line:
                        p_mean = float(re.search(r'Mean error:\s*([\d.-]+)', next_line).group(1))
                    elif 'Std deviation:' in next_line:
                        p_std = float(re.search(r'Std deviation:\s*([\d.]+)', next_line).group(1))
                    elif 'Median error:' in next_line:
                        p_median = float(re.search(r'Median error:\s*([\d.-]+)', next_line).group(1))
                    elif 'S-PHASE RESULTS:' in next_line:
                        break
            
            # S-phase metrics  
            elif 'S-PHASE RESULTS:' in line:
                # Look ahead in the next lines for S-phase metrics
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    if 'Matched picks:' in next_line:
                        s_matches = int(re.search(r'Matched picks:\s*(\d+)', next_line).group(1))
                    elif 'MAE:' in next_line:
                        s_mae = float(re.search(r'MAE:\s*([\d.]+)', next_line).group(1))
                    elif 'RMSE:' in next_line:
                        s_rmse = float(re.search(r'RMSE:\s*([\d.]+)', next_line).group(1))
                    elif 'Mean error:' in next_line:
                        s_mean = float(re.search(r'Mean error:\s*([\d.-]+)', next_line).group(1))
                    elif 'Std deviation:' in next_line:
                        s_std = float(re.search(r'Std deviation:\s*([\d.]+)', next_line).group(1))
                    elif 'Median error:' in next_line:
                        s_median = float(re.search(r'Median error:\s*([\d.-]+)', next_line).group(1))
                    elif line and j > i+8:  # Stop after reasonable number of lines
                        break
        
        # Debug: Print what was found
        print(f"    🔍 Parsed metrics:")
        print(f"      P: Matches={p_matches}, MAE={p_mae}, RMSE={p_rmse}, Mean={p_mean}, Std={p_std}, Median={p_median}")
        print(f"      S: Matches={s_matches}, MAE={s_mae}, RMSE={s_rmse}, Mean={s_mean}, Std={s_std}, Median={s_median}")
        
        # Check if we found the essential metrics
        if p_matches is None or p_mae is None or s_matches is None or s_mae is None:
            print(f"Warning: Essential metrics missing in {filepath}")
            # Try alternative parsing method as fallback
            return extract_metrics_alternative(filepath, content)
        
        # Get configuration name from file path to find predicted picks count
        config_name = os.path.basename(os.path.dirname(filepath))
        predicted_picks_file = f"{PREDICTED_PICKS_BASE_DIR}/{config_name}/picks.csv"
        
        # Count predicted picks (this is the total number of predictions made by the model)
        predicted_picks_count = count_predicted_picks(predicted_picks_file)
        
        # Count manual picks dynamically
        manual_picks_count = count_manual_picks()
        
        if manual_picks_count == 0:
            print(f"Warning: No manual picks found for ground truth")
            return None
        
        # Calculate precision/recall/F1 for P and S phases
        p_precision, p_recall, p_f1 = calculate_precision_recall_f1(
            manual_picks_count, predicted_picks_count, p_matches
        )
        s_precision, s_recall, s_f1 = calculate_precision_recall_f1(
            manual_picks_count, predicted_picks_count, s_matches
        )
        
        metrics = {
            'P_Matches': p_matches,
            'P_MAE_ms': p_mae,
            'P_RMSE_ms': p_rmse,
            'P_Mean_ms': p_mean,
            'P_Std_ms': p_std,
            'P_Median_ms': p_median,
            'P_Precision': p_precision,
            'P_Recall': p_recall,
            'P_F1': p_f1,
            'S_Matches': s_matches,
            'S_MAE_ms': s_mae,
            'S_RMSE_ms': s_rmse,
            'S_Mean_ms': s_mean,
            'S_Std_ms': s_std,
            'S_Median_ms': s_median,
            'S_Precision': s_precision,
            'S_Recall': s_recall,
            'S_F1': s_f1,
            'N_picks': predicted_picks_count,
            'Manual_Picks_Count': manual_picks_count
        }
        
        print(f"    ✅ Successfully extracted all metrics")
        
        return metrics
        
    except Exception as e:
        print(f"Warning: Error parsing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_metrics_alternative(filepath, content):
    """Alternative parsing method as fallback"""
    print(f"    🔧 Trying alternative parsing for {filepath}")
    
    try:
        # Try simple keyword-based parsing
        lines = content.split('\n')
        metrics = {}
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # P-phase metrics
            if 'P-PHASE RESULTS:' in line:
                # The next lines should contain the metrics
                for j in range(i+1, min(i+15, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    if 'Matched picks:' in next_line:
                        metrics['P_Matches'] = int(re.search(r'(\d+)', next_line).group(1))
                    elif 'MAE:' in next_line:
                        metrics['P_MAE_ms'] = float(re.search(r'([\d.]+)', next_line).group(1))
                    elif 'RMSE:' in next_line:
                        metrics['P_RMSE_ms'] = float(re.search(r'([\d.]+)', next_line).group(1))
                    elif 'Mean error:' in next_line:
                        metrics['P_Mean_ms'] = float(re.search(r'([\d.-]+)', next_line).group(1))
                    elif 'Std deviation:' in next_line:
                        metrics['P_Std_ms'] = float(re.search(r'([\d.]+)', next_line).group(1))
                    elif 'Median error:' in next_line:
                        metrics['P_Median_ms'] = float(re.search(r'([\d.-]+)', next_line).group(1))
            
            # S-phase metrics
            elif 'S-PHASE RESULTS:' in line:
                for j in range(i+1, min(i+15, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    
                    if 'Matched picks:' in next_line:
                        metrics['S_Matches'] = int(re.search(r'(\d+)', next_line).group(1))
                    elif 'MAE:' in next_line:
                        metrics['S_MAE_ms'] = float(re.search(r'([\d.]+)', next_line).group(1))
                    elif 'RMSE:' in next_line:
                        metrics['S_RMSE_ms'] = float(re.search(r'([\d.]+)', next_line).group(1))
                    elif 'Mean error:' in next_line:
                        metrics['S_Mean_ms'] = float(re.search(r'([\d.-]+)', next_line).group(1))
                    elif 'Std deviation:' in next_line:
                        metrics['S_Std_ms'] = float(re.search(r'([\d.]+)', next_line).group(1))
                    elif 'Median error:' in next_line:
                        metrics['S_Median_ms'] = float(re.search(r'([\d.-]+)', next_line).group(1))
        
        # Check if we got the essential metrics
        required = ['P_Matches', 'P_MAE_ms', 'S_Matches', 'S_MAE_ms']
        if all(key in metrics for key in required):
            print(f"    ✅ Alternative parsing successful")
            return metrics
        else:
            print(f"    ❌ Alternative parsing failed - missing required metrics")
            return None
            
    except Exception as e:
        print(f"Warning: Alternative parsing failed for {filepath}: {e}")
        return None

def check_input_files():
    """Check if all required input files exist"""
    print("🔍 CHECKING INPUT FILES")
    print("=" * 50)
    
    # Check manual picks file (GROUND TRUTH)
    manual_picks = f"{MANUAL_PICKS_BASE_DIR}/npz_data_list_detailed.csv"
    manual_picks_count = count_manual_picks()
    if os.path.exists(manual_picks):
        print(f"✅ Manual picks (GROUND TRUTH): {manual_picks}")
        print(f"   Contains {manual_picks_count} manual phase picks")
    else:
        print(f"❌ Manual picks (GROUND TRUTH) NOT FOUND: {manual_picks}")
    
    # Check predicted picks files (MODEL OUTPUTS)
    print(f"\n📊 Predicted picks from models:")
    configs = ['baseline', 'augmentation', 'temporal_loss', 'deeper', 'wider', 'full']
    for config in configs:
        picks_file = f"{PREDICTED_PICKS_BASE_DIR}/{config}/picks.csv"
        predicted_count = count_predicted_picks(picks_file)
        if os.path.exists(picks_file):
            print(f"   ✅ {config}: {predicted_count} predicted picks")
        else:
            print(f"   ❌ {config}: No predicted picks found")
    
    print("")

def main():
    print("=" * 80)
    print("PHASENET ABLATION STUDY - PREDICTION PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"📁 Manual Picks (Ground Truth): {MANUAL_PICKS_BASE_DIR}")
    print(f"📊 Predicted Picks (Model Outputs): {PREDICTED_PICKS_BASE_DIR}/")
    print("")
    print("🎯 Comparing 6 PhaseNet variants against manual annotations")
    print("")
    
    # Check input files first
    check_input_files()
    
    # Configuration names and descriptions
    configs = {
        'baseline': 'Baseline (Your Improved PhaseNet)',
        'augmentation': '+ Data Augmentation',
        'temporal_loss': '+ Temporal Consistency Loss',
        'deeper': '+ Deeper Network (6 levels)',
        'wider': '+ Wider Network (16 filters)',
        'full': 'Full Model (All Improvements)'
    }
    
    results = []
    
    print("🔍 LOADING EVALUATION RESULTS")
    print("=" * 50)
    print("Reading comparison results between PREDICTED vs MANUAL picks:")
    
    for config_name, description in configs.items():
        # Path to evaluation results (already compared predicted vs manual)
        filepath = f'{PREDICTED_PICKS_BASE_DIR}/{config_name}/phasenet_comprehensive_results.txt'
        print(f"  Loading: {filepath}")
        
        metrics = extract_metrics_from_file(filepath)
        
        if metrics:
            # Format precision/recall/F1 as percentages for display
            p_precision_pct = metrics['P_Precision'] * 100 if metrics['P_Precision'] is not None else None
            p_recall_pct = metrics['P_Recall'] * 100 if metrics['P_Recall'] is not None else None
            s_precision_pct = metrics['S_Precision'] * 100 if metrics['S_Precision'] is not None else None
            s_recall_pct = metrics['S_Recall'] * 100 if metrics['S_Recall'] is not None else None
            
            results.append({
                'Configuration': config_name,
                'Description': description,
                'P-Matches': metrics['P_Matches'],
                'P-MAE (ms)': metrics['P_MAE_ms'],
                'P-RMSE (ms)': metrics['P_RMSE_ms'],
                'P-Mean (ms)': metrics['P_Mean_ms'],
                'P-Std (ms)': metrics['P_Std_ms'],
                'P-Median (ms)': metrics['P_Median_ms'],
                'P-Precision': p_precision_pct,
                'P-Recall': p_recall_pct,
                'P-F1': metrics['P_F1'] * 100 if metrics['P_F1'] is not None else None,
                'S-Matches': metrics['S_Matches'],
                'S-MAE (ms)': metrics['S_MAE_ms'],
                'S-RMSE (ms)': metrics['S_RMSE_ms'],
                'S-Mean (ms)': metrics['S_Mean_ms'],
                'S-Std (ms)': metrics['S_Std_ms'],
                'S-Median (ms)': metrics['S_Median_ms'],
                'S-Precision': s_precision_pct,
                'S-Recall': s_recall_pct,
                'S-F1': metrics['S_F1'] * 100 if metrics['S_F1'] is not None else None,
                'N_picks': metrics['N_picks'],
                'Manual_Picks_Count': metrics['Manual_Picks_Count']
            })
            print(f"    ✅ {metrics['P_Matches']} P-pick matches, {metrics['S_Matches']} S-pick matches")
            if metrics['P_Precision'] is not None and metrics['P_Recall'] is not None:
                print(f"       P-phase: Precision={p_precision_pct:.1f}%, Recall={p_recall_pct:.1f}%, F1={metrics['P_F1']*100 if metrics['P_F1'] else 'N/A':.1f}%")
                print(f"       S-phase: Precision={s_precision_pct:.1f}%, Recall={s_recall_pct:.1f}%, F1={metrics['S_F1']*100 if metrics['S_F1'] else 'N/A':.1f}%")
        else:
            results.append({
                'Configuration': config_name,
                'Description': description,
                'P-Matches': 'N/A',
                'P-MAE (ms)': 'N/A',
                'P-RMSE (ms)': 'N/A',
                'P-Mean (ms)': 'N/A',
                'P-Std (ms)': 'N/A',
                'P-Median (ms)': 'N/A',
                'P-Precision': 'N/A',
                'P-Recall': 'N/A',
                'P-F1': 'N/A',
                'S-Matches': 'N/A',
                'S-MAE (ms)': 'N/A',
                'S-RMSE (ms)': 'N/A',
                'S-Mean (ms)': 'N/A',
                'S-Std (ms)': 'N/A',
                'S-Median (ms)': 'N/A',
                'S-Precision': 'N/A',
                'S-Recall': 'N/A',
                'S-F1': 'N/A',
                'N_picks': 'N/A',
                'Manual_Picks_Count': 'N/A'
            })
            print(f"    ❌ No evaluation results found")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate improvements relative to baseline - ONLY if baseline has valid data
    baseline_has_data = (len(results) > 0 and 
                        results[0]['P-MAE (ms)'] != 'N/A' and 
                        results[0]['P-MAE (ms)'] is not None and
                        results[0]['P-MAE (ms)'] > 0)
    
    if baseline_has_data:
        baseline_p_mae = results[0]['P-MAE (ms)']
        baseline_s_mae = results[0]['S-MAE (ms)']
        
        improvements = []
        for i, row in enumerate(results):
            if i == 0:
                improvements.append('(baseline)')
            elif (row['P-MAE (ms)'] != 'N/A' and row['P-MAE (ms)'] is not None and
                  row['S-MAE (ms)'] != 'N/A' and row['S-MAE (ms)'] is not None):
                p_change = ((row['P-MAE (ms)'] - baseline_p_mae) / baseline_p_mae) * 100
                s_change = ((row['S-MAE (ms)'] - baseline_s_mae) / baseline_s_mae) * 100
                improvements.append(f"P:{p_change:+.1f}%, S:{s_change:+.1f}%")
            else:
                improvements.append('N/A')
        
        df['MAE Change vs Baseline'] = improvements
    else:
        print("⚠️  No valid baseline data found - skipping improvement calculations")
        df['MAE Change vs Baseline'] = ['N/A'] * len(results)
    
    # Print results
    print("\n" + "="*120)
    print("PHASENET ABLATION STUDY - PREDICTION PERFORMANCE RESULTS")
    print("="*120)
    print("Comparison of PhaseNet model predictions against manual annotations")
    print("")
    
    # Print detailed table
    print("📊 DETAILED PERFORMANCE METRICS:\n")
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Save to CSV
    csv_path = 'ablation_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Create simplified table for report
    simple_columns = ['Configuration', 'Description', 'P-Matches', 'P-MAE (ms)', 'P-Precision', 'P-Recall', 'P-F1', 
                     'S-Matches', 'S-MAE (ms)', 'S-Precision', 'S-Recall', 'S-F1', 'MAE Change vs Baseline']
    simple_df = df[simple_columns] if all(col in df.columns for col in simple_columns) else df
    
    print("\n" + "="*100)
    print("SIMPLIFIED PERFORMANCE SUMMARY:")
    print("="*100)
    print(simple_df.to_markdown(index=False))
    
    # Save markdown table
    md_path = 'ablation_results_markdown.txt'
    with open(md_path, 'w') as f:
        # Get actual manual picks count from first valid result
        manual_count = None
        for result in results:
            if result['Manual_Picks_Count'] != 'N/A' and result['Manual_Picks_Count'] is not None:
                manual_count = result['Manual_Picks_Count']
                break
        
        f.write("# PhaseNet Ablation Study - Prediction Performance\n\n")
        f.write("## Dataset and Evaluation Setup\n")
        f.write(f"- **Manual Picks (Ground Truth)**: `{MANUAL_PICKS_BASE_DIR}/npz_data_list_detailed.csv`\n")
        f.write(f"- **Manual Picks Count**: {manual_count if manual_count else 'Unknown'} phase picks\n")
        f.write(f"- **Predicted Picks Location**: `{PREDICTED_PICKS_BASE_DIR}/[config]/picks.csv`\n")
        f.write("- **Evaluation**: Comparing model predictions against human-annotated phase picks\n")
        f.write("- **Metrics**: MAE (Mean Absolute Error) in milliseconds, Precision/Recall/F1 scores\n\n")
        
        f.write("## Performance Comparison\n\n")
        f.write(simple_df.to_markdown(index=False))
        f.write("\n\n## Key Findings:\n\n")
        
        # Only calculate findings if we have valid data
        valid_results = [r for r in results if r['P-MAE (ms)'] != 'N/A' and r['P-MAE (ms)'] is not None]
        
        if len(valid_results) > 1:
            baseline = valid_results[0]
            full_model = valid_results[-1] if valid_results[-1]['Configuration'] == 'full' else None
            
            if full_model:
                total_p_improvement = ((full_model['P-MAE (ms)'] - baseline['P-MAE (ms)']) / baseline['P-MAE (ms)']) * 100
                total_s_improvement = ((full_model['S-MAE (ms)'] - baseline['S-MAE (ms)']) / baseline['S-MAE (ms)']) * 100
                
                f.write(f"- **Overall P-phase accuracy improvement**: {total_p_improvement:+.1f}% MAE reduction\n")
                f.write(f"- **Overall S-phase accuracy improvement**: {total_s_improvement:+.1f}% MAE reduction\n")
                
                # Add precision/recall findings
                if full_model['P-Precision'] != 'N/A' and baseline['P-Precision'] != 'N/A':
                    p_precision_improvement = full_model['P-Precision'] - baseline['P-Precision']
                    p_recall_improvement = full_model['P-Recall'] - baseline['P-Recall']
                    p_f1_improvement = full_model['P-F1'] - baseline['P-F1'] if full_model['P-F1'] != 'N/A' and baseline['P-F1'] != 'N/A' else None
                    
                    f.write(f"- **P-phase precision improvement**: {p_precision_improvement:+.1f}%\n")
                    f.write(f"- **P-phase recall improvement**: {p_recall_improvement:+.1f}%\n")
                    if p_f1_improvement is not None:
                        f.write(f"- **P-phase F1 improvement**: {p_f1_improvement:+.1f}%\n")
        else:
            f.write("- **Insufficient data**: Need to run evaluation for more models to get meaningful comparisons\n")
    
    print(f"\n✅ Markdown report saved to: {md_path}")
    
    # Create visualization if we have valid data
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Filter out configurations with no data
        valid_configs = []
        valid_p_mae = []
        valid_s_mae = []
        valid_p_f1 = []
        valid_s_f1 = []
        
        for i, row in enumerate(results):
            if (row['P-MAE (ms)'] != 'N/A' and row['P-MAE (ms)'] is not None and
                row['S-MAE (ms)'] != 'N/A' and row['S-MAE (ms)'] is not None):
                valid_configs.append(row['Configuration'])
                valid_p_mae.append(row['P-MAE (ms)'])
                valid_s_mae.append(row['S-MAE (ms)'])
                valid_p_f1.append(row['P-F1'] if row['P-F1'] != 'N/A' else 0)
                valid_s_f1.append(row['S-F1'] if row['S-F1'] != 'N/A' else 0)
        
        if len(valid_configs) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot 1: MAE comparison (Prediction Accuracy)
            x = np.arange(len(valid_configs))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, valid_p_mae, width, label='P-phase MAE', alpha=0.8, color='skyblue')
            bars2 = ax1.bar(x + width/2, valid_s_mae, width, label='S-phase MAE', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('PhaseNet Configuration', fontsize=12)
            ax1.set_ylabel('MAE (ms) - Lower is Better', fontsize=12)
            ax1.set_title('Phase Picking Accuracy: Prediction Error vs Manual Picks', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(valid_configs, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            for bar in bars2:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            
            # Plot 2: F1 Score comparison (Detection Quality)
            ax2.bar(x - width/2, valid_p_f1, width, label='P-phase F1', alpha=0.8, color='skyblue')
            ax2.bar(x + width/2, valid_s_f1, width, label='S-phase F1', alpha=0.8, color='lightcoral')
            
            ax2.set_xlabel('PhaseNet Configuration', fontsize=12)
            ax2.set_ylabel('F1 Score (%) - Higher is Better', fontsize=12)
            ax2.set_title('Detection Quality: F1 Score Comparison', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(valid_configs, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('ablation_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to: ablation_comparison.png")
            
        else:
            print(f"⚠️  Not enough valid data for visualization (need at least 2 models with data)")
            
    except ImportError:
        print(f"⚠️  matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")
    
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"📁 Ground Truth: {MANUAL_PICKS_BASE_DIR}/npz_data_list_detailed.csv")
    print(f"📊 Model Predictions: {PREDICTED_PICKS_BASE_DIR}/[config]/picks.csv")
    print(f"📄 Output Files: ablation_comparison.csv, ablation_results_markdown.txt, ablation_comparison.png")

if __name__ == '__main__':
    main()