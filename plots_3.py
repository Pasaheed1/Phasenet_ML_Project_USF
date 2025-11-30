#!/usr/bin/env python3
"""
Standalone script to plot manual vs predicted picks on waveform components for ALL model configurations
Run after predict.py to compare picks.csv from all models with manual picks
Creates separate images for each model, station, and component
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model configurations to process
CONFIGS = ["baseline", "augmentation", "temporal_loss", "deeper", "wider", "full"]

def extract_station_from_filename(filename):
    """Extract station name from NPZ filename"""
    basename = os.path.basename(filename)
    if basename.endswith('.npz'):
        basename = basename[:-4]
    
    parts = basename.split('.')
    if len(parts) >= 2:
        return parts[1]
    else:
        return basename

def load_predicted_picks(picks_file, config_name):
    """Load predicted picks from PhaseNet output CSV"""
    if not os.path.exists(picks_file):
        logging.error(f"Predicted picks file not found: {picks_file}")
        return []
    
    df = pd.read_csv(picks_file)
    logging.info(f"Loaded {len(df)} predicted picks from {picks_file}")
    
    picks = []
    for _, row in df.iterrows():
        actual_station = extract_station_from_filename(row['file_name'])
        
        picks.append({
            'config': config_name,
            'station_id': actual_station,
            'begin_time': UTCDateTime(row['begin_time']),
            'phase_index': row['phase_index'],
            'phase_time': UTCDateTime(row['phase_time']),
            'phase_score': row['phase_score'],
            'phase_type': row['phase_type'],
            'file_name': row['file_name']
        })
    
    stations = set([p['station_id'] for p in picks])
    logging.info(f"Found predicted picks for {len(stations)} stations in {config_name}")
    
    return picks

def load_manual_picks(manual_picks_file):
    """Load manual picks from detailed CSV - FIXED for relative seconds format"""
    df = pd.read_csv(manual_picks_file, sep='\t')
    logging.info(f"Loaded {len(df)} manual pick records from {manual_picks_file}")
    
    picks = []
    for _, row in df.iterrows():
        station = row['station']
        fname = row['fname']
        
        # Process P pick - these are in SECONDS from waveform start
        if pd.notna(row['p_time']) and row['p_time'] != '' and row['p_time'] != 0:
            try:
                # Extract just the numeric part (remove any trailing text like "IP")
                p_time_str = str(row['p_time']).strip()
                # Remove non-numeric characters at the end
                p_time_clean = re.sub(r'[^0-9.]', '', p_time_str.split()[0] if ' ' in p_time_str else p_time_str)
                p_time_sec = float(p_time_clean)
                
                picks.append({
                    'station': station,
                    'file_name': fname,
                    'phase': 'P',
                    'pick_time': row['p_time'],
                    'pick_time_sec': p_time_sec,  # Store as seconds from start
                    'pick_time_utc': None,  # Will be calculated when we have waveform start time
                    'confidence': row.get('p_weight', 1.0),
                    'remark': row.get('p_remark', ''),
                    'index': row.get('p_idx', 0)
                })
            except Exception as e:
                logging.warning(f"Could not parse P pick time '{row['p_time']}' for {station}: {e}")
                continue
        
        # Process S pick - these are in SECONDS from waveform start  
        if pd.notna(row['s_time']) and row['s_time'] != '' and row['s_time'] != 0:
            try:
                s_time_str = str(row['s_time']).strip()
                s_time_clean = re.sub(r'[^0-9.]', '', s_time_str.split()[0] if ' ' in s_time_str else s_time_str)
                s_time_sec = float(s_time_clean)
                
                picks.append({
                    'station': station,
                    'file_name': fname,
                    'phase': 'S',
                    'pick_time': row['s_time'],
                    'pick_time_sec': s_time_sec,  # Store as seconds from start
                    'pick_time_utc': None,  # Will be calculated when we have waveform start time
                    'confidence': row.get('s_weight', 1.0),
                    'remark': row.get('s_remark', ''),
                    'index': row.get('s_idx', 0)
                })
            except Exception as e:
                logging.warning(f"Could not parse S pick time '{row['s_time']}' for {station}: {e}")
                continue
    
    # Log the stations found
    stations = set([p['station'] for p in picks])
    logging.info(f"Found manual picks for {len(stations)} stations")
    logging.info(f"Processed {len(picks)} manual phase picks")
    
    # Debug: Show some pick times to verify
    logging.info("Sample manual pick times (seconds from start):")
    for i, pick in enumerate(picks[:5]):
        logging.info(f"  {pick['phase']} pick: {pick['pick_time_sec']:.3f}s")
    
    return picks

def load_waveform_data(npz_file):
    """Load waveform data from NPZ file"""
    try:
        data = np.load(npz_file)
        
        if 'data' in data:
            waveform = data['data']
        elif 'waveform' in data:
            waveform = data['waveform']
        else:
            for key in data.files:
                waveform = data[key]
                break
        
        logging.info(f"Loaded waveform from {npz_file}, shape: {waveform.shape}")
        return waveform
        
    except Exception as e:
        logging.error(f"Error loading {npz_file}: {e}")
        return None

def get_waveform_timing_info(npz_file):
    """Extract timing information from NPZ file"""
    try:
        data = np.load(npz_file)
        
        if 'start_time' in data:
            start_time_str = str(data['start_time'])
            return UTCDateTime(start_time_str)
        elif 'begin_time' in data:
            start_time_str = str(data['begin_time'])
            return UTCDateTime(start_time_str)
        
        basename = os.path.basename(npz_file)
        if '.' in basename:
            parts = basename.split('.')
            for part in parts:
                try:
                    if len(part) == 14 and part.isdigit():
                        return UTCDateTime(part)
                except:
                    continue
        
        logging.warning(f"Could not determine start time for {npz_file}")
        return None
        
    except Exception as e:
        logging.error(f"Error getting timing info for {npz_file}: {e}")
        return None

def create_component_plot(npz_filename, station_id, component_data, comp_name, manual_picks, predicted_picks,
                         start_time, sampling_rate=100, output_dir="picks_comparison", config_name="unknown"):
    """Create a single plot for one station and one component - FIXED for relative seconds"""
    
    colors = {'P': 'red', 'S': 'blue'}
    plt.figure(figsize=(15, 8))
    
    n_samples = len(component_data)
    times = [start_time + i * (1.0/sampling_rate) for i in range(n_samples)]
    mpl_times = mdates.date2num([t.datetime for t in times])
    
    plt.plot(mpl_times, component_data, 'k-', linewidth=1.0, alpha=0.8, label=f'Waveform {comp_name}')
    plt.ylabel(f'Amplitude ({comp_name})', fontsize=12)
    
    # Plot manual picks (solid lines) - using relative seconds
    manual_labels_used = set()
    for pick in manual_picks:
        if pick['file_name'] != os.path.basename(npz_filename):
            continue
        
        # Calculate pick time from relative seconds
        if 'pick_time_sec' in pick and pick['pick_time_sec'] is not None:
            pick_time_absolute = start_time + pick['pick_time_sec']
            pick_mpl = mdates.date2num(pick_time_absolute.datetime)
        else:
            logging.warning(f"Cannot determine time for manual {pick['phase']} pick: {pick['pick_time']}")
            continue
            
        color = colors.get(pick['phase'], 'green')
        
        # Verify pick time is within waveform time range
        if pick_mpl < mpl_times[0] or pick_mpl > mpl_times[-1]:
            logging.warning(f"Manual {pick['phase']} pick at {pick_time_absolute} is outside waveform time range")
            logging.warning(f"Pick time: {pick_time_absolute}, Waveform range: {times[0]} to {times[-1]}")
            continue
        
        # Vertical line for pick
        label = f"Manual {pick['phase']}"
        plt.axvline(x=pick_mpl, color=color, linestyle='-', 
                   linewidth=2.5, alpha=0.9, label=label if label not in manual_labels_used else "")
        manual_labels_used.add(label)
        
        # Annotation
        ymin, ymax = plt.ylim()
        y_pos = ymin + 0.85 * (ymax - ymin)
        remark = pick.get('remark', '')
        label_text = f"M{pick['phase']}" + (f' ({remark})' if remark else f' (conf: {pick["confidence"]:.1f})')
        plt.text(pick_mpl, y_pos, label_text, 
                ha='center', va='bottom', fontweight='bold', color=color, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    # Plot predicted picks (dashed lines) - these should already be in absolute time
    predicted_labels_used = set()
    for pick in predicted_picks:
        if (pick['file_name'] != os.path.basename(npz_filename) or 
            pick['config'] != config_name):
            continue
            
        pick_mpl = mdates.date2num(pick['phase_time'].datetime)
        color = colors.get(pick['phase_type'], 'green')
        
        # Verify pick time is within waveform time range
        if pick_mpl < mpl_times[0] or pick_mpl > mpl_times[-1]:
            logging.warning(f"Predicted {pick['phase_type']} pick at {pick['phase_time']} is outside waveform time range")
            continue
        
        # Vertical line for pick
        label = f"Predicted {pick['phase_type']}"
        plt.axvline(x=pick_mpl, color=color, linestyle='--', 
                   linewidth=2.5, alpha=0.9, label=label if label not in predicted_labels_used else "")
        predicted_labels_used.add(label)
        
        # Annotation
        ymin, ymax = plt.ylim()
        y_pos = ymin + 0.8 * (ymax - ymin)
        plt.text(pick_mpl, y_pos, f"P{pick['phase_type']} ({pick['phase_score']:.3f})", 
                ha='center', va='bottom', fontweight='bold', color=color, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
    
    plt.xlabel('Time (UTC)', fontsize=12)
    plt.title(f'{station_id} - {config_name.upper()} - Manual vs Predicted Picks - {comp_name} Component\nFile: {os.path.basename(npz_filename)}', 
              fontsize=14, fontweight='bold')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.grid(True, alpha=0.3)
    
    # Format x-axis for time display
    date_format = mdates.DateFormatter('%H:%M:%S')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    filename = f"{config_name}_{station_id}_{comp_name}_picks_{os.path.basename(npz_filename).replace('.npz', '')}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved plot: {save_path}")
    return True

def process_npz_file_for_config(npz_file, manual_picks, all_predicted_picks, config_name, 
                               sampling_rate=100, output_dir="picks_comparison"):
    """Process a single NPZ file for a specific configuration"""
    station_id = extract_station_from_filename(npz_file)
    
    logging.info(f"Processing {config_name}: {os.path.basename(npz_file)} - Station: {station_id}")
    
    waveform_data = load_waveform_data(npz_file)
    if waveform_data is None:
        return 0
    
    start_time = get_waveform_timing_info(npz_file)
    if start_time is None:
        config_predicted = [p for p in all_predicted_picks if p['config'] == config_name]
        station_predicted = [p for p in config_predicted if p['file_name'] == os.path.basename(npz_file)]
        if station_predicted:
            start_time = station_predicted[0]['begin_time']
            logging.info(f"Using start time from predicted picks: {start_time}")
        else:
            logging.error(f"Could not determine start time for {npz_file}")
            return 0
    
    component_names = ['Z', 'N', 'E']
    waveform_components = []
    
    if waveform_data.ndim == 3 and waveform_data.shape[1] == 1 and waveform_data.shape[2] == 3:
        for i in range(3):
            component_data = waveform_data[:, 0, i]
            waveform_components.append(component_data)
    elif waveform_data.ndim == 2 and waveform_data.shape[1] == 3:
        for i in range(3):
            component_data = waveform_data[:, i]
            waveform_components.append(component_data)
    else:
        logging.warning(f"Unexpected waveform shape: {waveform_data.shape}")
        if waveform_data.ndim > 1:
            waveform_plot = waveform_data[:, 0]
        else:
            waveform_plot = waveform_data
        waveform_components = [waveform_plot]
        component_names = ['Component']
    
    success_count = 0
    for comp_idx, (component_data, comp_name) in enumerate(zip(waveform_components, component_names)):
        success = create_component_plot(
            npz_filename=npz_file,
            station_id=station_id,
            component_data=component_data,
            comp_name=comp_name,
            manual_picks=manual_picks,
            predicted_picks=all_predicted_picks,
            start_time=start_time,
            sampling_rate=sampling_rate,
            output_dir=output_dir,
            config_name=config_name
        )
        
        if success:
            success_count += 1
    
    return success_count

def main():
    parser = argparse.ArgumentParser(
        description='Plot manual vs predicted phase picks on waveform components for ALL model configurations'
    )
    
    parser.add_argument('--results_dir', required=True, 
                       help='Directory containing model results (should have baseline/, augmentation/, etc. subdirectories)')
    parser.add_argument('--manual_picks', required=True,
                       help='CSV file with manual picks (npz_data_list_detailed.csv)')
    parser.add_argument('--data_dir', required=True,
                       help='Directory containing NPZ waveform files')
    parser.add_argument('--output_dir', default='all_picks_comparison',
                       help='Output directory for all plots')
    parser.add_argument('--sampling_rate', type=float, default=100,
                       help='Sampling rate in Hz (default: 100)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of NPZ files to process per config')
    parser.add_argument('--configs', nargs='+', default=CONFIGS,
                       help=f'Model configurations to process (default: {CONFIGS})')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        logging.error(f"Results directory not found: {args.results_dir}")
        return
    
    if not os.path.exists(args.manual_picks):
        logging.error(f"Manual picks file not found: {args.manual_picks}")
        return
    
    if not os.path.exists(args.data_dir):
        logging.error(f"Data directory not found: {args.data_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("Loading manual picks...")
    manual_picks = load_manual_picks(args.manual_picks)
    
    logging.info("Loading predicted picks for all configurations...")
    all_predicted_picks = []
    
    for config_name in args.configs:
        picks_file = os.path.join(args.results_dir, config_name, 'picks.csv')
        config_picks = load_predicted_picks(picks_file, config_name)
        all_predicted_picks.extend(config_picks)
        logging.info(f"✓ Loaded {len(config_picks)} picks for {config_name}")
    
    npz_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.npz')]
    
    if args.max_files:
        npz_files = npz_files[:args.max_files]
    
    logging.info(f"Found {len(npz_files)} NPZ files to process")
    
    total_plots_created = 0
    config_summary = {}
    
    for config_name in args.configs:
        logging.info(f"\n{'='*60}")
        logging.info(f"PROCESSING CONFIGURATION: {config_name.upper()}")
        logging.info(f"{'='*60}")
        
        config_plots_created = 0
        config_successful_files = 0
        
        for npz_file in npz_files:
            plots_created = process_npz_file_for_config(
                npz_file=npz_file,
                manual_picks=manual_picks,
                all_predicted_picks=all_predicted_picks,
                config_name=config_name,
                sampling_rate=args.sampling_rate,
                output_dir=args.output_dir
            )
            
            if plots_created > 0:
                config_successful_files += 1
                config_plots_created += plots_created
        
        config_summary[config_name] = {
            'plots_created': config_plots_created,
            'successful_files': config_successful_files,
            'total_files': len(npz_files)
        }
        
        total_plots_created += config_plots_created
        logging.info(f"✓ {config_name}: {config_plots_created} plots created for {config_successful_files}/{len(npz_files)} files")
    
    logging.info(f"\n{'='*80}")
    logging.info("ALL CONFIGURATIONS PROCESSING COMPLETE")
    logging.info(f"{'='*80}")
    
    for config_name, summary in config_summary.items():
        logging.info(f"{config_name:15} : {summary['plots_created']:4d} plots for {summary['successful_files']:3d}/{summary['total_files']:3d} files")
    
    logging.info(f"{'Total':15} : {total_plots_created:4d} plots created")
    logging.info(f"Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()