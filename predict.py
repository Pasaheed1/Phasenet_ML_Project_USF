import argparse
import logging
import multiprocessing
import os
import pickle
import time
from functools import partial

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from data_reader import DataReader_mseed_array, DataReader_pred
from model import ModelConfig, UNet
from postprocess import (
    extract_amplitude,
    extract_picks,
    save_picks,
    save_picks_json,
    save_prob_h5,
)
from tqdm import tqdm
from visulization import plot_waveform

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime
from datetime import datetime
import numpy as np

def load_manual_picks_npz_format(manual_picks_file):
    """
    Load manual picks from your NPZ detailed CSV format
    """
    try:
        if not os.path.exists(manual_picks_file):
            logging.warning(f"Manual picks file not found: {manual_picks_file}")
            return None
            
        manual_df = pd.read_csv(manual_picks_file, sep='\t')  # Tab-separated
        logging.info(f"Loaded {len(manual_df)} manual picks from {manual_picks_file}")
        
        # Convert to standard format for comparison
        manual_picks_list = []
        
        for _, row in manual_df.iterrows():
            station = row['station']
            network = row['network']
            location = row.get('location_code', '')
            fname = row['fname']
            dt = row.get('dt', 0.01)
            
            # Add P pick if exists
            if pd.notna(row['p_time']) and row['p_time'] != '':
                try:
                    p_idx = row.get('p_idx', 0)
                    manual_picks_list.append({
                        'station': station,
                        'network': network,
                        'location': location,
                        'phase': 'P',
                        'pick_time': row['p_time'],
                        'pick_time_utc': UTCDateTime(row['p_time']),
                        'pick_time_datetime': UTCDateTime(row['p_time']).datetime,
                        'confidence': row.get('p_weight', 1.0),
                        'remark': row.get('p_remark', ''),
                        'index': row.get('p_idx', 0),
                        'dt': dt
                    })
                except Exception as e:
                    print(f"Debug: Error processing manual S-pick {row}: {e}") 
            
            # Add S pick if exists
            if pd.notna(row['s_time']) and row['s_time'] != '':
                try:
                    s_idx = row.get('s_idx', 0)
                    manual_picks_list.append({
                        'station': station,
                        'network': network,
                        'location': location,
                        'phase': 'S',
                        'pick_time': row['s_time'],
                        'pick_time_utc': UTCDateTime(row['s_time']),
                        'pick_time_datetime': UTCDateTime(row['s_time']).datetime,
                        'confidence': row.get('s_weight', 1.0),
                        'remark': row.get('s_remark', ''),
                        'index': row.get('s_idx', 0),
                        'dt': dt
                    })
                except Exception as e:
                    print(f"Debug: Error processing manual S-pick {row}: {e}")
        
        manual_picks_df = pd.DataFrame(manual_picks_list)
        logging.info(f"Processed {len(manual_picks_df)} manual phase picks")
        return manual_picks_df
        
    except Exception as e:
        logging.warning(f"Could not load manual picks: {e}")
        return None
    
def debug_comparison_setup(args, picks_):
    """Debug function to see why comparison plots aren't generating"""
    print(f"\n=== DEBUG COMPARISON SETUP ===")
    print(f"compare_picks flag: {args.compare_picks}")
    print(f"manual_picks_file: {args.manual_picks_file}")
    print(f"comparison_dir: {args.comparison_dir}")
    print(f"Number of picks in batch: {len(picks_)}")
    
    if args.compare_picks and len(picks_) > 0:
        print("✓ Comparison plotting should be active")
        # Check if manual picks file exists
        if os.path.exists(args.manual_picks_file):
            print(f"✓ Manual picks file exists: {args.manual_picks_file}")
            manual_picks_df = load_manual_picks_npz_format(args.manual_picks_file)
            if manual_picks_df is not None:
                print(f"✓ Loaded {len(manual_picks_df)} manual picks")
                unique_stations = set([pick['station_id'] for pick in picks_])
                print(f"Stations in batch: {unique_stations}")
            else:
                print("Failed to load manual picks")
        else:
            print(f"✗ Manual picks file not found: {args.manual_picks_file}")
    else:
        print("✗ Comparison plotting not active (check flags or empty picks)")
    print("=== END DEBUG ===\n")

def create_comparison_plot_for_station(waveform_data, times, station_id, predicted_picks, manual_picks_df=None, 
                                     figure_dir="comparison_figures", sampling_rate=100, filename=None,
                                     begin_time=None):
    """
    Create comparison plot for a specific station
    """
    try:
        # Create figure directory
        os.makedirs(figure_dir, exist_ok=True)
        print(f"DEBUG: Creating plot for station {station_id}")
        
        # Convert times to matplotlib format
        if isinstance(times[0], UTCDateTime):
            mpl_times = mdates.date2num([t.datetime for t in times])
            time_label = "Time (UTC)"
        elif isinstance(times[0], datetime):
            mpl_times = mdates.date2num(times)
            time_label = "Time (UTC)"
        else:
            # Assume numeric times (samples or seconds)
            mpl_times = np.arange(len(waveform_data)) / sampling_rate
            time_label = "Time (seconds)"
        
        print(f"Debug: Original waveform shape {waveform_data.shape}")
        
        # Process waveform data for all three components
        waveform_components = []
        component_names = ['Z', 'N', 'E']
        if waveform_data.ndim ==3 and waveform_data.shape[1] == 1 and waveform_data.shape[2] == 3:
            # Extract all three components
            for i in range(3):
                component_data = waveform_data[:, 0, i]
                if component_data.ndim > 1:
                    component_data = component_data.flatten()
                waveform_components.append(component_data)
                print(f"Debug: Component {component_names[i]} shape: {component_data.shape}")
        else:
            # If not 3-component data, use what we have
            if waveform_data.ndim > 1:
                waveform_plot = waveform_data.flatten()
            else:
                waveform_plot = waveform_data
            waveform_components = [waveform_plot]
            component_names = ['Waveform']
            print(f"DEBUG: Using single component, shape: {waveform_plot.shape}")
                   
        # Organize predicted picks by phase for this station
        pred_picks_dict = {}
        for pick in predicted_picks:
            
            pick_station_id = pick.get('station_id', '')
            if pick_station_id != station_id:
                continue
            
            if 'phase_type' in pick:
                phase_type = pick['phase_type']
            elif 'type' in pick:
                phase_type = pick['type']
            else:
                print(f"Debug: Pick missing phase_type: {pick}")
                continue
            
            try:
                pick_time = UTCDateTime(pick['phase_time'])
                # Store the picks with its information
                pred_picks_dict[f"{phase_type}_{pick.get('phase_index', 0)}"] ={
                    'phase_type': phase_type,
                    'datetime': UTCDateTime(pick['phase_time']).datetime,
                    'utc': UTCDateTime(pick['phase_time']),
                    'score': pick['phase_score'],
                    'index': pick.get('phase_index', 0)
                }
                
                print(f"DEBUG: Added {phase_type} pick for {station_id} at {pick_time}")
            except Exception as e:
                print(f"Debug: Error processing pick {pick}: {e}")
                        
        # Now group by phase type, keeping only the highest score for each phase
        final_pred_picks = {}
        for pick_key, pick_info in pred_picks_dict.items():
            phase_type = pick_info['phase_type']
            if (phase_type not in final_pred_picks or 
                pick_info['score'] > final_pred_picks[phase_type]['score']):
                final_pred_picks[phase_type] = pick_info
        
        pred_picks_dict = final_pred_picks
        print(f"DEBUG: Final predicted picks for {station_id}: {list(pred_picks_dict.keys())}")

        # Get manual picks for this station
        manual_picks_dict = {}
        if manual_picks_df is not None:
            station_manual_picks = manual_picks_df[manual_picks_df['station'] == station_id]
            print(f"DEBUG: Found {len(station_manual_picks)} manual picks for station {station_id}")
            for _, row in station_manual_picks.iterrows():
                phase_type = row['phase']
                try:
                    pick_time = row['pick_time_utc']
                    manual_picks_dict[phase_type] = {
                        'datetime': row['pick_time_datetime'],
                        'utc': row['pick_time_utc'],
                        'score': row.get('confidence', 1.0),
                        'index': row.get('index', 0),
                        'remark': row.get('remark', '')
                    }
                    
                    print(f"DEBUG: Added manual {phase_type} pick for {station_id} at {pick_time}")
                except Exception as e:
                    print(f"DEBUG: Error processing manual pick {row}: {e}")
                    continue
        
        print(f"DEBUG: Manual picks for {station_id}: {list(manual_picks_dict.keys())}")
                
        # Only create plot if we have either manual or predicted picks
        if not manual_picks_dict and not pred_picks_dict:
            print(f"DEBUG: No picks found for station {station_id}, skipping plot")
            return False
        
        colors = {'P': 'red', 'S': 'blue'}
        success_count = 0
        
        for comp_idx, (component_data, comp_name) in enumerate(zip(waveform_components, component_names)):
            
            # Create the plot for this component
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # Plot 1: Manual picks
            ax1.plot(mpl_times, component_data, 'k-', linewidth=1.0, alpha=0.7, label=f'Waveform{comp_name}')
            
            for phase, pick_info in manual_picks_dict.items():
                if 'datetime' in pick_info:
                    pick_mpl = mdates.date2num(pick_info['datetime'])
                else:
                    # Fallback to sample-based timing
                    pick_mpl = pick_info.get('index', 0) / sampling_rate
            
                color = colors.get(phase, 'green')
                ax1.axvline(x=pick_mpl, color=color, linestyle='-',
                            linewidth=3, alpha=0.9, label=f'Manual {phase}')
            
                # Add annotation
                ymin, ymax = ax1.get_ylim()
                y_range = ymax - ymin
                y_pos = ymin + 0.85 * (y_range)
                remark = pick_info.get('remark', '')
                label = f'M{phase}' + (f' ({remark})' if remark else '')
                ax1.text(pick_mpl, y_pos, label,
                         ha='center', va='bottom', fontweight='bold', color=color,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
            ax1.set_ylabel(f'Amplitude ({comp_name})', fontsize=12)
            ax1.set_title(f'{station_id} - Manual Picks - {comp_name} Component', fontsize=12, fontweight='bold')
            if manual_picks_dict:
                ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
        
            # Plot 2: Predicted picks
            ax2.plot(mpl_times, component_data, 'k-', linewidth=1.0, alpha=0.7, label=f'Waveform {comp_name}')
        
            for phase, pick_info in pred_picks_dict.items():
                if 'datetime' in pick_info:
                    pick_mpl = mdates.date2num(pick_info['datetime'])
                else:
                    # Fallback to sample-based timing
                    pick_mpl = pick_info.get('index', 0) / sampling_rate
            
                color = colors.get(phase, 'green')
                ax2.axvline(x=pick_mpl, color=color, linestyle='-',
                            linewidth=2, alpha=0.9, label=f'Predicted {phase}')
            
                # Add annotation
                ymin, ymax = ax2.get_ylim()
                y_range = ymax - ymin
                y_pos = ymin + 0.85 * (y_range)
                ax2.text(pick_mpl, y_pos, f'P{phase} ({pick_info["score"]:.2f})',
                         ha='center', va='bottom', fontweight='bold', color=color, fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
            ax2.set_xlabel(time_label, fontsize=12)
            ax2.set_ylabel('Amplitude', fontsize=12)
            ax2.set_title(f'{station_id} - Predicted Picks - {comp_name} Component', fontsize=12, fontweight='bold')
            if pred_picks_dict:
                ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        
            # Format x-axis if using datetime
            if isinstance(times[0], (UTCDateTime, datetime)):
                date_format = mdates.DateFormatter('%H:%M:%S')
                ax2.xaxis.set_major_formatter(date_format)
                plt.xticks(rotation=45)
        
            plt.tight_layout()
        
            # Save figure
            if filename is None:
                comp_filename = f"{station_id}_comparison_{comp_name}.png"
            else:
                name, ext = os.path.splitext(filename)
                comp_filename = f"{name}_{comp_name}{ext}"
            save_path = os.path.join(figure_dir, comp_filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
            print (f"Debug: Saved comparison plot: {save_path}")
            success_count += 1
        return success_count > 0
       
    except Exception as e:
        print(f"Debug: Error creating comparison plot for {station_id}: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False

def plot_comparison_overlay_for_station(waveform_data, times, station_id, predicted_picks, manual_picks_df=None,
                                      figure_dir="comparison_figures", sampling_rate=100, filename=None):
    """
    Create overlay comparison plot on single figure
    """
    try:
        os.makedirs(figure_dir, exist_ok=True)
        
        # Convert times
        if isinstance(times[0], UTCDateTime):
            mpl_times = mdates.date2num([t.datetime for t in times])
            time_label = "Time (UTC)"
        elif isinstance(times[0], datetime):
            mpl_times = mdates.date2num(times)
            time_label = "Time (UTC)"
        else:
            mpl_times = np.arange(len(waveform_data)) / sampling_rate
            time_label = "Time (seconds)"
        
        # Process waveform components
        waveform_components = []
        component_names = ['Z', 'N', 'E']
        
        if waveform_data.ndim == 3 and waveform_data.shape[1] == 1 and waveform_data.shape[2] == 3:
            for i in range(3):
                component_data = waveform_data[:, 0, i]
                if component_data.ndim > 1:
                    component_data = component_data.flatten()
                waveform_components.append(component_data)
        else:
            if waveform_data.ndim > 1:
                waveform_plot = waveform_data.flatten()
            else:
                waveform_plot = waveform_data
            waveform_components = [waveform_plot]
            component_names = ['Waveform']
        
        # Organize picks
        pred_picks_dict = {}
        for pick in predicted_picks:
            if pick['station_id'] == station_id:
                phase_type = pick['phase_type']
                pred_picks_dict[phase_type] = {
                    'datetime': UTCDateTime(pick['phase_time']).datetime,
                    'utc': UTCDateTime(pick['phase_time']),
                    'score': pick['phase_score'],
                    'index': pick.get('phase_index', 0)
                }
        
        manual_picks_dict = {}
        if manual_picks_df is not None:
            station_manual_picks = manual_picks_df[manual_picks_df['station'] == station_id]
            for _, row in station_manual_picks.iterrows():
                phase_type = row['phase']
                manual_picks_dict[phase_type] = {
                    'datetime': row['pick_time_datetime'],
                    'utc': row['pick_time_utc'],
                    'score': row.get('confidence', 1.0),
                    'index': row.get('index', 0),
                    'remark': row.get('remark', '')
                }
        
        success_count = 0
        colors = {'P': 'red', 'S': 'blue'}
        
        # Create overlay plot
        for comp_idx, (component_data, comp_name) in enumerate(zip(waveform_components, component_names)):
            plt.figure(figsize=(15, 8))
            plt.plot(mpl_times, component_data, 'k-', linewidth=1.0, alpha=0.7, label=f'Waveform {comp_name}')
            
            # Plot manual picks (solid lines)
            for phase, pick_info in manual_picks_dict.items():
                if 'datetime' in pick_info:
                    pick_mpl = mdates.date2num(pick_info['datetime'])
                else:
                    pick_mpl = pick_info.get('index', 0) / sampling_rate
            
                color = colors.get(phase, 'green')
                remark = pick_info.get('remark', '')
                label = f'Manual {phase}' + (f' ({remark})' if remark else '')
                plt.axvline(x=pick_mpl, color=color, linestyle='-', 
                            linewidth=2, alpha=0.9, label=label)
        
            # Plot predicted picks (dashed lines)
            for phase, pick_info in pred_picks_dict.items():
                if 'datetime' in pick_info:
                    pick_mpl = mdates.date2num(pick_info['datetime'])
                else:
                    pick_mpl = pick_info.get('index', 0) / sampling_rate
            
                color = colors.get(phase, 'green')
                plt.axvline(x=pick_mpl, color=color, linestyle='--',
                            linewidth=2.5, alpha=0.9, label=f'Predicted {phase} ({pick_info["score"]:.2f})')
        
            plt.xlabel(time_label, fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.title(f'{station_id} - Manual vs Predicted Picks - {comp_name}Component', fontsize=14, fontweight='bold')
            plt.legend(loc='upper right', fontsize=8)
            plt.grid(True, alpha=0.3)
        
            if isinstance(times[0], (UTCDateTime, datetime)):
                date_format = mdates.DateFormatter('%H:%M:%S')
                plt.gca().xaxis.set_major_formatter(date_format)
                plt.xticks(rotation=45)
        
            plt.tight_layout()
        
            if filename is None:
                comp_filename = f"{station_id}_overlay_{comp_name}.png"
            else:
                name, ext = os.path.splitext(filename)
                comp_filename = f"{name}_{comp_name}{ext}"
            save_path = os.path.join(figure_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
            logging.info(f"Saved overlay plot: {save_path}")
            success_count += 1
        return success_count > 0
        
    except Exception as e:
        logging.error(f"Error creating overlay plot for {station_id}: {e}")
        plt.close('all')
        return False

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def verify_pick_timing(station_id, manual_picks_df, predicted_picks, begin_time, sampling_rate):
    """Debug function to verify pick timing alignment"""
    print(f"\n=== TIME VERIFICATION FOR {station_id} ===")
    print(f"Begin time: {begin_time}")
    print(f"Sampling rate: {sampling_rate} Hz")
    
    # Manual picks for this station
    station_manual = manual_picks_df[manual_picks_df['station'] == station_id]
    print(f"Manual picks found: {len(station_manual)}")
    for _, manual in station_manual.iterrows():
        phase = manual['phase']
        manual_time = manual['pick_time_utc']
        manual_idx = manual['index']
        calculated_time = begin_time + (manual_idx / sampling_rate)
        print(f"Manual {phase}: Index={manual_idx}, Time={manual_time}, Calculated={calculated_time}")
    
    # Predicted picks for this station
    station_predicted = [p for p in predicted_picks if p.get('station_id') == station_id]
    print(f"Predicted picks found: {len(station_predicted)}")
    for pred in station_predicted:
        phase = pred['phase_type']
        pred_time = UTCDateTime(pred['phase_time'])
        pred_idx = pred.get('phase_index', 0)
        calculated_time = begin_time + (pred_idx / sampling_rate)
        print(f"Predicted {phase}: Index={pred_idx}, Time={pred_time}, Calculated={calculated_time}")
    
    print("=== END TIME VERIFICATION ===\n")

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="batch size")
    parser.add_argument("--model_dir", help="Checkpoint directory (default: None)")
    parser.add_argument("--data_dir", default="", help="Input file directory")
    parser.add_argument("--data_list", default="", help="Input csv file")
    parser.add_argument("--hdf5_file", default="", help="Input hdf5 file")
    parser.add_argument("--hdf5_group", default="data", help="data group name in hdf5 file")
    parser.add_argument("--result_dir", default="results", help="Output directory")
    parser.add_argument("--result_fname", default="picks", help="Output file")
    parser.add_argument("--min_p_prob", default=0.3, type=float, help="Probability threshold for P pick")
    parser.add_argument("--min_s_prob", default=0.3, type=float, help="Probability threshold for S pick")
    parser.add_argument("--mpd", default=50, type=float, help="Minimum peak distance")
    parser.add_argument("--amplitude", action="store_true", help="if return amplitude value")
    parser.add_argument("--format", default="numpy", help="input format")
    parser.add_argument("--s3_url", default="localhost:9000", help="s3 url")
    parser.add_argument("--stations", default="", help="seismic station info")
    parser.add_argument("--plot_figure", action="store_true", help="If plot figure for test")
    parser.add_argument("--save_prob", action="store_true", help="If save result for test")
    parser.add_argument("--pre_sec", default=1, type=float, help="Window length before pick")
    parser.add_argument("--post_sec", default=4, type=float, help="Window length after pick")

    parser.add_argument("--highpass_filter", default=0.0, type=float, help="Highpass filter")
    parser.add_argument("--response_xml", default=None, type=str, help="response xml file")
    parser.add_argument("--sampling_rate", default=100, type=float, help="sampling rate")
    parser.add_argument("--compare_picks", action="store_true", help="Generate comparison plots with manual picks")
    parser.add_argument("--manual_picks_file", default="", help="CSV file with manual picks for comparison")
    parser.add_argument("--comparison_dir", default="comparison_plots", help="Directory for comparison plots")
    args = parser.parse_args()

    return args


def pred_fn(args, data_reader, figure_dir=None, prob_dir=None, log_dir=None):
    current_time = time.strftime("%y%m%d-%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(args.log_dir, "pred", current_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if (args.plot_figure == True) and (figure_dir is None):
        figure_dir = os.path.join(log_dir, "figures")
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
    if (args.save_prob == True) and (prob_dir is None):
        prob_dir = os.path.join(log_dir, "probs")
        if not os.path.exists(prob_dir):
            os.makedirs(prob_dir)
    if args.save_prob:
        h5 = h5py.File(os.path.join(args.result_dir, "result.h5"), "w", libver="latest")
        prob_h5 = h5.create_group("/prob")
    logging.info("Pred log: %s" % log_dir)
    logging.info("Dataset size: {}".format(data_reader.num_data))

    with tf.compat.v1.name_scope("Input_Batch"):
        if args.format == "mseed_array":
            batch_size = 1
        else:
            batch_size = args.batch_size
        dataset = data_reader.dataset(batch_size)
        batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

    config = ModelConfig(X_shape=data_reader.X_shape)
    with open(os.path.join(log_dir, "config.log"), "w") as fp:
        fp.write("\n".join("%s: %s" % item for item in vars(config).items()))

    model = UNet(config=config, input_batch=batch, mode="pred")
    # model = UNet(config=config, mode="pred")
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    # sess_config.log_device_placement = False

    with tf.compat.v1.Session(config=sess_config) as sess:
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        latest_check_point = tf.train.latest_checkpoint(args.model_dir)
        logging.info(f"restoring model {latest_check_point}")
        saver.restore(sess, latest_check_point)

        picks = []
        amps = [] if args.amplitude else None
        if args.plot_figure:
            multiprocessing.set_start_method("spawn")
            pool = multiprocessing.Pool(multiprocessing.cpu_count())

        for _ in tqdm(range(0, data_reader.num_data, batch_size), desc="Pred"):
            if args.amplitude:
                pred_batch, X_batch, amp_batch, fname_batch, t0_batch, station_batch = sess.run(
                    [model.preds, batch[0], batch[1], batch[2], batch[3], batch[4]],
                    feed_dict={model.drop_rate: 0, model.is_training: False},
                )
            #    X_batch, amp_batch, fname_batch, t0_batch = sess.run([batch[0], batch[1], batch[2], batch[3]])
            else:
                # For non-amplitude case, only fetch the available elements
                if len(batch) >= 5:
                    pred_batch, X_batch, fname_batch, t0_batch, station_batch = sess.run(
                        [model.preds, batch[0], batch[1], batch[2], batch[3]],
                        feed_dict={model.drop_rate: 0, model.is_training: False},
                    )
                    itp_batch, its_batch = None, None
                else:
                    pred_batch, X_batch, fname_batch, t0_batch = sess.run(
                        [model.preds, batch[0], batch[1], batch[2]],
                        feed_dict={model.drop_rate: 0, model.is_training: False},
                    )
                    station_batch, itp_batch, its_batch = None, None, None
            
            #    X_batch, fname_batch, t0_batch = sess.run([model.preds, batch[0], batch[1], batch[2]])
            # pred_batch = []
            # for i in range(0, len(X_batch), 1):
            #     pred_batch.append(sess.run(model.preds, feed_dict={model.X: X_batch[i:i+1], model.drop_rate: 0, model.is_training: False}))
            # pred_batch = np.vstack(pred_batch)

            waveforms = None
            if args.amplitude:
                waveforms = amp_batch

            picks_ = extract_picks(
                preds=pred_batch,
                file_names=fname_batch,
                station_ids=station_batch,
                begin_times=t0_batch,
                config=args,
                waveforms=waveforms,
                use_amplitude=args.amplitude,
                dt=1.0 / args.sampling_rate,
            )
            
            debug_comparison_setup(args, picks_)
           
            picks.extend(picks_)

            if args.compare_picks and len(picks_) > 0:
                print(f"DEBUG: Starting comparison plotting with {len(picks_)} picks")
                try:
                # Load manual picks
                    manual_picks_df = load_manual_picks_npz_format(args.manual_picks_file)
        
                    if manual_picks_df is None:
                        print("DEBUG: No manual picks loaded")
                    else:
                        print(f"DEBUG: Loaded {len(manual_picks_df)} manual picks")
                        
                        # Create comparison directory
                        comparison_dir = os.path.join(args.result_dir, args.comparison_dir)
                        os.makedirs(comparison_dir, exist_ok=True)
                        print(f"DEBUG: Comparison directory: {comparison_dir}")
                        
                        # Process each station in this batch
                        batch_stations = set([pick['station_id'] for pick in picks_])
                        print(f"DEBUG: Stations in batch: {batch_stations}")
                        
                        for station_id in batch_stations:
                            logging.info(f"Creating comparison plots for station: {station_id}")
                            print(f"DEBUG: Processing station {station_id}")
                            
                            # Find the index of this station in the batch
                            station_indices = []
                            if station_batch is not None:
                                station_indices = [i for i, sid in enumerate(station_batch)
                                                   if (sid.decode() if hasattr(sid, 'decode') else sid) == station_id]
                            if not station_indices and len(fname_batch) > 0:
                                print(f'Debug: Station {station_id} not found in station_batch, trying to extract from file names')
                                
                                for i, fname in enumerate(fname_batch):
                                    fname_str = fname.decode() if hasattr(fname, 'decode') else fname
                                    print(f"Debug: Checking file {fname_str} for station {station_id}")
                                    
                                    if station_id in fname_str:
                                        station_indices = [i]
                                        print(f"Debug: Found station {station_id} in filename {fname_str} at index {i}")
                                        break
                                    
                                    elif station_id == '0000':
                                        base_name = os.path.basename(fname_str)
                                        if base_name.endswith('.npz'):
                                            station_part = base_name.replace('.npz', '')
                                            parts = station_part.split('.')
                                            if len(parts) >= 2:
                                                actual_station = parts[1]
                                                print(f"Debug: Extracted station {actual_station} from {base_name}")
                                                station_id = actual_station
                                                station_indices = [i]
                                                break
                            
                            if station_indices:
                                station_idx = station_indices[0]
                                station_waveform = X_batch[station_idx]
                                print(f"DEBUG: Found waveform for {station_id}, shape: {station_waveform.shape}")
                                
                                # Create time array based on begin_time
                                actual_station_name = None
                                if len(fname_batch) > station_idx:
                                    fname = fname_batch[station_idx].decode() if hasattr(fname_batch[station_idx], 'decode') else fname_batch[station_idx]
                                    print(f"Debug: File name for station {station_id}: {fname}")
                                    
                                    base_name = os.path.basename(fname)
                                    if base_name.endswith('.npz'):
                                        station_part = base_name.replace('.npz', '')
                                        parts = station_part.split('.')
                                        if len(parts) >= 2:
                                            actual_station_name = parts[1]
                                            print(f"Debug: Extracted station name: {actual_station_name}")
                                                                
                                if len(t0_batch) > station_idx:
                                    begin_time = UTCDateTime(t0_batch[station_idx])
                                    n_samples = len(station_waveform)
                                    times = [begin_time + i * (1.0/args.sampling_rate) 
                                             for i in range(n_samples)]
                                    print(f"DEBUG: Created time array from {begin_time}, {n_samples} samples")
                                    
                                    plot_station_id = actual_station_name if actual_station_name else station_id
                                    print(f"Debug: Using station ID for plotting: {plot_station_id}")
                                    
                                    # VERIFY TIMING
                                    verify_pick_timing(plot_station_id, manual_picks_df, picks_, begin_time, args.sampling_rate)
                                    
                                    # Create comparison plots
                                    success1 = create_comparison_plot_for_station(
                                        waveform_data=station_waveform,
                                        times=times,
                                        station_id=plot_station_id,
                                        predicted_picks=picks_,
                                        manual_picks_df=manual_picks_df,
                                        figure_dir=comparison_dir,
                                        sampling_rate=args.sampling_rate
                                        )
                                    
                                    success2 = plot_comparison_overlay_for_station(
                                        waveform_data=station_waveform,
                                        times=times,
                                        station_id=plot_station_id,
                                        predicted_picks=picks_,
                                        manual_picks_df=manual_picks_df,
                                        figure_dir=comparison_dir,
                                        sampling_rate=args.sampling_rate
                                        )
                                    
                                    print(f"DEBUG: Plot results - Comparison: {success1}, Overlay: {success2}")
                                else:
                                    print(f"DEBUG: No begin_time for station {station_id}")
                                        
                            else:
                                print(f"DEBUG: No waveform found for station {station_id}")
                    
                except Exception as e:
                    print(f"DEBUG: Error in comparison plotting: {e}")
                    import traceback
                    traceback.print_exc()
                    
            if args.plot_figure:
                if not (isinstance(fname_batch, np.ndarray) or isinstance(fname_batch, list)):
                    fname_batch = [fname_batch.decode().rstrip(".mseed") + "_" + x.decode() for x in station_batch]
                else:
                    fname_batch = [x.decode() for x in fname_batch]
                pool.starmap(
                    partial(
                        plot_waveform,
                        figure_dir=figure_dir,
                        itp_batch=itp_batch,
                        its_batch=its_batch,
                    ),
                    # zip(X_batch, pred_batch, [x.decode() for x in fname_batch]),
                    zip(X_batch, pred_batch, fname_batch),
                )

            if args.save_prob:
                # save_prob(pred_batch, fname_batch, prob_dir=prob_dir)
                if not (isinstance(fname_batch, np.ndarray) or isinstance(fname_batch, list)):
                    fname_batch = [fname_batch.decode().rstrip(".mseed") + "_" + x.decode() for x in station_batch]
                else:
                    fname_batch = [x.decode() for x in fname_batch]
                save_prob_h5(pred_batch, fname_batch, prob_h5)

        if len(picks) > 0:
            # save_picks(picks, args.result_dir, amps=amps, fname=args.result_fname+".csv")
            # save_picks_json(picks, args.result_dir, dt=data_reader.dt, amps=amps, fname=args.result_fname+".json")
            df = pd.DataFrame(picks)
            # df["fname"] = df["file_name"]
            # df["id"] = df["station_id"]
            # df["timestamp"] = df["phase_time"]
            # df["prob"] = df["phase_prob"]
            # df["type"] = df["phase_type"]

            base_columns = [
                "station_id",
                "begin_time",
                "phase_index",
                "phase_time",
                "phase_score",
                "phase_type",
                "file_name",
            ]
            if args.amplitude:
                base_columns.append("phase_amplitude")
                base_columns.append("phase_amp")
                df["phase_amp"] = df["phase_amplitude"]

            df = df[base_columns]
            df.to_csv(os.path.join(args.result_dir, args.result_fname + ".csv"), index=False)

            print(
                f"Done with {len(df[df['phase_type'] == 'P'])} P-picks and {len(df[df['phase_type'] == 'S'])} S-picks"
            )
        else:
            print(f"Done with 0 P-picks and 0 S-picks")
            # Generate comparison plots if requested (for each batch)
            if args.compare_picks and len(picks_) > 0:
                print(f"Debug: Starting comparison plotting with {len(picks_)} picks")
                try:
                    # Load manual picks
                    manual_picks_df = load_manual_picks_npz_format(args.manual_picks_file)
                    
                    if manual_picks_df is None:
                        print("DEBUG: No manual picks loaded")
                    else:
                        print(f"DEBUG: Loaded {len(manual_picks_df)} manual picks")
                        print(f"DEBUG: Manual stations available: {manual_picks_df['station'].unique()}")
                        
                        
                        
                        # Create comparison directory
                        comparison_dir = os.path.join(args.result_dir, args.comparison_dir)
                        os.makedirs(comparison_dir, exist_ok=True)
                        print(f"DEBUG: Comparison directory: {comparison_dir}")
                        
                        # Process each station in this batch
                        batch_stations = set([pick['station_id'] for pick in picks_])
                        print(f"DEBUG: Predicted stations in batch: {batch_stations}")
                        
                        # DEBUG: Print all file names and station batch to understand the mapping
                        print(f"DEBUG: All file names in batch: {fname_batch}")
                        print(f"DEBUG: All station IDs in batch: {[s.decode() if hasattr(s, 'decode') else s for s in station_batch]}")
                    
                        for station_id in batch_stations:
                            logging.info(f"Creating comparison plots for station: {station_id}")
                            
                            # Use the actual waveform data from this batch
                            # Find the index of this station in the batch
                            station_indices = [i for i, sid in enumerate (station_batch)
                                               if (sid.decode() if hasattr(sid, 'decode') else sid) == station_id]
                            if station_indices:
                                station_idx = station_indices[0]
                                station_waveform = X_batch[station_idx]
                                print(f"Debug: Found waveform for {station_id}, shape: {station_waveform.shape}")
                                
                                # Use the actual waveform data and Create time array based on begin_time
                                
                                actual_station_name = None
                                if len(fname_batch) > station_idx:
                                    fname = fname_batch[station_idx].decode() if hasattr(fname_batch[station_idx], 'decode') else fname_batch[station_idx]
                                    print(f"DEBUG: File name for station {station_id}: {fname}")
                                    
                                    if '.' in fname:
                                        parts = fname.split('.')
                                        
                                        if len(parts) >= 2:
                                            actual_station_name = parts[1]
                                            print(f"Debug:Extracted station name: {actual_station_name} from {fname}")
                                        else:
                                            base_name = os.path.splitext(os.path.basename(fname))[0]
                                            
                                            import re
                                            station_match = re.search(r'[A-Z]{3,4}', base_name)
                                            
                                            if station_match:
                                                actual_station_name = station_match.group()
                                                print(f"Debug: Extracted station name via regex: {actual_station_name}")
                                            
                                if len(t0_batch) > station_idx:
                                    begin_time = UTCDateTime(t0_batch[station_idx])
                                    n_samples = len(station_waveform)
                                    times = [begin_time + i * (1.0/args.sampling_rate)
                                             for i in range(n_samples)]
                                    print(f"DEBUG: Created time array from {begin_time}, {n_samples} samples")
                                    
                                    # Use actual station name if found, otherwise use numeric ID
                                    plot_station_id = actual_station_name if actual_station_name else station_id
                                    print(f"DEBUG: Using station ID for plotting: {plot_station_id}")
                                    
                                    # Check if we have manual picks for this station
                                    if manual_picks_df is not None:
                                        manual_picks_for_station = manual_picks_df[manual_picks_df['station'] == plot_station_id]
                                        print(f"DEBUG: Found {len(manual_picks_for_station)} manual picks for station {plot_station_id}")
                                    
                                # Create comparison plots
                                success1 = create_comparison_plot_for_station(
                                    waveform_data=station_waveform,
                                    times=times,
                                    station_id=plot_station_id,
                                    predicted_picks=picks_,
                                    manual_picks_df=manual_picks_df,
                                    figure_dir=comparison_dir,
                                    sampling_rate=args.sampling_rate
                                )
                                
                                success2 = plot_comparison_overlay_for_station(
                                    waveform_data=station_waveform,
                                    times=times,
                                    station_id=station_id,
                                    predicted_picks=picks_,
                                    manual_picks_df=manual_picks_df,
                                    figure_dir=comparison_dir,
                                    sampling_rate=args.sampling_rate
                                )
                                
                                print(f"DEBUG: Plot results - Comparison: {success1}, Overlay: {success2}")
                                    
                                if success1 or success2:
                                    print(f"DEBUG: ✓ Successfully created plots for {plot_station_id}")
                                else:
                                    print(f"DEBUG: ✗ Failed to create plots for {plot_station_id}")
                            else:
                                print(f"DEBUG: No begin_time for station {station_id}")
                        else:
                            print(f"DEBUG: No waveform found for station {station_id}")
                    
                except Exception as e:
                    print(f"DEBUG: Error in comparison plotting: {e}")
                    import traceback
                    traceback.print_exc()

def test_plotting_functionality():
    """Test if matplotlib plotting and file saving works"""
    print("=== TESTING PLOTTING FUNCTIONALITY ===")
    try:
        # Create a simple test plot
        plt.figure(figsize=(10, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3], 'b-o')
        plt.title("Test Plot - Matplotlib Functionality")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        
        test_dir = "test_plots"
        os.makedirs(test_dir, exist_ok=True)
        test_path = os.path.join(test_dir, "test_plot.png")
        plt.savefig(test_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_path):
            print(f"✓ Plotting test PASSED - file saved: {test_path}")
            return True
        else:
            print("✗ Plotting test FAILED - file not created")
            return False
            
    except Exception as e:
        print(f"✗ Plotting test FAILED with error: {e}")
        return False

def main(args):
    # Test plotting functionality first
    if not test_plotting_functionality():
        print("WARNING: Plotting functionality test failed!")
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    with tf.compat.v1.name_scope("create_inputs"):
        if args.format == "mseed_array":
            data_reader = DataReader_mseed_array(
                data_dir=args.data_dir,
                data_list=args.data_list,
                stations=args.stations,
                amplitude=args.amplitude,
                highpass_filter=args.highpass_filter,
            )
        else:
            data_reader = DataReader_pred(
                format=args.format,
                data_dir=args.data_dir,
                data_list=args.data_list,
                hdf5_file=args.hdf5_file,
                hdf5_group=args.hdf5_group,
                amplitude=args.amplitude,
                highpass_filter=args.highpass_filter,
                response_xml=args.response_xml,
                sampling_rate=args.sampling_rate,
            )

        pred_fn(args, data_reader, log_dir=args.result_dir)

    return


if __name__ == "__main__":
    args = read_args()
    main(args)
