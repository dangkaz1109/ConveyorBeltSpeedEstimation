import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_telemetry_csv(csv_path, output_plot_path=None):
    """
    Read telemetry CSV data, calculate statistics, and plot performance analysis charts.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Validate and sanitize paths
    csv_path = os.path.abspath(csv_path)
    if output_plot_path:
        output_plot_path = os.path.abspath(output_plot_path)

    # Read data
    df = pd.read_csv(csv_path)
    
    # Define blocking components in main loop
    blocking_cols = {
        't_frame_io': 'Frame IO/Resize',
        't_depth_copy': 'Depth Lock & Copy',
        't_speed_total': 'Speed Engine (Total)',
        't_kalman_filter': 'Kalman Filter',
        't_vis_io': 'Vis Overlay & Video Write'
    }
    
    # Define Speed Engine subcomponents
    speed_engine_cols = {
        't_speed_preprocess': 'Preprocess Image',
        't_speed_raft_inference': 'RAFT Flow Inference',
        't_speed_flow_grid': 'Flow Grid & Sample',
        't_speed_3d_projection': '3D Projection',
        't_speed_filter': 'Speed IQR Filter'
    }

    # Print statistics report
    print("\n" + "="*70)
    print(" PERFORMANCE STATISTICS REPORT (Unit: ms)")
    print("="*70)
    
    total_avg_time = df['t_total_frame'].mean() * 1000.0
    print(f"Average frame processing time: {total_avg_time:.2f} ms ({1000.0/total_avg_time:.1f} FPS)")
    print("-"*70)
    print(f"{'Component':<30} | {'Mean':<10} | {'Std':<8} | {'Min':<8} | {'Max':<8} | {'%':<8}")
    print("-"*70)
    
    # Print blocking components statistics
    for col, name in blocking_cols.items():
        if col in df.columns:
            vals = df[col] * 1000.0
            pct = (vals.mean() / total_avg_time) * 100
            print(f"{name:<30} | {vals.mean():8.2f} | {vals.std():8.2f} | {vals.min():8.2f} | {vals.max():8.2f} | {pct:6.1f}%")
            
    # Add async depth inference info
    if 't_depth_inference' in df.columns:
        vals = df['t_depth_inference'] * 1000.0
        print(f"{'Depth Model Inference (Async)':<30} | {vals.mean():8.2f} | {vals.std():8.2f} | {vals.min():8.2f} | {vals.max():8.2f} | (N/A Thread)")
        
    print("-"*70)
    print("Speed Engine subcomponents:")
    if 't_speed_total' in df.columns:
        speed_engine_total_mean = df['t_speed_total'].mean() * 1000.0
        for col, name in speed_engine_cols.items():
            if col in df.columns:
                vals = df[col] * 1000.0
                pct_of_engine = (vals.mean() / speed_engine_total_mean) * 100
                pct_of_total = (vals.mean() / total_avg_time) * 100
                print(f" - {name:<27} | {vals.mean():8.2f} | {vals.std():8.2f} | {vals.min():8.2f} | {vals.max():8.2f} | {pct_of_total:5.1f}% ({pct_of_engine:.1f}% engine)")
    print("="*70 + "\n")

    # Plot charts using matplotlib
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"PERFORMANCE ANALYSIS (Conveyor Belt Speed Estimation)\nFile: {os.path.basename(csv_path)}", fontsize=16, fontweight='bold')

    frames = df['frame_idx']

    # 1. Stacked area chart for pipeline components
    pipeline_data = {}
    for col, name in blocking_cols.items():
        if col in df.columns:
            pipeline_data[name] = df[col] * 1000.0  # ms

    # Calculate overhead
    sum_measured = np.sum([pipeline_data[name] for name in pipeline_data], axis=0)
    overhead = (df['t_total_frame'] * 1000.0) - sum_measured
    overhead = np.clip(overhead, 0, None)
    pipeline_data['Overhead / Sync'] = overhead

    axs[0, 0].stackplot(frames, pipeline_data.values(), labels=pipeline_data.keys(), alpha=0.85)
    axs[0, 0].plot(frames, df['t_total_frame'] * 1000.0, color='black', linestyle='--', linewidth=1.5, label='Total Frame Time')
    axs[0, 0].set_title("Pipeline processing time per frame (ms)", fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel("Frame index", fontsize=10)
    axs[0, 0].set_ylabel("Time (ms)", fontsize=10)
    axs[0, 0].legend(loc='upper left')
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Stacked area chart for Speed Engine
    speed_engine_data = {}
    for col, name in speed_engine_cols.items():
        if col in df.columns:
            speed_engine_data[name] = df[col] * 1000.0  # ms
            
    if speed_engine_data:
        axs[0, 1].stackplot(frames, speed_engine_data.values(), labels=speed_engine_data.keys(), alpha=0.85)
        if 't_speed_total' in df.columns:
            axs[0, 1].plot(frames, df['t_speed_total'] * 1000.0, color='darkred', linestyle='--', linewidth=1.5, label='Total Speed Engine')
        axs[0, 1].set_title("Speed Engine processing time (ms)", fontsize=12, fontweight='bold')
        axs[0, 1].set_xlabel("Frame index", fontsize=10)
        axs[0, 1].set_ylabel("Time (ms)", fontsize=10)
        axs[0, 1].legend(loc='upper left')
        axs[0, 1].grid(True, alpha=0.3)

    # 3. Bar chart comparing average times of components
    labels = []
    means = []
    stds = []
    
    # Blocking components
    for col, name in blocking_cols.items():
        if col in df.columns:
            labels.append(name)
            means.append(df[col].mean() * 1000.0)
            stds.append(df[col].std() * 1000.0)
            
    # Add async depth
    if 't_depth_inference' in df.columns:
        labels.append('Depth Model Inference (Async)')
        means.append(df['t_depth_inference'].mean() * 1000.0)
        stds.append(df['t_depth_inference'].std() * 1000.0)
        
    y_pos = np.arange(len(labels))
    colors = ['teal'] * len(blocking_cols)
    if 't_depth_inference' in df.columns:
        colors.append('orange')
        
    axs[1, 0].barh(y_pos, means, xerr=stds, align='center', alpha=0.7, color=colors, capsize=5)
    axs[1, 0].set_yticks(y_pos)
    axs[1, 0].set_yticklabels(labels)
    axs[1, 0].invert_yaxis()
    axs[1, 0].set_xlabel('Average time (ms)', fontsize=10)
    axs[1, 0].set_title('Component average times (with standard deviation)', fontsize=12, fontweight='bold')
    axs[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. Pie chart for main loop time allocation
    pie_means = []
    pie_labels = []
    for name, series in pipeline_data.items():
        pie_means.append(series.mean())
        pie_labels.append(name)
        
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))
    axs[1, 1].pie(pie_means, labels=pie_labels, autopct='%1.1f%%', startangle=140, colors=colors_pie, textprops={'fontsize': 9})
    axs[1, 1].set_title('Main loop time allocation ratio', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=150)
        print(f"Telemetry analysis chart saved to: {output_plot_path}")
        
    plt.close(fig)
    return {
        'avg_total_ms': total_avg_time,
        'avg_fps': 1000.0/total_avg_time if total_avg_time > 0 else 0
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze conveyor belt speed estimation telemetry latency.")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV file with telemetry data")
    parser.add_argument("--output", type=str, default=None, help="Path to save output chart (default: replace .csv with _latency.png)")
    
    args = parser.parse_args()
    
    csv_file = args.csv
    if csv_file is None:
        # Auto-find latest telemetry.csv in output or current directory
        search_dirs = ['output', '.']
        found_files = []
        for d in search_dirs:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith('_telemetry.csv'):
                        found_files.append(os.path.join(d, f))
        
        if found_files:
            # Sort by modification time, newest first
            found_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            csv_file = found_files[0]
            print(f"--csv not specified. Auto-using latest telemetry file: {csv_file}")
        else:
            print("Error: No telemetry CSV files found (*_telemetry.csv) in 'output/' or current directory.")
            print("Run 'python main.py' to generate data or specify path with --csv <path>")
            exit(1)
            
    output_plot = args.output
    if output_plot is None:
        output_plot = csv_file.replace('.csv', '_latency.png')
        
    analyze_telemetry_csv(csv_file, output_plot)
