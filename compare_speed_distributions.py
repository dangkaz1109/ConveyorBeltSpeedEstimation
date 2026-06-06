#!/usr/bin/env python3
"""
Comparison Script: PyTorch RAFT vs ONNX RAFT Speed Estimation
This script runs both models frame-by-frame on the same input video
using identical depth maps and compares their speed distributions and latencies.
"""

import os
import cv2
import numpy as np
import math
import torch
import time
import argparse
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

# Try to use a clean style for matplotlib
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass

from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from src.depth_estimator import DepthEstimator
from src.speed_engine import RAFTSpeedEngine as RAFTSpeedEngineONNX

class ConveyorKalmanFilter:
    def __init__(self, fps=30.0):
        dt = 1.0 / fps
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.transitionMatrix = np.array([[1.0, dt], [0.0, 1.0]], np.float32)
        self.kf.measurementMatrix = np.array([[1.0, 0.0]], np.float32)
        self.kf.processNoiseCov = np.array([[1e-4, 0.0], [0.0, 1e-5]], np.float32)
        self.kf.measurementNoiseCov = np.array([[1.0]], np.float32)
        self.kf.statePost = np.array([[0.0], [0.0]], np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)

    def predict(self):
        predicted = self.kf.predict()
        return max(0.0, float(predicted[0, 0]))

    def correct(self, measured_speed):
        measurement = np.array([[np.float32(measured_speed)]])
        estimated = self.kf.correct(measurement)
        return max(0.0, float(estimated[0, 0]))

    def reset_hold_state(self, hold_speed):
        self.kf.statePost = np.array([[np.float32(hold_speed)], [0.0]], np.float32)

class RAFTSpeedEngineTorch:
    def __init__(self, fps=30.0, w_img=640, h_img=480):
        self.fps = fps
        self.FOV, self.H_IMG, self.W_IMG = 95, h_img, w_img
        self.fx = (self.H_IMG / 2.0) / math.tan(math.radians(self.FOV / 2.0))
        self.fy = self.fx
        self.cx, self.cy = self.W_IMG / 2.0, self.H_IMG / 2.0
        self.kinematic_kf = ConveyorKalmanFilter(fps=fps)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=weights, progress=False).to(self.device)
        self.model.eval()
        self.transforms = weights.transforms()
        self.last_timings = {
            'preprocess': 0.0,
            'raft_inference': 0.0,
            'flow_grid': 0.0,
            '3d_projection': 0.0,
            'speed_filter': 0.0,
            'total': 0.0
        }

    def get_3d_point(self, u, v, depth_map):
        y, x = int(v), int(u)
        h, w = depth_map.shape
        y = max(0, min(h-1, y))
        x = max(0, min(w-1, x))
        y1, y2 = max(0, y-2), min(h, y+3)
        x1, x2 = max(0, x-2), min(w, x+3)
        region_depth = depth_map[y1:y2, x1:x2]
        valid_depths = region_depth[(region_depth > 0.1) & (region_depth < 10.0)]
        if len(valid_depths) == 0: return None
        Z = np.median(valid_depths)
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return np.array([X, Y, Z])

    def preprocess_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _get_roi_bounds(self, h, w):
        x_min = int(w * 0.20)
        x_max = int(w * 0.80)
        y_min = int(h * 0.40)
        y_max = int(h * 0.95)
        return x_min, x_max, y_min, y_max

    def measure_speed(self, prev_frame, curr_frame, depth_map_curr, grid_size=(20, 20)):
        t_start = time.time()
        
        t0 = time.time()
        img1_batch = self.preprocess_image(prev_frame)
        img2_batch = self.preprocess_image(curr_frame)
        img1_batch, img2_batch = self.transforms(img1_batch, img2_batch)
        t_preprocess = time.time() - t0

        t0 = time.time()
        with torch.no_grad():
            list_of_flows = self.model(img1_batch, img2_batch)
            predicted_flow = list_of_flows[-1][0]
        t_raft_inference = time.time() - t0

        t0 = time.time()
        u_flow = predicted_flow[0].cpu().numpy()
        v_flow = predicted_flow[1].cpu().numpy()
        h, w = u_flow.shape

        x_min, x_max, y_min, y_max = self._get_roi_bounds(h, w)
        x_coords = np.linspace(x_min, x_max, grid_size[0], dtype=int)
        y_coords = np.linspace(y_min, y_max, grid_size[1], dtype=int)

        X_grid, Y_grid = np.meshgrid(x_coords, y_coords)
        pts_u_old = X_grid.flatten()
        pts_v_old = Y_grid.flatten()

        u_vals = u_flow[pts_v_old, pts_u_old]
        v_vals = v_flow[pts_v_old, pts_u_old]

        pts_u_new = pts_u_old + u_vals
        pts_v_new = pts_v_old + v_vals

        valid_bounds = (pts_u_new >= 0) & (pts_u_new < w) & (pts_v_new >= 0) & (pts_v_new < h)

        pts_u_old = pts_u_old[valid_bounds]
        pts_v_old = pts_v_old[valid_bounds]
        pts_u_new = pts_u_new[valid_bounds].astype(int)
        pts_v_new = pts_v_new[valid_bounds].astype(int)
        t_flow_grid = time.time() - t0

        t0 = time.time()
        if len(pts_u_old) == 0:
            self.last_timings = {
                'preprocess': t_preprocess,
                'raft_inference': t_raft_inference,
                'flow_grid': t_flow_grid,
                '3d_projection': 0.0,
                'speed_filter': 0.0,
                'total': time.time() - t_start
            }
            return 0.0, 0.0, u_flow, v_flow

        Z_old = depth_map_curr[pts_v_old, pts_u_old]
        Z_new = depth_map_curr[pts_v_new, pts_u_new]

        valid_depth = (Z_old > 0.5) & (Z_old < 8.0) & (Z_new > 0.5) & (Z_new < 8.0)
        pts_u_old, pts_v_old = pts_u_old[valid_depth], pts_v_old[valid_depth]
        pts_u_new, pts_v_new = pts_u_new[valid_depth], pts_v_new[valid_depth]
        Z_old, Z_new = Z_old[valid_depth], Z_new[valid_depth]

        if len(Z_old) < 5:
            self.last_timings = {
                'preprocess': t_preprocess,
                'raft_inference': t_raft_inference,
                'flow_grid': t_flow_grid,
                '3d_projection': time.time() - t0,
                'speed_filter': 0.0,
                'total': time.time() - t_start
            }
            return 0.0, 0.0, u_flow, v_flow

        X3D_old = (pts_u_old - self.cx) * Z_old / self.fx
        Y3D_old = (pts_v_old - self.cy) * Z_old / self.fy

        X3D_new = (pts_u_new - self.cx) * Z_new / self.fx
        Y3D_new = (pts_v_new - self.cy) * Z_new / self.fy

        dist_3d = np.sqrt((X3D_new - X3D_old)**2 + (Y3D_new - Y3D_old)**2 + (Z_new - Z_old)**2)
        speeds = dist_3d * self.fps
        t_3d_projection = time.time() - t0

        t0 = time.time()
        # Compute flow magnitude for each point to filter background noise
        flow_mags = np.sqrt(u_flow[pts_v_old, pts_u_old]**2 + v_flow[pts_v_old, pts_u_old]**2)
        motion_threshold = 1.0  # pixels
        moving_mask = flow_mags > motion_threshold
        
        if np.sum(moving_mask) >= 5:
            speeds = speeds[moving_mask]
            flow_mags = flow_mags[moving_mask]
        else:
            self.last_timings = {
                'preprocess': t_preprocess,
                'raft_inference': t_raft_inference,
                'flow_grid': t_flow_grid,
                '3d_projection': t_3d_projection,
                'speed_filter': time.time() - t0,
                'total': time.time() - t_start
            }
            return 0.0, 0.0, u_flow, v_flow

        Q1 = np.percentile(speeds, 25)
        Q3 = np.percentile(speeds, 75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 2.0 * IQR)
        upper_bound = min(10.0, Q3 + 2.0 * IQR)

        valid_speeds = speeds[(speeds >= lower_bound) & (speeds <= upper_bound)]

        if len(valid_speeds) == 0:
            self.last_timings = {
                'preprocess': t_preprocess,
                'raft_inference': t_raft_inference,
                'flow_grid': t_flow_grid,
                '3d_projection': t_3d_projection,
                'speed_filter': time.time() - t0,
                'total': time.time() - t_start
            }
            return 0.0, 0.0, u_flow, v_flow

        final_speed = float(np.median(valid_speeds))
        confidence = float(len(valid_speeds) / (grid_size[0] * grid_size[1]))
        t_speed_filter = time.time() - t0

        self.last_timings = {
            'preprocess': t_preprocess,
            'raft_inference': t_raft_inference,
            'flow_grid': t_flow_grid,
            '3d_projection': t_3d_projection,
            'speed_filter': t_speed_filter,
            'total': time.time() - t_start
        }

        return final_speed, confidence, u_flow, v_flow

def main():
    parser = argparse.ArgumentParser(description="So sánh phân bố tốc độ và thời gian xử lý của PyTorch và ONNX Speed Engine.")
    parser.add_argument("--video", type=str, default="data/XV1_III.mp4", help="Đường dẫn đến video đầu vào")
    parser.add_argument("--depth-model", type=str, default="models/DA3METRIC-LARGE.onnx", help="Đường dẫn đến mô hình độ sâu ONNX")
    parser.add_argument("--onnx-raft", type=str, default="models/raft_small.onnx", help="Đường dẫn đến mô hình RAFT ONNX")
    parser.add_argument("--max-frames", type=int, default=300, help="Số frame tối đa để so sánh")
    parser.add_argument("--skip-frames", type=int, default=5, help="Chu kỳ cập nhật bản đồ độ sâu (frames)")
    parser.add_argument("--gt-speed", type=float, default=2.5, help="Tốc độ Ground Truth (m/s) để hiển thị")
    parser.add_argument("--output-plot", type=str, default="output/speed_comparison_distribution.png", help="Đường dẫn lưu đồ thị so sánh")
    parser.add_argument("--output-csv", type=str, default="output/speed_comparison_results.csv", help="Đường dẫn lưu dữ liệu kết quả")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" BẮT ĐẦU CHƯƠNG TRÌNH SO SÁNH PHÂN BỐ TỐC ĐỘ: PYTORCH vs ONNX")
    print("="*80)
    
    # Check paths
    for p in [args.video, args.depth_model, args.onnx_raft]:
        if not os.path.exists(p):
            print(f"Lỗi: Không tìm thấy file {p}")
            return

    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print("Đang khởi tạo mô hình ước lượng độ sâu (Depth Estimator)...")
    depth_estimator = DepthEstimator(model_path=args.depth_model)

    cap = cv2.VideoCapture(args.video)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc video.")
        return

    target_width = 640
    original_h, original_w = prev_frame.shape[:2]
    target_height = int(target_width * (original_h / original_w))
    target_width, target_height = (target_width // 8) * 8, (target_height // 8) * 8
    prev_frame_resized = cv2.resize(prev_frame, (target_width, target_height))

    print(f"Kích thước khung hình xử lý: {target_width}x{target_height} @ {fps_video:.2f} FPS")

    print("Khởi tạo mô hình ONNX Speed Engine...")
    onnx_engine = RAFTSpeedEngineONNX(model_path=args.onnx_raft, fps=fps_video, w_img=target_width, h_img=target_height)

    print("Khởi tạo mô hình PyTorch Speed Engine...")
    torch_engine = RAFTSpeedEngineTorch(fps=fps_video, w_img=target_width, h_img=target_height)

    print("Tính toán độ sâu cho frame đầu tiên...")
    current_depth_map = depth_estimator.predict(prev_frame_resized)

    # Kalman Filter states for tracking
    onnx_stable_speeds = deque(maxlen=30)
    onnx_hold_speed = 0.0
    
    torch_stable_speeds = deque(maxlen=30)
    torch_hold_speed = 0.0

    records = []
    frame_idx = 1

    print(f"\nĐang tiến hành xử lý tối đa {args.max_frames} frames...")
    
    t_start = time.time()
    
    while True:
        if args.max_frames and frame_idx > args.max_frames:
            break

        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_frame_resized = cv2.resize(curr_frame, (target_width, target_height))

        # Update depth map synchronously
        if frame_idx % args.skip_frames == 0:
            current_depth_map = depth_estimator.predict(curr_frame_resized)

        # 1. Run ONNX estimation
        t0 = time.time()
        V_raw_onnx, conf_onnx, u_flow_onnx, v_flow_onnx = onnx_engine.measure_speed(
            prev_frame_resized, curr_frame_resized, current_depth_map
        )
        t_infer_onnx = onnx_engine.last_timings['raft_inference'] * 1000.0  # ms
        t_total_onnx = (time.time() - t0) * 1000.0  # ms

        # Apply Kalman Filter to ONNX raw speed
        onnx_engine.kinematic_kf.predict()
        if V_raw_onnx > 0.05:
            V_f_onnx = onnx_engine.kinematic_kf.correct(V_raw_onnx)
            onnx_stable_speeds.append(V_f_onnx)
            onnx_hold_speed = np.median(onnx_stable_speeds)
        else:
            V_f_onnx = onnx_hold_speed
            onnx_engine.kinematic_kf.reset_hold_state(onnx_hold_speed)

        # 2. Run PyTorch estimation
        t0 = time.time()
        V_raw_torch, conf_torch, u_flow_torch, v_flow_torch = torch_engine.measure_speed(
            prev_frame_resized, curr_frame_resized, current_depth_map
        )
        t_infer_torch = torch_engine.last_timings['raft_inference'] * 1000.0  # ms
        t_total_torch = (time.time() - t0) * 1000.0  # ms

        # Apply Kalman Filter to PyTorch raw speed
        torch_engine.kinematic_kf.predict()
        if V_raw_torch > 0.05:
            V_f_torch = torch_engine.kinematic_kf.correct(V_raw_torch)
            torch_stable_speeds.append(V_f_torch)
            torch_hold_speed = np.median(torch_stable_speeds)
        else:
            V_f_torch = torch_hold_speed
            torch_engine.kinematic_kf.reset_hold_state(torch_hold_speed)

        # Difference metric
        raw_diff = abs(V_raw_torch - V_raw_onnx)
        filtered_diff = abs(V_f_torch - V_f_onnx)

        records.append({
            'frame_idx': frame_idx,
            'V_raw_onnx': V_raw_onnx,
            'V_filtered_onnx': V_f_onnx,
            'conf_onnx': conf_onnx,
            't_infer_onnx': t_infer_onnx,
            't_total_onnx': t_total_onnx,
            'V_raw_torch': V_raw_torch,
            'V_filtered_torch': V_f_torch,
            'conf_torch': conf_torch,
            't_infer_torch': t_infer_torch,
            't_total_torch': t_total_torch,
            'raw_diff': raw_diff,
            'filtered_diff': filtered_diff
        })

        if frame_idx % 20 == 0 or frame_idx == 1:
            print(f"Frame {frame_idx:4d} | ONNX: Raw={V_raw_onnx:5.2f}m/s Filtered={V_f_onnx:5.2f}m/s Latency={t_infer_onnx:5.1f}ms | Torch: Raw={V_raw_torch:5.2f}m/s Filtered={V_f_torch:5.2f}m/s Latency={t_infer_torch:5.1f}ms")

        prev_frame_resized = curr_frame_resized.copy()
        frame_idx += 1

    cap.release()
    total_run_time = time.time() - t_start
    print(f"\nHoàn tất chạy thử nghiệm! Tổng thời gian chạy: {total_run_time:.2f}s")

    # Analyze data
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"Dữ liệu chi tiết lưu vào: {args.output_csv}")

    # Compute Statistical Summaries
    print("\n" + "="*80)
    print(" BÁO CÁO THỐNG KÊ CHI TIẾT PHÂN BỐ TỐC ĐỘ (M/S) VÀ THỜI GIAN CHẠY (MS)")
    print("="*80)
    
    stats_summary = []
    metrics_cols = ['V_raw_onnx', 'V_raw_torch', 'V_filtered_onnx', 'V_filtered_torch', 't_infer_onnx', 't_infer_torch']
    
    print(f"{'Chỉ số (Metric)':<25} | {'Mean':<10} | {'Median':<10} | {'Std':<8} | {'Min':<8} | {'Max':<8}")
    print("-"*80)
    for col in metrics_cols:
        mean_val = df[col].mean()
        med_val = df[col].median()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"{col:<25} | {mean_val:10.4f} | {med_val:10.4f} | {std_val:8.4f} | {min_val:8.4f} | {max_val:8.4f}")
        stats_summary.append({
            'metric': col, 'mean': mean_val, 'median': med_val, 'std': std_val, 'min': min_val, 'max': max_val
        })
    print("-"*80)

    # Compute comparison errors
    raw_mae = np.mean(df['raw_diff'])
    raw_rmse = np.sqrt(np.mean(df['raw_diff']**2))
    
    filtered_mae = np.mean(df['filtered_diff'])
    filtered_rmse = np.sqrt(np.mean(df['filtered_diff']**2))
    
    # Correlation coefficients
    raw_corr = df['V_raw_onnx'].corr(df['V_raw_torch'])
    filtered_corr = df['V_filtered_onnx'].corr(df['V_filtered_torch'])
    
    print(f"Sai số tuyệt đối trung bình (MAE) giữa tốc độ Raw:      {raw_mae:.4f} m/s")
    print(f"Sai số bình phương trung bình (RMSE) giữa tốc độ Raw:    {raw_rmse:.4f} m/s")
    print(f"Hệ số tương quan Pearson giữa tốc độ Raw:                {raw_corr:.4f}")
    print("-"*80)
    print(f"Sai số tuyệt đối trung bình (MAE) giữa tốc độ Filtered: {filtered_mae:.4f} m/s")
    print(f"Sai số bình phương trung bình (RMSE) giữa tốc độ Filtered:  {filtered_rmse:.4f} m/s")
    print(f"Hệ số tương quan Pearson giữa tốc độ Filtered:            {filtered_corr:.4f}")
    print("="*80)

    # Plot graphs
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("So Sánh Phân Bố Tốc Độ & Latency: PyTorch vs ONNX Speed Engine", fontsize=16, fontweight='bold')

    # Plot 1: Speed over time
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(df['frame_idx'], df['V_raw_onnx'], label='Raw Speed (ONNX)', color='skyblue', alpha=0.5, linestyle='--')
    ax1.plot(df['frame_idx'], df['V_raw_torch'], label='Raw Speed (Torch)', color='salmon', alpha=0.5, linestyle='--')
    ax1.plot(df['frame_idx'], df['V_filtered_onnx'], label='Filtered Speed (ONNX)', color='darkblue', linewidth=2)
    ax1.plot(df['frame_idx'], df['V_filtered_torch'], label='Filtered Speed (Torch)', color='darkred', linewidth=2)
    ax1.axhline(y=args.gt_speed, color='green', linestyle=':', linewidth=2, label=f'Ground Truth ({args.gt_speed} m/s)')
    ax1.set_title("Vận tốc đo được qua từng khung hình (m/s)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Chỉ số khung hình (Frame Index)")
    ax1.set_ylabel("Tốc độ (m/s)")
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: Boxplot distribution comparison
    ax2 = plt.subplot2grid((3, 2), (1, 0))
    speed_data = [df['V_raw_onnx'], df['V_raw_torch'], df['V_filtered_onnx'], df['V_filtered_torch']]
    box = ax2.boxplot(speed_data, patch_artist=True, labels=[
        'Raw\nONNX', 'Raw\nTorch', 'Filtered\nONNX', 'Filtered\nTorch'
    ])
    colors = ['lightblue', 'lightcolors', 'blue', 'red']
    # Use color scheme
    box_colors = ['#aec7e8', '#ffbb78', '#1f77b4', '#ff7f0e']
    for patch, color in zip(box['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for median in box['medians']:
        median.set(color='black', linewidth=1.5)
    ax2.set_title("Hộp phân bố (Boxplot) vận tốc đo được", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Tốc độ (m/s)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Plot 3: Scatter plot comparison
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.scatter(df['V_filtered_onnx'], df['V_filtered_torch'], alpha=0.6, color='purple', edgecolors='none', label='Mẫu vận tốc')
    # Perfect alignment line
    min_speed = min(df['V_filtered_onnx'].min(), df['V_filtered_torch'].min())
    max_speed = max(df['V_filtered_onnx'].max(), df['V_filtered_torch'].max())
    ax3.plot([min_speed, max_speed], [min_speed, max_speed], color='red', linestyle='--', label='Đường khớp tuyệt đối (Y=X)')
    ax3.set_title(f"Tương quan vận tốc Filtered: ONNX vs PyTorch (r = {filtered_corr:.4f})", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Vận tốc Filtered ONNX (m/s)")
    ax3.set_ylabel("Vận tốc Filtered PyTorch (m/s)")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Plot 4: Boxplot distribution latency comparison
    ax4 = plt.subplot2grid((3, 2), (2, 0))
    latency_data = [df['t_infer_onnx'], df['t_infer_torch']]
    box_lat = ax4.boxplot(latency_data, patch_artist=True, labels=['Inference ONNX', 'Inference Torch'])
    lat_colors = ['#2ca02c', '#d62728']
    for patch, color in zip(box_lat['boxes'], lat_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for median in box_lat['medians']:
        median.set(color='black', linewidth=1.5)
    ax4.set_title("Hộp phân bố thời gian suy luận (RAFT Inference Latency)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Thời gian (ms)")
    ax4.grid(True, linestyle='--', alpha=0.5)

    # Plot 5: Latency over time
    ax5 = plt.subplot2grid((3, 2), (2, 1))
    ax5.plot(df['frame_idx'], df['t_infer_onnx'], label='ONNX Latency', color='green', alpha=0.7)
    ax5.plot(df['frame_idx'], df['t_infer_torch'], label='Torch Latency', color='red', alpha=0.7)
    ax5.set_title("Thời gian suy luận qua từng khung hình (ms)", fontsize=12, fontweight='bold')
    ax5.set_xlabel("Chỉ số khung hình (Frame Index)")
    ax5.set_ylabel("Thời gian (ms)")
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.output_plot, dpi=150)
    print(f"Đồ thị so sánh chi tiết đã lưu vào: {args.output_plot}")
    plt.close()
    
    # Save a text report summarizing the findings
    report_path = args.output_plot.replace('.png', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" BÁO CÁO SO SÁNH VẬN TỐC BĂNG TẢI: PYTORCH VS ONNX\n")
        f.write("="*80 + "\n\n")
        f.write(f"Đường dẫn video: {args.video}\n")
        f.write(f"Số lượng frames xử lý: {len(df)}\n")
        f.write(f"Vận tốc Ground Truth: {args.gt_speed} m/s\n\n")
        f.write("-"*80 + "\n")
        f.write("THỐNG KÊ CHI TIẾT TỪNG BIẾN ĐO LƯỜNG:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Biến đo':<20} | {'Mean':<10} | {'Median':<10} | {'Std':<8} | {'Min':<8} | {'Max':<8}\n")
        f.write("-"*80 + "\n")
        for col in metrics_cols:
            f.write(f"{col:<20} | {df[col].mean():10.4f} | {df[col].median():10.4f} | {df[col].std():8.4f} | {df[col].min():8.4f} | {df[col].max():8.4f}\n")
        f.write("-"*80 + "\n\n")
        f.write("CHỈ SỐ SO SÁNH VÀ SAI SỐ GIỮA HAI PHƯƠNG PHÁP:\n")
        f.write(f" - Tốc độ Raw: MAE={raw_mae:.4f} m/s, RMSE={raw_rmse:.4f} m/s, Pearson r={raw_corr:.4f}\n")
        f.write(f" - Tốc độ Filtered: MAE={filtered_mae:.4f} m/s, RMSE={filtered_rmse:.4f} m/s, Pearson r={filtered_corr:.4f}\n\n")
        
        # Calculate latency ratios
        onnx_mean_lat = df['t_infer_onnx'].mean()
        torch_mean_lat = df['t_infer_torch'].mean()
        speedup = torch_mean_lat / onnx_mean_lat if onnx_mean_lat > 0 else 0
        f.write("HIỆU NĂNG TÍNH TOÁN (RAFT INFERENCE):\n")
        f.write(f" - ONNX Mean Inference: {onnx_mean_lat:.2f} ms ({1000.0/onnx_mean_lat:.1f} FPS)\n")
        f.write(f" - Torch Mean Inference: {torch_mean_lat:.2f} ms ({1000.0/torch_mean_lat:.1f} FPS)\n")
        f.write(f" - ONNX nhanh hơn PyTorch khoảng {speedup:.2f} lần\n")
    print(f"Bản báo cáo text được lưu tại: {report_path}")

if __name__ == '__main__':
    main()
