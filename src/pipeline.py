import os
import cv2
import numpy as np
import time
import threading
import queue
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from src.depth_estimator import DepthEstimator
from src.speed_engine import RAFTSpeedEngine
from analyze_telemetry import analyze_telemetry_csv

class SharedDepthState:
    def __init__(self):
        self.lock = threading.Lock()
        self.depth_map = None
        self.last_inference_time = 0.0
        self.depth_frame_idx = 0  # Theo dõi frame gốc tạo ra depth map này

def depth_worker_thread(frame_queue: queue.Queue, shared_state: SharedDepthState, depth_model: DepthEstimator):
    print("[Depth Thread] Started.")
    while True:
        item = frame_queue.get()
        if item is None:
            break

        f_idx, frame = item
        start_t = time.time()
        new_depth_map = depth_model.predict(frame)
        inference_time = time.time() - start_t

        with shared_state.lock:
            shared_state.depth_map = new_depth_map
            shared_state.last_inference_time = inference_time
            shared_state.depth_frame_idx = f_idx

        frame_queue.task_done()
    print("\n[Depth Thread] Closed.")

def run_3d_pipeline(video_path, model_path, out_video_path, gt_speed=2.5, max_frames=240, skip_frames=5, n_frame_update=10):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found {model_path}")
        return

    print("Initializing Depth Model...")
    depth_estimator = DepthEstimator(model_path=model_path)

    shared_state = SharedDepthState()
    frame_queue = queue.Queue(maxsize=1)

    depth_thread = threading.Thread(
        target=depth_worker_thread,
        args=(frame_queue, shared_state, depth_estimator),
        daemon=True
    )
    depth_thread.start()

    cap = cv2.VideoCapture(video_path)
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, prev_frame = cap.read()
    if not ret: return

    target_width = 640
    original_h, original_w = prev_frame.shape[:2]
    target_height = int(target_width * (original_h / original_w))
    target_width, target_height = (target_width // 8) * 8, (target_height // 8) * 8

    prev_frame_resized = cv2.resize(prev_frame, (target_width, target_height))

    print("Computing Depth for first frame...")
    t0_depth = time.time()
    initial_depth = depth_estimator.predict(prev_frame_resized)
    first_depth_time = time.time() - t0_depth
    print(f"First frame depth computed in: {first_depth_time:.3f}s")

    with shared_state.lock:
        shared_state.depth_map = initial_depth
        shared_state.last_inference_time = first_depth_time
        shared_state.depth_frame_idx = 0

    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (target_width, target_height))
    engine = RAFTSpeedEngine(fps=fps_video, w_img=target_width, h_img=target_height)

    all_frame_times = []
    telemetry_records = []
    error_analysis_records = []  # Thu thập dữ liệu để phân tích nguyên nhân sai số
    history_v, history_gt, frames = [], [], []
    history_mean_depth, history_flow_mag = [], []
    frame_idx, hold_speed, stable_speeds = 1, 0.0, deque(maxlen=30)
    chunk_speeds = []
    chunk_mean_depths = []
    chunk_flow_mags = []
    display_speed = 0.0

    print(f"Starting main loop... (Skip frames: {skip_frames}, Update every {n_frame_update} frames)")

    while True:
        if max_frames and frame_idx > max_frames: break

        t_loop_start = time.time()

        t0 = time.time()
        ret, curr_frame = cap.read()
        if not ret: break

        curr_frame_resized = cv2.resize(curr_frame, (target_width, target_height))
        debug_vis = curr_frame_resized.copy()
        t_frame_io = time.time() - t0

        if frame_idx % skip_frames == 0:
            try:
                frame_queue.put((frame_idx, curr_frame_resized.copy()), block=False)
            except queue.Full:
                pass

        t0 = time.time()
        with shared_state.lock:
            current_depth_map = shared_state.depth_map.copy()
            current_depth_time = shared_state.last_inference_time
            current_depth_frame_idx = shared_state.depth_frame_idx
        t_depth_copy = time.time() - t0

        # Tính độ tuổi bản đồ depth (khoảng cách frame trễ bất đồng bộ)
        depth_age = frame_idx - current_depth_frame_idx

        V_raw, conf, u_flow, v_flow = engine.measure_speed(prev_frame_resized, curr_frame_resized, current_depth_map)
        t_speed_timings = engine.last_timings.copy()

        # Tính mean_depth và flow_mag cho frame hiện tại
        flow_mag = 0.0
        mean_depth = 0.0
        try:
            h_f, w_f = u_flow.shape
            x_min, x_max, y_min, y_max = engine._get_roi_bounds(h_f, w_f)
            roi_u = u_flow[y_min:y_max, x_min:x_max]
            roi_v = v_flow[y_min:y_max, x_min:x_max]
            flow_mag = float(np.mean(np.sqrt(roi_u**2 + roi_v**2)))
            
            roi_depth = current_depth_map[y_min:y_max, x_min:x_max]
            mean_depth = float(np.mean(roi_depth))
        except Exception:
            pass

        t0 = time.time()
        engine.kinematic_kf.predict()
        if V_raw > 0.05:
            V_f = engine.kinematic_kf.correct(V_raw)
            stable_speeds.append(V_f)
            hold_speed = np.median(stable_speeds)
        else:
            V_f = hold_speed
            engine.kinematic_kf.reset_hold_state(hold_speed)
        t_kf = time.time() - t0

        t0 = time.time()
        chunk_speeds.append(V_f)
        chunk_mean_depths.append(mean_depth)
        chunk_flow_mags.append(flow_mag)

        if frame_idx == 1:
            display_speed = V_f

        if frame_idx % n_frame_update == 0:
            display_speed = sum(chunk_speeds) / len(chunk_speeds)
            chunk_speeds.clear()

            avg_mean_depth = sum(chunk_mean_depths) / len(chunk_mean_depths)
            chunk_mean_depths.clear()

            avg_flow_mag = sum(chunk_flow_mags) / len(chunk_flow_mags)
            chunk_flow_mags.clear()

            if frame_idx > 15:
                history_v.append(display_speed)
                history_gt.append(gt_speed)
                history_mean_depth.append(avg_mean_depth)
                history_flow_mag.append(avg_flow_mag)
                frames.append(frame_idx)

        # ---------------- THU THẬP DỮ LIỆU ĐỂ PHÂN TÍCH SAI SỐ ----------------
        try:
            # Sai số tuyệt đối của vận tốc tại frame hiện tại so với Ground Truth
            abs_error = abs(V_f - gt_speed)
            
            error_analysis_records.append({
                'frame_idx': frame_idx,
                'abs_error': abs_error,
                'confidence': conf,
                'flow_magnitude': flow_mag,
                'mean_depth': mean_depth,
                'depth_age': depth_age,
                't_speed_total': t_speed_timings['total']
            })
        except Exception as e:
            pass
        # --------------------------------------------------------------------

        cv2.putText(debug_vis, f"RAFT Speed: {display_speed:.2f} m/s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_vis, f"GT: {gt_speed} m/s", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(debug_vis, f"Frame: {frame_idx}", (target_width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        out_video.write(debug_vis)
        prev_frame_resized = curr_frame_resized.copy()
        t_vis_io = time.time() - t0

        t_total_frame = time.time() - t_loop_start
        all_frame_times.append(t_total_frame)

        # Record telemetry timestamps
        telemetry_records.append({
            'frame_idx': frame_idx,
            't_frame_io': t_frame_io,
            't_depth_copy': t_depth_copy,
            't_depth_inference': current_depth_time,
            't_speed_preprocess': t_speed_timings['preprocess'],
            't_speed_raft_inference': t_speed_timings['raft_inference'],
            't_speed_flow_grid': t_speed_timings['flow_grid'],
            't_speed_3d_projection': t_speed_timings['3d_projection'],
            't_speed_filter': t_speed_timings['speed_filter'],
            't_speed_total': t_speed_timings['total'],
            't_kalman_filter': t_kf,
            't_vis_io': t_vis_io,
            't_total_frame': t_total_frame
        })

        print(f"\rFrame: {frame_idx} | Speed: {display_speed:.2f} | Time: {t_total_frame:.3f}s | Depth Time: {current_depth_time:.3f}s", end="")
        frame_idx += 1

    cap.release()
    out_video.release()

    frame_queue.put(None)
    depth_thread.join()

    print(f"\n\nComplete! Video saved to: {out_video_path}")
    avg_fps = 1.0 / np.mean(all_frame_times)
    print(f"Average processing speed: {avg_fps:.1f} FPS")

    # Save telemetry data
    telemetry_csv_path = out_video_path.replace('.mp4', '_telemetry.csv')
    try:
        telemetry_csv_path = os.path.abspath(telemetry_csv_path)
        with open(telemetry_csv_path, 'w', newline='', encoding='utf-8') as f:
            if telemetry_records:
                writer = csv.DictWriter(f, fieldnames=telemetry_records[0].keys())
                writer.writeheader()
                writer.writerows(telemetry_records)
        print(f"Telemetry data saved to: {telemetry_csv_path}")

        # Analyze and plot telemetry data
        telemetry_plot_path = out_video_path.replace('.mp4', '_telemetry_plot.png')
        analyze_telemetry_csv(telemetry_csv_path, telemetry_plot_path)
    except Exception as e:
        print(f"Error saving/analyzing telemetry: {e}")

    # ----------- THỰC HIỆN PHÂN TÍCH THỐNG KÊ NGUYÊN NHÂN SAI SỐ -----------
    if error_analysis_records:
        try:
            df_err = pd.DataFrame(error_analysis_records)
            # Tính toán Hệ số tương quan Pearson giữa Sai số tuyệt đối với các thành phần quan trọng
            correlations = df_err.corr()['abs_error'].drop(['abs_error', 'frame_idx'])
            
            print("\n" + "="*75)
            print(" BÁO CÁO PHÂN TÍCH CÁC YẾU TỐ ẢNH HƯỞNG ĐẾN SAI SỐ VẬN TỐC DỰ ĐOÁN")
            print("="*75)
            print(f"{'Yếu tố phân tích':<25} | {'Hệ số tương quan (Pearson)':<28} | {'Mức độ & Hướng ảnh hưởng'}")
            print("-"*75)
            
            for factor, corr_val in correlations.items():
                if np.isnan(corr_val):
                    print(f"{factor:<25} | {'N/A':>26} | N/A (Dữ liệu không biến thiên)")
                    continue
                    
                abs_c = abs(corr_val)
                if abs_c > 0.5:
                    impact = "Rất mạnh"
                elif abs_c > 0.3:
                    impact = "Mạnh"
                elif abs_c > 0.1:
                    impact = "Yếu"
                else:
                    impact = "Không đáng kể"
                    
                direction = "Thuận (Yếu tố TĂNG -> SAI SỐ TĂNG ❌)" if corr_val > 0 else "Nghịch (Yếu tố TĂNG -> SAI SỐ GIẢM  )"
                if impact == "Không đáng kể":
                    print(f"{factor:<25} | {corr_val:>26.4f} | {impact}")
                else:
                    print(f"{factor:<25} | {corr_val:>26.4f} | {impact} - {direction}")
            print("="*75)
            
            # Biểu diễn đồ thị trực quan hóa phân tích sai số
            fig, axs = plt.subplots(2, 2, figsize=(15, 11))
            fig.suptitle("PHÂN TÍCH SỰ ẢNH HƯỞNG ĐẾN SAI SỐ VẬN TỐC BĂNG TẢI", fontsize=15, fontweight='bold')
            
            # 1. Biểu đồ cột hệ số tương quan tổng quan
            colors = ['crimson' if c > 0 else 'teal' for c in correlations.values]
            axs[0, 0].bar(correlations.index, correlations.values, color=colors, alpha=0.75, edgecolor='black')
            axs[0, 0].axhline(0, color='black', linewidth=0.8, linestyle='-')
            axs[0, 0].set_title("Hệ số tương quan Pearson với Sai số Tuyệt đối", fontsize=11, fontweight='bold')
            axs[0, 0].set_ylabel("Hệ số tương quan (r)")
            axs[0, 0].grid(True, alpha=0.3)
            axs[0, 0].set_xticklabels(correlations.index, rotation=15)
            
            # 2. Đồ thị phân tán: Confidence vs Abs Error
            axs[0, 1].scatter(df_err['confidence'], df_err['abs_error'], alpha=0.4, color='teal', edgecolors='none')
            axs[0, 1].set_title("Confidence (Bộ lọc IQR) vs Sai số", fontsize=11, fontweight='bold')
            axs[0, 1].set_xlabel("Confidence Rate")
            axs[0, 1].set_ylabel("Absolute Error (m/s)")
            axs[0, 1].grid(True, alpha=0.3)
            
            # 3. Đồ thị phân tán: Flow Magnitude vs Abs Error
            axs[1, 0].scatter(df_err['flow_magnitude'], df_err['abs_error'], alpha=0.4, color='darkorange', edgecolors='none')
            axs[1, 0].set_title("Flow Magnitude (Độ dịch chuyển dòng) vs Sai số", fontsize=11, fontweight='bold')
            axs[1, 0].set_xlabel("Average Flow Magnitude (pixels)")
            axs[1, 0].set_ylabel("Absolute Error (m/s)")
            axs[1, 0].grid(True, alpha=0.3)
            
            # 4. Đồ thị phân tán: Depth Age vs Abs Error
            axs[1, 1].scatter(df_err['depth_age'], df_err['abs_error'], alpha=0.4, color='purple', edgecolors='none')
            axs[1, 1].set_title("Depth Age (Độ trễ frame của Depth) vs Sai số", fontsize=11, fontweight='bold')
            axs[1, 1].set_xlabel("Depth Map Age (frames)")
            axs[1, 1].set_ylabel("Absolute Error (m/s)")
            axs[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            err_plot_path = out_video_path.replace('.mp4', '_error_analysis.png')
            plt.savefig(err_plot_path, dpi=150)
            print(f"Error analysis plot saved to: {err_plot_path}")
            plt.close()
        except Exception as e:
            print(f"Error doing error analysis: {e}")
    # -----------------------------------------------------------------------

    if history_v:
        history_v_np = np.array(history_v)
        history_gt_np = np.array(history_gt)
        mae = np.mean(np.abs(history_v_np - history_gt_np))
        rmse = np.sqrt(np.mean((history_v_np - history_gt_np)**2))

        print(f"--- VALIDATION RESULTS (computed on update intervals) ---")
        print(f"Total updates: {len(history_v)}")
        print(f"Average MAE: {mae:.4f} m/s")
        print(f"Average RMSE: {rmse:.4f} m/s")

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Speed Estimation Performance & Key Metrics\n(MAE: {mae:.4f} m/s, RMSE: {rmse:.4f} m/s)", fontsize=14, fontweight='bold')

        # Subplot 1: Speed
        axs[0].plot(frames, history_v, label=f'Estimated Speed (Update every {n_frame_update} frames)', color='#1f77b4', linewidth=2, marker='o', markersize=4)
        axs[0].axhline(y=gt_speed, color='#d62728', linestyle='--', label=f'Ground Truth ({gt_speed} m/s)')
        axs[0].set_ylabel("Speed (m/s)", fontsize=11, fontweight='bold')
        axs[0].legend(loc='upper right')
        axs[0].grid(True, linestyle='--', alpha=0.5)
        axs[0].set_ylim(0, max(gt_speed, max(history_v) if history_v else 0) + 1.0)
        axs[0].set_title("Estimated Speed vs Ground Truth", fontsize=12, fontweight='bold', loc='left')

        # Subplot 2: Mean Depth
        axs[1].plot(frames, history_mean_depth, label='Mean Depth (ROI)', color='#2ca02c', linewidth=2, marker='s', markersize=4)
        axs[1].set_ylabel("Depth (m)", fontsize=11, fontweight='bold')
        axs[1].legend(loc='upper right')
        axs[1].grid(True, linestyle='--', alpha=0.5)
        if history_mean_depth:
            axs[1].set_ylim(max(0, min(history_mean_depth) - 0.5), max(history_mean_depth) + 0.5)
        axs[1].set_title("Mean Depth over Time", fontsize=12, fontweight='bold', loc='left')

        # Subplot 3: Flow Magnitude
        axs[2].plot(frames, history_flow_mag, label='Flow Magnitude (ROI)', color='#9467bd', linewidth=2, marker='^', markersize=4)
        axs[2].set_ylabel("Flow (pixels)", fontsize=11, fontweight='bold')
        axs[2].set_xlabel("Frame Index", fontsize=12, fontweight='bold')
        axs[2].legend(loc='upper right')
        axs[2].grid(True, linestyle='--', alpha=0.5)
        if history_flow_mag:
            axs[2].set_ylim(max(0, min(history_flow_mag) - 2.0), max(history_flow_mag) + 2.0)
        axs[2].set_title("Optical Flow Magnitude over Time", fontsize=12, fontweight='bold', loc='left')

        plt.tight_layout()
        plot_path = out_video_path.replace('.mp4', '_plot.png')
<<<<<<< HEAD
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close()
=======
        plt.savefig(plot_path, dpi=150)
        print(f"Đã lưu biểu đồ tại: {plot_path}")
        plt.close()
>>>>>>> 8750562 (Done)
