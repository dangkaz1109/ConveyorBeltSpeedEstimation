import os
import cv2
import numpy as np
import time
import threading
import queue
import csv
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

def depth_worker_thread(frame_queue: queue.Queue, shared_state: SharedDepthState, depth_model: DepthEstimator):
    print("[Depth Thread] Bắt đầu chạy...")
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        start_t = time.time()
        new_depth_map = depth_model.predict(frame)
        inference_time = time.time() - start_t

        with shared_state.lock:
            shared_state.depth_map = new_depth_map
            shared_state.last_inference_time = inference_time

        frame_queue.task_done()
    print("\n[Depth Thread] Đã đóng.")

def run_3d_pipeline(video_path, model_path, out_video_path, gt_speed=2.5, max_frames=240, skip_frames=5, n_frame_update=10):
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video {video_path}")
        return

    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file mô hình {model_path}")
        return

    print("Đang khởi tạo Depth Model...")
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

    print("Đang tính toán Depth cho frame đầu tiên...")
    t0_depth = time.time()
    initial_depth = depth_estimator.predict(prev_frame_resized)
    first_depth_time = time.time() - t0_depth
    print(f"Hoàn thành tính toán Depth frame đầu tiên trong: {first_depth_time:.3f}s")

    with shared_state.lock:
        shared_state.depth_map = initial_depth
        shared_state.last_inference_time = first_depth_time

    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (target_width, target_height))
    engine = RAFTSpeedEngine(fps=fps_video, w_img=target_width, h_img=target_height)

    all_frame_times = []
    telemetry_records = []
    history_v, history_gt, frames = [], [], []
    frame_idx, hold_speed, stable_speeds = 1, 0.0, deque(maxlen=30)
    chunk_speeds = []
    display_speed = 0.0

    print(f"Bắt đầu xử lý luồng chính... (Skip frames: {skip_frames}, Update every {n_frame_update} frames)")

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
                frame_queue.put(curr_frame_resized.copy(), block=False)
            except queue.Full:
                pass

        t0 = time.time()
        with shared_state.lock:
            current_depth_map = shared_state.depth_map.copy()
            current_depth_time = shared_state.last_inference_time
        t_depth_copy = time.time() - t0

        V_raw, conf, u_flow, v_flow = engine.measure_speed(prev_frame_resized, curr_frame_resized, current_depth_map)
        t_speed_timings = engine.last_timings.copy()

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

        if frame_idx == 1:
            display_speed = V_f

        if frame_idx % n_frame_update == 0:
            display_speed = sum(chunk_speeds) / len(chunk_speeds)
            chunk_speeds.clear()

            if frame_idx > 15:
                history_v.append(display_speed)
                history_gt.append(gt_speed)
                frames.append(frame_idx)

        cv2.putText(debug_vis, f"RAFT Speed: {display_speed:.2f} m/s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_vis, f"GT: {gt_speed} m/s", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(debug_vis, f"Frame: {frame_idx}", (target_width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        out_video.write(debug_vis)
        prev_frame_resized = curr_frame_resized.copy()
        t_vis_io = time.time() - t0

        t_total_frame = time.time() - t_loop_start
        all_frame_times.append(t_total_frame)

        # Ghi nhận các mốc thời gian telemetry
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

    print(f"\n\nHoàn tất! Video lưu tại: {out_video_path}")
    avg_fps = 1.0 / np.mean(all_frame_times)
    print(f"Tốc độ xử lý trung bình: {avg_fps:.1f} FPS")

    # Lưu dữ liệu telemetry
    telemetry_csv_path = out_video_path.replace('.mp4', '_telemetry.csv')
    try:
        telemetry_csv_path = os.path.abspath(telemetry_csv_path)
        with open(telemetry_csv_path, 'w', newline='', encoding='utf-8') as f:
            if telemetry_records:
                writer = csv.DictWriter(f, fieldnames=telemetry_records[0].keys())
                writer.writeheader()
                writer.writerows(telemetry_records)
        print(f"Đã lưu dữ liệu telemetry tại: {telemetry_csv_path}")

        # Gọi phân tích để vẽ biểu đồ và hiển thị thống kê
        telemetry_plot_path = out_video_path.replace('.mp4', '_telemetry_plot.png')
        analyze_telemetry_csv(telemetry_csv_path, telemetry_plot_path)
    except Exception as e:
        print(f"Lỗi khi lưu/phân tích telemetry: {e}")

    if history_v:
        history_v_np = np.array(history_v)
        history_gt_np = np.array(history_gt)
        mae = np.mean(np.abs(history_v_np - history_gt_np))
        rmse = np.sqrt(np.mean((history_v_np - history_gt_np)**2))

        print(f"--- KẾT QUẢ KIỂM TRA (Tính trên các mốc cập nhật) ---")
        print(f"Tổng số lần cập nhật: {len(history_v)}")
        print(f"Average MAE: {mae:.4f} m/s")
        print(f"Average RMSE: {rmse:.4f} m/s")

        plt.figure(figsize=(12, 6))
        plt.plot(frames, history_v, label=f'Estimated Speed (Update every {n_frame_update} frames)', color='blue', linewidth=2, marker='o', markersize=4)
        plt.axhline(y=gt_speed, color='red', linestyle='--', label=f'Ground Truth ({gt_speed} m/s)')

        plt.title(f"Speed Estimation Performance (MAE: {mae:.4f} m/s)", fontsize=14)
        plt.xlabel("Frame Index", fontsize=12)
        plt.ylabel("Speed (m/s)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, gt_speed + 1.0)
        
        plot_path = out_video_path.replace('.mp4', '_plot.png')
        plt.savefig(plot_path)
        print(f"Đã lưu biểu đồ tại: {plot_path}")
        plt.close()