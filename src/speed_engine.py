import cv2
import numpy as np
import math
import time
import onnxruntime as ort

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

class RAFTSpeedEngine:
    def __init__(self, model_path, fps=30.0, w_img=640, h_img=480):
        self.fps = fps
        self.FOV, self.H_IMG, self.W_IMG = 95, h_img, w_img
        self.fx = (self.H_IMG / 2.0) / math.tan(math.radians(self.FOV / 2.0))
        self.fy = self.fx
        self.cx, self.cy = self.W_IMG / 2.0, self.H_IMG / 2.0
        self.kinematic_kf = ConveyorKalmanFilter(fps=fps)

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name_1 = self.session.get_inputs()[0].name
        self.input_name_2 = self.session.get_inputs()[1].name
        
        self.last_timings = {
            'preprocess': 0.0,
            'raft_inference': 0.0,
            'flow_grid': 0.0,
            '3d_projection': 0.0,
            'speed_filter': 0.0,
            'total': 0.0
        }

    def preprocess_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
        tensor = np.transpose(img_normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

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
        t_preprocess = time.time() - t0

        t0 = time.time()
        outputs = self.session.run(None, {
            self.input_name_1: img1_batch,
            self.input_name_2: img2_batch
        })
        predicted_flow = outputs[-1][0]
        t_raft_inference = time.time() - t0

        t0 = time.time()
        u_flow = predicted_flow[0]
        v_flow = predicted_flow[1]
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