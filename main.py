import argparse
import os
from src.pipeline import run_3d_pipeline

def main():
    parser = argparse.ArgumentParser(description="Đo tốc độ băng tải bằng RAFT Optical Flow và mô hình độ sâu ONNX.")
    parser.add_argument("--video", type=str, default="data/XV1_III.mp4", help="Đường dẫn đến video đầu vào")
    parser.add_argument("--model", type=str, default="models/DA3METRIC-LARGE.onnx", help="Đường dẫn đến mô hình ONNX")
    parser.add_argument("--output", type=str, default="output/debug_output_raft_3.mp4", help="Đường dẫn lưu video kết quả")
    parser.add_argument("--gt-speed", type=float, default=2.5, help="Tốc độ Ground Truth (m/s) để so sánh")
    parser.add_argument("--max-frames", type=int, default=1249, help="Số frame tối đa cần xử lý")
    parser.add_argument("--skip-frames", type=int, default=5, help="Khoảng cách frames để update luồng depth")
    parser.add_argument("--n-update", type=int, default=15, help="Chu kỳ update tốc độ (tính theo frame)")

    args = parser.parse_args()

    # Đảm bảo thư mục output tồn tại
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    run_3d_pipeline(
        video_path=args.video,
        model_path=args.model,
        out_video_path=args.output,
        gt_speed=args.gt_speed,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
        n_frame_update=args.n_update
    )

if __name__ == "__main__":
    main()