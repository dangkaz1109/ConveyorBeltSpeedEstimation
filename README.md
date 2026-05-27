# Conveyor Belt Speed Estimation (RAFT + Depth)

## 📌 Overview
This project provides an automated pipeline to estimate the physical moving speed of a conveyor belt from video footage. It utilizes **RAFT (Recurrent All-Pairs Field Transforms)** for precise optical flow extraction and **ONNX-based Monocular Depth Estimation (Depth Anything V3)** for accurate 3D spatial projection.

## ✨ Features
- **Accurate Optical Flow:** Utilizes PyTorch's pre-trained RAFT-Small model for dense optical flow estimation.
- **Metric Depth Estimation:** Integrates ONNXRuntime to run Depth models to calculate actual physical distances.
- **Multi-threaded Pipeline:** Processes depth inference in a separate background thread (`depth_worker_thread`) to maintain video processing performance.
- **Kinematic Smoothing:** Implements a custom 1D Kalman Filter (`ConveyorKalmanFilter`) to stabilize raw speed measurements and reduce noise over time.
- **Comprehensive Evaluation:** Automatically calculates Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) against ground truth speed, generating an evaluation plot upon completion.

## 🚀 Prerequisites
- Python 3.12+
- CUDA-enabled GPU (Highly recommended for running RAFT and ONNX models efficiently).

## 🛠 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/dangkaz1109/ConveyorBeltSpeedEstimation
cd ConveyorBeltSpeedEstimation
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` file uses `onnxruntime-gpu` by default. If you are running on a machine without a GPU, please change it to `onnxruntime` before installing.

## 📥 Downloading Assets

Before running the pipeline, please download the necessary data and model files:

1. **Demo Video:**
   - Download the test video from [Google Drive](https://drive.google.com/file/d/1RjeCGXiVzHHm7D-Iha4yFFENsTlFQmvV/view?usp=sharing).
   - Place it inside the `data/` folder.

2. **Depth Anything V3 (ONNX Model):**
   - Download the ONNX depth model from [Google Drive](https://drive.google.com/file/d/1ZH8j3hnBRCjEe9aJvI2jyrmHtyQqGOal/view?usp=sharing).
   - Place it inside the `models/` folder.

## 💻 Usage

To run the pipeline with the default configuration, simply execute:

```bash
python main.py
```

### ⚙️ Command-Line Arguments

You can customize the pipeline execution using the following arguments:

| Argument | Description | Default |
|---|---|---|
| `--video` | Path to the input video | `data/XV1_III.mp4` |
| `--model` | Path to the ONNX depth model | `models/DA3METRIC-LARGE.onnx` |
| `--output` | Path to save the processed output video | `output/debug_output_raft_3.mp4` |
| `--gt-speed` | Ground truth speed (in m/s) for MAE/RMSE evaluation | `2.5` |
| `--max-frames` | Maximum number of frames to process | `1249` |
| `--skip-frames` | Frame interval for updating the depth map | `5` |
| `--n-update` | Frame interval for recalculating and updating the display speed | `15` |

## 📂 Project Structure

```text
├── data/                  # Directory for input demo videos
├── models/                # Directory for storing downloaded ONNX models
├── output/                # Directory where output videos and evaluation plots are saved
├── src/
│   ├── depth_estimator.py # Contains the ONNX inference class for Depth Estimation
│   ├── pipeline.py        # Handles threading, video reading/writing, and metric plotting
│   └── speed_engine.py    # Implements RAFT flow extraction, 3D projection, and Kalman Filter
├── .gitignore             # Standard git ignore configurations
├── main.py                # Main entry point for the CLI application
├── requirements.txt       # Project Python dependencies
└── README.md              # Project documentation
```

## 📊 Outputs & Evaluation

Upon successful execution, the pipeline will generate two files in the `output/` directory:

1. **Processed Video:** The output video will contain on-screen text displaying the estimated RAFT speed (m/s), the Ground Truth speed, and the current frame index.
2. **Performance Plot:** A `.png` chart visualizing the estimated speed over time compared to the ground truth, annotated with the overall MAE score.
