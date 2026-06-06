import torch
import numpy as np
import onnxruntime as ort
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weights = Raft_Small_Weights.DEFAULT
torch_model = raft_small(weights=weights, progress=False).to(device)
torch_model.eval()

onnx_model_path = "models/raft_small.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
onnx_session = ort.InferenceSession(onnx_model_path, providers=providers)

np.random.seed(42)
img1_np = np.random.uniform(-1.0, 1.0, (1, 3, 360, 640)).astype(np.float32)
img2_np = np.random.uniform(-1.0, 1.0, (1, 3, 360, 640)).astype(np.float32)

img1_torch = torch.from_numpy(img1_np).to(device)
img2_torch = torch.from_numpy(img2_np).to(device)
with torch.no_grad():
    torch_outputs = torch_model(img1_torch, img2_torch)
    torch_flow = torch_outputs[-1].cpu().numpy()

input_names = [x.name for x in onnx_session.get_inputs()]
onnx_outputs = onnx_session.run(None, {
    input_names[0]: img1_np,
    input_names[1]: img2_np
})

onnx_flow = onnx_outputs[-1]

abs_diff = np.abs(torch_flow - onnx_flow)
max_diff = np.max(abs_diff)
mean_diff = np.mean(abs_diff)
print(f"Max absolute difference: {max_diff}")
print(f"Mean absolute difference: {mean_diff}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.boxplot(abs_diff.flatten(), vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
ax1.set_title("Distribution of Absolute Differences")
ax1.set_xticklabels(["|Torch - ONNX|"])
ax1.set_ylabel("Absolute Difference")
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.boxplot([torch_flow.flatten(), onnx_flow.flatten()], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightgreen', color='green'),
            medianprops=dict(color='red'))
ax2.set_title("Flow Values Comparison")
ax2.set_xticklabels(["Torch Flow", "ONNX Flow"])
ax2.set_ylabel("Value")
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("flow_comparison_boxplots.png", dpi=300, bbox_inches='tight')
plt.close(fig)