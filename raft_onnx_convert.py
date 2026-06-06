import torch
import torch.nn as nn
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

class RAFTWrapper(nn.Module):
    def __init__(self, base_model):
        super(RAFTWrapper, self).__init__()
        self.base_model = base_model

    def forward(self, img1, img2):
        predictions = self.base_model(img1, img2)
        return predictions[-1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = Raft_Small_Weights.DEFAULT
raw_model = raft_small(weights=weights, progress=False).to(device)
raw_model.eval()

wrapped_model = RAFTWrapper(raw_model)

dummy_img1 = torch.randn(1, 3, 360, 640).to(device)
dummy_img2 = torch.randn(1, 3, 360, 640).to(device)

torch.onnx.export(
    wrapped_model,
    (dummy_img1, dummy_img2),
    "models/raft_small.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['img1', 'img2'],
    output_names=['flow_predictions'],
    dynamic_axes={
        'img1': {0: 'batch'},
        'img2': {0: 'batch'}
    }
)