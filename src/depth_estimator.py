import cv2
import numpy as np
import onnxruntime as ort
from typing import Union

class DepthEstimator:
    def __init__(self, model_path: str, providers: list = None):
        """
        Khởi tạo mô hình ước lượng độ sâu.
        """
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Union[str, np.ndarray], target_size: tuple = (504, 280)) -> np.ndarray:
        if isinstance(image, str):
            original_img = cv2.imread(image)
            if original_img is None:
                raise ValueError(f"Không thể đọc ảnh từ: {image}")
        else:
            original_img = image

        h_orig, w_orig = original_img.shape[:2]
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_rgb, target_size)

        image_tensor = resized_img.astype(np.float32) / 255.0
        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(image_tensor, axis=0)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        depth_raw = outputs[0]
        depth_metric = np.squeeze(depth_raw)

        depth_metric_resized = cv2.resize(depth_metric, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return depth_metric_resized