"""OFIQ ONNX face quality scorer (Stage 3 ranking)."""

import cv2
import numpy as np
import onnxruntime as ort
import torch


class OFIQScorer:
    def __init__(self, model_path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        providers = (
            ["CUDAExecutionProvider"]
            if device == "cuda" and ort.get_device() == "GPU"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (112, 112)

    def _preprocess(self, face_img):
        img = cv2.resize(face_img, self.input_shape)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img[np.newaxis, ...]

    def get_score(self, face_img):
        if face_img is None or face_img.size == 0:
            return 0.0
        try:
            output = self.session.run(None, {self.input_name: self._preprocess(face_img)})
            return float(output[0].item())
        except Exception as e:
            print(f"OFIQ scoring error: {e}")
            return 0.0
