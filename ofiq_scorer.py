"""OFIQ ONNX face quality scorer (Stage 3 ranking)."""

import os
import sys

import cv2
import numpy as np
import torch


def _ensure_torch_cuda_dlls():
    """On Windows, onnxruntime-gpu often needs torch's CUDA/cuDNN DLL dirs on PATH."""
    if sys.platform != "win32":
        return
    try:
        base = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(base):
            os.add_dll_directory(base)
            path = os.environ.get("PATH", "")
            if base not in path:
                os.environ["PATH"] = base + os.pathsep + path
    except Exception:
        pass


_ensure_torch_cuda_dlls()
import onnxruntime as ort  # noqa: E402


class OFIQScorer:
    def __init__(self, model_path, device=None):
        want_cuda = (device or ("cuda" if torch.cuda.is_available() else "cpu")) == "cuda"
        available = ort.get_available_providers()
        if want_cuda and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        active = self.session.get_providers()
        print(f"OFIQ providers available={available} active={active}")
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
