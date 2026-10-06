"""
ONNX Runtime based Person Re-ID Model Loader.
Optimized for high-speed CPU execution using ONNX Runtime C++ backend.
"""

import os
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path

class DINOv2ExportWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.model(x)
        norm = torch.norm(emb, p=2, dim=1, keepdim=True) + 1e-8
        return emb / norm

class ONNXReIDModel:
    def __init__(self, onnx_path: str = None):
        import onnxruntime as ort
        
        base_dir = os.path.dirname(__file__)
        if not onnx_path:
            onnx_path = os.path.join(base_dir, "dinov2_vits14.onnx")
            
        self.onnx_path = onnx_path
        
        if not os.path.exists(self.onnx_path):
            print(f"Exporting DINOv2 to ONNX format at {self.onnx_path} for CPU acceleration...")
            self._export_dinov2_onnx(self.onnx_path)
            
        print(f"Loading ONNX Runtime Session: {self.onnx_path}")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, os.cpu_count() - 1)
        
        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

    def _export_dinov2_onnx(self, output_path: str):
        wrapper = DINOv2ExportWrapper()
        dummy_input = torch.randn(1, 3, 252, 126, device='cpu')
        
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False
        )
        print(f"✓ ONNX export complete: {output_path}")

    def extract_features(self, x: torch.Tensor, l2_norm: bool = True) -> torch.Tensor:
        """
        Extract features using ONNX Runtime.
        Input x: torch.Tensor (B, 3, H, W) or numpy array
        Output: torch.Tensor (B, D)
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        x_np = x_np.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: x_np})
        features = outputs[0]

        if l2_norm:
            norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
            features = features / norms

        return torch.from_numpy(features)


def load_onnx_reid_model(onnx_path: str = None) -> ONNXReIDModel:
    return ONNXReIDModel(onnx_path=onnx_path)
