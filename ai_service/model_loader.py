"""
DINOv2-based Person Re-ID model loader.
Uses the robust official facebookresearch/dinov2 foundation model.
"""

import torch
import torch.nn as nn
import numpy as np

class DINOv2ReIDModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        print("Loading DINOv2 ViT-S/14 from torch hub...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model = self.model.to(device)
        self.model.eval()
        
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def extract_features(self, x: torch.Tensor, l2_norm: bool = True) -> torch.Tensor:
        """Extract embedding vectors for Re-ID."""
        with torch.no_grad():
            emb = self.forward(x)
            if l2_norm:
                emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def extract_features_batch(self, x_list, batch_size: int = 8) -> np.ndarray:
        """Batch helper for list/tensor input."""
        all_features = []
        for i in range(0, len(x_list), batch_size):
            batch = x_list[i : i + batch_size]
            if isinstance(batch[0], np.ndarray):
                batch = torch.stack([
                    torch.from_numpy(img) if isinstance(img, np.ndarray) else img
                    for img in batch
                ])
            else:
                batch = torch.stack(batch)

            if next(self.parameters()).is_cuda:
                batch = batch.cuda(non_blocking=True)

            features = self.extract_features(batch, l2_norm=True)
            all_features.append(features.cpu().numpy())

        return np.vstack(all_features)


def load_pretrained_model(
    model_path: str = None,
    num_classes: int = 0,
    device: str = "cuda",
    strict: bool = False,
    **kwargs,
) -> DINOv2ReIDModel:
    """
    Load DINOv2 Re-ID model. Ignores model_path as DINOv2 is a foundation model loaded via hub.
    """
    model = DINOv2ReIDModel(device=device)
    return model

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pretrained_model(device=dev)

    # DINOv2 expects inputs to be divisible by patch size (14)
    # Using 252 x 126 instead of 256x128
    x = torch.randn(2, 3, 252, 126, device=dev)
    with torch.no_grad():
        features = model.extract_features(x)

    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print("Model test passed with DINOv2.")
