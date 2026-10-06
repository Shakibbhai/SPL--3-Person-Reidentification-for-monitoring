"""
LUPerson ViT-Base (Vision Transformer) Person Re-ID Model Loader
Loads weights from checkpoint0260.pth / vit_base_checkpoint0260.pth
Trained on LUPerson with (256, 128) pedestrian crops
"""

import os
import torch
import torch.nn as nn
import numpy as np


class ViTLUPersonModel(nn.Module):
    def __init__(self, model_path=None, device="cuda"):
        super().__init__()
        import timm
        self.device = device
        self.embedding_dim = 768
        
        # Build standard ViT-Base for 256x128 person Re-ID
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            img_size=(256, 128),
            num_classes=0
        )

        # Candidate paths to locate checkpoint
        candidates = [
            model_path,
            os.path.join(os.path.dirname(__file__), "..", "vit_base_checkpoint0260.pth"),
            os.path.join(os.path.dirname(__file__), "vit_base_checkpoint0260.pth"),
            os.path.join(os.path.dirname(__file__), "..", "checkpoint0260.pth"),
            os.path.join(os.path.dirname(__file__), "checkpoint0260.pth"),
            "/workspace/Percepta-ReID/vit_base_checkpoint0260.pth",
            "/workspace/actions-runner/_work/SPL--3-Person-Reidentification-for-monitoring/SPL--3-Person-Reidentification-for-monitoring/backend/vit_base_checkpoint0260.pth",
            "/workspace/actions-runner/_work/SPL--3-Person-Reidentification-for-monitoring/SPL--3-Person-Reidentification-for-monitoring/vit_base_checkpoint0260.pth"
        ]
        
        loaded = False
        for p in candidates:
            if p and os.path.exists(p):
                print(f"[ViT-Base] Loading pretrained LUPerson weights from: {p}")
                try:
                    ckpt = torch.load(p, map_location="cpu", weights_only=False)
                    if "state_dict" in ckpt:
                        state_dict = ckpt["state_dict"]
                    elif "teacher" in ckpt:
                        state_dict = {
                            k[len("backbone."):]: v 
                            for k, v in ckpt["teacher"].items() 
                            if k.startswith("backbone.")
                        }
                    else:
                        state_dict = ckpt
                    
                    msg = self.model.load_state_dict(state_dict, strict=False)
                    print(f"✓ [ViT-Base] Pretrained weights loaded successfully: {msg}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠ [ViT-Base] Failed to load from {p}: {e}")

        if not loaded:
            print("⚠ [ViT-Base] Notice: Checkpoint not found in paths. Running with initialized architecture.")

        self.model = self.model.to(device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor, l2_norm: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if not x.is_cuda and self.device == "cuda":
                x = x.to(self.device)
            feat = self.forward(x)
            if l2_norm:
                feat = nn.functional.normalize(feat, p=2, dim=1)
        return feat

    def extract_features_batch(self, x_list, batch_size: int = 16) -> np.ndarray:
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
            if self.device == "cuda":
                batch = batch.cuda(non_blocking=True)
            feat = self.extract_features(batch, l2_norm=True)
            all_features.append(feat.cpu().numpy())
        return np.vstack(all_features)
