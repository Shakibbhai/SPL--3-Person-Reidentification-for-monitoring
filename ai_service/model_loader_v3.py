"""
DINOv3-based Person Re-ID model loader.
Compatible with Layumi Person_reID_baseline_pytorch checkpoints.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.nn import init


def weights_init_kaiming(m: nn.Module) -> None:
    """Kaiming init used by the original training code."""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m: nn.Module) -> None:
    """Classifier init used by the original training code."""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_num: int,
        droprate: float,
        linear: int = 512,
        return_f: bool = False,
    ):
        super().__init__()
        self.return_f = return_f

        add_block = []
        if linear > 0:
            add_block += [nn.Linear(input_dim, linear)]
        else:
            linear = input_dim

        add_block += [nn.BatchNorm1d(linear), nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]

        self.add_block = nn.Sequential(*add_block)
        self.add_block.apply(weights_init_kaiming)

        self.classifier = nn.Sequential(nn.Linear(linear, class_num))
        self.classifier.apply(weights_init_classifier)

        self.linear_num = linear

    def forward(self, x: torch.Tensor):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        x = self.classifier(x)
        return x


class DINoV3ReIDModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 751,
        input_size: Tuple[int, int] = (256, 128),
        droprate: float = 0.5,
        linear_num: int = 512,
    ):
        super().__init__()

        self.model = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=False,
            img_size=input_size,
            drop_path_rate=0.2,
        )
        self.model.head = nn.Sequential()

        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = ClassBlock(
            input_dim=768,
            class_num=num_classes,
            droprate=droprate,
            linear=linear_num,
            return_f=False,
        )

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = self.avgpool1d(x.permute(0, 2, 1))
        else:
            x = self.avgpool2d(x.permute(0, 3, 1, 2))
        return x.view(x.size(0), x.size(1))

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        feat = self.model.forward_features(x)
        feat = self._pool_features(feat)

        if return_feat:
            emb = self.classifier.add_block(feat)
            logits = self.classifier.classifier(emb)
            return logits, emb

        logits = self.classifier(feat)
        return logits

    def extract_features(self, x: torch.Tensor, l2_norm: bool = True) -> torch.Tensor:
        with torch.no_grad():
            _, emb = self.forward(x, return_feat=True)
            if l2_norm:
                emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def extract_features_batch(self, x_list, batch_size: int = 8) -> np.ndarray:
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


def _unwrap_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_prefix_if_present(state_dict, prefix: str):
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_pretrained_model(
    model_path: str = None,
    num_classes: int = 751,
    device: str = "cuda",
    strict: bool = False,
    **kwargs,
) -> DINoV3ReIDModel:
    model = DINoV3ReIDModel(num_classes=num_classes, **kwargs)

    if model_path and Path(model_path).exists():
        print(f"Loading trained weights from {model_path} ...")
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = _unwrap_checkpoint_state_dict(checkpoint)
        state_dict = _strip_prefix_if_present(state_dict, "module.")
        model.load_state_dict(state_dict, strict=strict)
    else:
        print(f"[Notice] Checkpoint '{model_path}' not found. Initialized model architecture for inference.")
    
    model = model.to(device)
    model.eval()
    return model

