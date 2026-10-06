"""
Utility functions for DINOv3-based Person Re-ID pipeline.
Based on https://github.com/layumi/Person_reID_baseline_pytorch
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional


def load_state_dict_mute(model, state_dict, strict=True):
    """
    Load state dict to model, suppressing non-critical errors.
    """
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        if strict:
            raise
        else:
            print(f"Warning: {e}")
            incompatible = model.load_state_dict(state_dict, strict=False)
            print(f"Incompatible keys: {incompatible}")


def extract_bbox_from_detection(detection, frame_shape):
    """
    Extract bounding box from detection.
    Handles YOLO format: [x_center, y_center, width, height] normalized
    """
    h, w = frame_shape[:2]
    x_center, y_center, bbox_w, bbox_h = detection[:4]
    
    # Denormalize
    x_center, y_center = int(x_center * w), int(y_center * h)
    bbox_w, bbox_h = int(bbox_w * w), int(bbox_h * h)
    
    x1 = max(0, x_center - bbox_w // 2)
    y1 = max(0, y_center - bbox_h // 2)
    x2 = min(w, x_center + bbox_w // 2)
    y2 = min(h, y_center + bbox_h // 2)
    
    return (x1, y1, x2, y2)


def crop_person_roi(frame, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Crop person region of interest (ROI) from frame.
    
    Args:
        frame: Input image frame
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Cropped person image or None if invalid
    """
    x1, y1, x2, y2 = bbox
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None
        
    return roi


def resize_and_normalize(image: np.ndarray, 
                          target_size: Tuple[int, int] = (256, 128)) -> torch.Tensor:
    """
    Resize and normalize image for model input.
    
    Args:
        image: Input image (BGR)
        target_size: Target size (height, width)
        
    Returns:
        Normalized tensor ready for model
    """
    # Resize with aspect ratio preservation
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Resize
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # Normalize ImageNet style
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor (C, H, W)
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    
    return image


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Euclidean distance between two embeddings.
    """
    return np.linalg.norm(x - y)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    Returns value in [0, 1] where 1 is identical.
    """
    x = x / (np.linalg.norm(x) + 1e-8)
    y = y / (np.linalg.norm(y) + 1e-8)
    return np.dot(x, y)


def draw_bbox_with_id(frame: np.ndarray, 
                       bbox: Tuple[int, int, int, int], 
                       person_id: int, 
                       confidence: float = 1.0,
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Draw clean, high-contrast bounding box with person ID on frame.
    """
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    
    # Prepare clean text badge
    if confidence > 1.5:
        text = f"{person_id}"
    else:
        score_pct = int(min(confidence, 1.0) * 100)
        text = f"{person_id} [{score_pct}%]"
    
    # Get text size
    font_scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Draw background badge for text
    badge_y1 = max(0, y1 - text_height - baseline - 6)
    badge_y2 = y1
    badge_x2 = min(frame.shape[1], x1 + text_width + 8)
    cv2.rectangle(
        frame,
        (x1, badge_y1),
        (badge_x2, badge_y2),
        color,
        -1
    )
    
    # Draw crisp text
    cv2.putText(
        frame,
        text,
        (x1 + 4, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
    
    return frame


def generate_unique_color(person_id: int) -> Tuple[int, int, int]:
    """
    Generate a consistent color for a given person ID.
    """
    np.random.seed(person_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


class AverageMeter:
    """Compute and store the average of a metric."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
