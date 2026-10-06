# Re-ID System Configuration

# Calibrated similarity thresholds for LUPerson ViT-Base (256x128)
# Measured empirical margin:
# - Different persons: 0.30 - 0.38
# - Same person across scenes/angles: 0.45 - 0.72
THRESHOLDS = {
    "video_tracking": 0.44,      # Within video re-identification
    "cross_camera_reid": 0.44,   # Cross-camera matching
    "gallery_search": 0.45,      # Image gallery search
}

# Camera configuration
CAMERAS = {
    "camera_1": {"name": "Source Camera", "reid_threshold": 0.44},
    "camera_2": {"name": "Target Camera", "reid_threshold": 0.44},
}

# Confidence thresholds for track confirmation
MIN_TRACK_LENGTH_FOR_GALLERY = 10  # Frames before confirming gallery identity
CONFIDENCE_DECAY_RATE = 0.95       # Per-frame decay for aged tracks
MAX_TRACK_AGE = 120                # Frames before moving active track to Re-ID memory

# Identity Exemplar Storage (stores diverse templates per person for robust Re-ID)
MAX_EXEMPLARS_PER_IDENTITY = 8     # Store best N diverse samples per person
EXEMPLAR_SAMPLING_INTERVAL = 8     # Frames between exemplar captures per track
EXEMPLAR_DRIFT_THRESHOLD = 0.10    # Minimum distance to sample a new angle/pose exemplar
CREATE_IDENTITY_THRESHOLD = 0.44   # If similarity >= 0.44, merge with existing identity (don't duplicate)

# File paths
GALLERY_INDEX_PATH = "ai_service/video_gallery.faiss"
GALLERY_META_PATH = "ai_service/video_meta.pkl"
IMAGE_INDEX_PATH = "ai_service/image_gallery.faiss"
IMAGE_META_PATH = "ai_service/image_meta.pkl"
