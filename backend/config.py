# Re-ID System Configuration

# Dynamic similarity thresholds per camera
# Range: 0.0 (very permissive) to 1.0 (very strict)
# Recommended: 0.40-0.60 for DINOv2

THRESHOLDS = {
    "video_tracking": 0.65,      # Within same camera: higher threshold = stricter
    "cross_camera_reid": 0.50,   # Cross-camera matching: more lenient
    "gallery_search": 0.55,      # Image gallery search: medium threshold
}

# Camera configuration
CAMERAS = {
    "camera_1": {"name": "Source Camera", "reid_threshold": 0.50},
    "camera_2": {"name": "Target Camera", "reid_threshold": 0.50},
}

# Confidence thresholds for track confirmation
MIN_TRACK_LENGTH_FOR_GALLERY = 15  # Frames before adding to gallery
CONFIDENCE_DECAY_RATE = 0.95       # Per-frame decay for aged tracks
MAX_TRACK_AGE = 90                 # Frames before removing track

# Identity Exemplar Storage (prevents duplicates and gallery clutter)
MAX_EXEMPLARS_PER_IDENTITY = 5     # Store best N samples per person, not all crops
EXEMPLAR_SAMPLING_INTERVAL = 10    # Frames between exemplar captures per track
EXEMPLAR_DRIFT_THRESHOLD = 0.12    # L2 distance threshold for sampling new exemplar (avoid duplicates)
CREATE_IDENTITY_THRESHOLD = 0.65   # Match confidence needed to assign to existing identity (not create new)

# File paths
GALLERY_INDEX_PATH = "ai_service/video_gallery.faiss"
GALLERY_META_PATH = "ai_service/video_meta.pkl"
IMAGE_INDEX_PATH = "ai_service/image_gallery.faiss"
IMAGE_META_PATH = "ai_service/image_meta.pkl"
