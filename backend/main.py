import pathlib
import os

# Fix for Windows Store Python ACL permissions bug on parent directory traversal in ultralytics
_orig_path_exists = pathlib.Path.exists
def _safe_path_exists(self):
    try:
        return _orig_path_exists(self)
    except OSError:
        return False
pathlib.Path.exists = _safe_path_exists

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import shutil
import cv2
import numpy as np
import json
import faiss

# Import configuration
from config import THRESHOLDS, MIN_TRACK_LENGTH_FOR_GALLERY, MAX_EXEMPLARS_PER_IDENTITY, EXEMPLAR_SAMPLING_INTERVAL, CREATE_IDENTITY_THRESHOLD

# Add ai_service to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ai_service"))
from gallery_manager import VideoGalleryManager, ImageGalleryManager

# Triggering reload to pick up faiss_tracker fix
app = FastAPI(title="Re-ID API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Galleries with camera tracking
gallery_path = os.path.join(os.path.dirname(__file__), "..", "ai_service")
video_manager = VideoGalleryManager(
    index_path=os.path.join(gallery_path, "video_gallery.faiss"),
    meta_path=os.path.join(gallery_path, "video_meta.pkl"),
    camera_id="shared_video_gallery"  # Shared across all video cameras
)
image_manager = ImageGalleryManager(
    model_weights=os.path.join(gallery_path, "net_last.pth"),
    index_path=os.path.join(gallery_path, "image_gallery.faiss"),
    meta_path=os.path.join(gallery_path, "image_meta.pkl"),
    camera_id="image_search_gallery"
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

STATS_FILE = os.path.join(os.path.dirname(__file__), "stats.json")

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {
        "videos_processed": 0,
        "total_detections": 0,
        "reid_matches": 0
    }

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

def reset_gallery_state():
    """Clear gallery indexes, metadata, stats, and uploaded files."""
    # Reset in-memory galleries
    global video_manager, image_manager
    video_manager.index = faiss.IndexFlatIP(video_manager.embedding_dim)
    image_manager.index = faiss.IndexFlatIP(image_manager.embedding_dim)
    video_manager.metadata = {}
    video_manager.next_id = 0
    image_manager.metadata = {}
    image_manager.next_id = 0

    # Remove persisted gallery files if present
    for path in [video_manager.index_path, video_manager.meta_path, image_manager.index_path, image_manager.meta_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    # Reset dashboard stats
    save_stats({"videos_processed": 0, "total_detections": 0, "reid_matches": 0})

    # Clear uploaded files and generated crops
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass

    return {
        "status": "reset",
        "video_gallery_size": video_manager.index.ntotal,
        "image_gallery_size": image_manager.index.ntotal
    }

@app.get("/api/health")
def read_health():
    return {"status": "ok", "message": "Percepta-ReID API is running", "device": getattr(video_manager, 'device', 'unknown')}

@app.get("/api/config")
def get_config():
    """Get current Re-ID system configuration."""
    return {
        "thresholds": THRESHOLDS,
        "min_track_length": MIN_TRACK_LENGTH_FOR_GALLERY,
        "gallery_stats": {
            "video_gallery_size": video_manager.index.ntotal,
            "image_gallery_size": image_manager.index.ntotal,
            "video_gallery_camera": video_manager.camera_id,
            "image_gallery_camera": image_manager.camera_id
        }
    }

@app.post("/api/config/threshold")
def update_threshold(key: str, value: float):
    """Update a specific threshold value.
    
    Args:
        key: Threshold key (e.g., 'video_tracking', 'cross_camera_reid', 'gallery_search')
        value: New threshold value (0.0-1.0)
    """
    if key not in THRESHOLDS:
        raise HTTPException(status_code=400, detail=f"Unknown threshold key: {key}")
    if not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=400, detail=f"Threshold must be between 0.0 and 1.0")
    
    THRESHOLDS[key] = value
    return {
        "status": "updated",
        "key": key,
        "value": value,
        "all_thresholds": THRESHOLDS
    }

@app.get("/api/identities")
def get_identities():
    """Get all stored identities from the gallery."""
    # Convert metadata dict to a list for the frontend
    identity_list = []
    for key, value in image_manager.metadata.items():
        img_path = value.get("image_path")
        # Fallback for older entries before the gallery was made flat
        if not img_path and "exemplars" in value and len(value["exemplars"]) > 0:
            img_path = value["exemplars"][0].get("image_path")
            
        identity_list.append({
            "id": key,
            "person_id": value.get("person_id"),
            "name": value.get("name"),
            "image_path": img_path,
            "camera_id": value.get("camera_id"),  # Show which camera added this
            "model_type": value.get("model_type")
        })
    return {"identities": identity_list}

@app.get("/api/stats")
def get_stats():
    """Get dashboard high-level statistics."""
    stats = load_stats()
    # Calculate unique identities properly from video_manager.metadata
    unique_ids = len(set(v.get('person_id') for v in video_manager.metadata.values()))
    return {
        "videos_processed": stats.get("videos_processed", 0),
        "total_detections": stats.get("total_detections", 0),
        "unique_identities": unique_ids, # Unique person IDs instead of total image crops
        "reid_matches": stats.get("reid_matches", 0)
    }

@app.post("/api/reset")
def reset_all():
    """Reset identities, dashboard stats, and uploaded/generated files."""
    return reset_gallery_state()

@app.get("/api/analytics")
def get_analytics():
    """Get the evaluated metrics for the two engines."""
    eval_file = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    if os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            return json.load(f)
    return {"status": "pending", "message": "Run evaluate_models.py to generate analytics"}


@app.get("/api/identities/video/grouped")
def get_video_identities_grouped():
    """
    Get all video gallery identities grouped by UUID with exemplar counts.
    This replaces flat identity list with a grouped view showing exemplars per person.
    """
    identities = video_manager.get_all_identities()
    return {
        "identities": identities,
        "total_unique_people": len(identities),
        "total_exemplars": sum(i['num_exemplars'] for i in identities)
    }

@app.get("/api/identities/video/{identity_uuid}/exemplars")
def get_identity_exemplars(identity_uuid: str):
    """
    Get all exemplar crops and metadata for a specific identity UUID.
    Returns: {uuid, person_id, camera_id, created_at, exemplars: [{image_path, timestamp}, ...]}
    """
    exemplars = video_manager.get_identity_exemplars(identity_uuid)
    if exemplars is None:
        raise HTTPException(status_code=404, detail=f"Identity {identity_uuid} not found")
    return exemplars

@app.get("/api/identities/image/grouped")
def get_image_identities_grouped():
    """
    Get all image gallery identities (static uploads) grouped with exemplar info.
    """
    identities = image_manager.get_all_identities()
    return {
        "identities": identities,
        "total_unique_images": len(identities)
    }

class ImportDatasetRequest(BaseModel):
    data_dir: str
    max_images: int = 250
    max_per_id: int = 5

import glob

def background_import_market1501(data_dir: str, max_images: int, max_per_id: int):
    """Background task to import Market1501 dataset directly into the image manager."""
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        return

    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    if not image_paths:
        print(f"No .jpg images found in {data_dir}")
        return

    image_paths.sort()
    id_counts = {}
    uploaded = 0

    print(f"Starting background import from {data_dir}...")
    for path in image_paths:
        if uploaded >= max_images:
            break

        filename = os.path.basename(path)
        person_id = filename.split("_")[0]

        if person_id in ["-1", "0000"]:
            continue

        if id_counts.get(person_id, 0) >= max_per_id:
            continue

        name = f"Market_{person_id}"
        
        try:
            image_manager.add_identity(path, person_id, name)
            uploaded += 1
            id_counts[person_id] = id_counts.get(person_id, 0) + 1
        except Exception as e:
            print(f"Failed to import {filename}: {e}")

    print(f"Background import complete. Added {uploaded} images for {len(id_counts)} identities.")

@app.post("/api/datasets/import")
async def import_dataset(request: ImportDatasetRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(request.data_dir):
        raise HTTPException(status_code=400, detail="Directory does not exist")
        
    background_tasks.add_task(
        background_import_market1501, 
        request.data_dir, 
        request.max_images, 
        request.max_per_id
    )
    return {"status": "success", "message": "Import started in background."}

@app.post("/api/identities")
async def add_identity(
    person_id: str = Form(...),
    name: str = Form(""),
    file: UploadFile = File(...)
):
    """Add a new identity to the gallery from an image."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        idx = image_manager.add_identity(file_path, person_id, name)
        return {"status": "success", "id": idx, "person_id": person_id, "name": name}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/identities/video/{identity_uuid}")
async def update_video_identity(
    identity_uuid: str,
    person_id: str = Form(...),
    name: str = Form("")
):
    """Update the name/person_id of an existing video identity."""
    if identity_uuid not in video_manager.uuid_to_faiss_id:
        raise HTTPException(status_code=404, detail="Identity not found")
        
    faiss_id = video_manager.uuid_to_faiss_id[identity_uuid]
    video_manager.metadata[faiss_id]['person_id'] = person_id
    video_manager.metadata[faiss_id]['name'] = name
    video_manager.save_index()
    
    return {"status": "success", "uuid": identity_uuid, "person_id": person_id, "name": name}

@app.post("/api/search")
async def search_image(file: UploadFile = File(...), top_k: int = Form(5)):
    """Search for matches of an uploaded image in the gallery."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    img = cv2.imread(file_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        results = image_manager.search(img, top_k=top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image")
async def get_image(path: str):
    """Serve an image file from the local filesystem."""
    from fastapi.responses import FileResponse
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)
@app.post("/api/video/upload")
async def upload_video(
    mode: str = Form("target"), 
    file: UploadFile = File(...),
    camera_id: str = Form("camera_1"),
    threshold: float = Form(None)  # Optional override
):
    """Upload a video and return a stream URL.
    
    Args:
        mode: 'source' (add to gallery), 'target' (match against gallery), 'continuous'
        file: Video file
        camera_id: Camera identifier for metadata tracking
        threshold: Optional override for similarity threshold (uses config default if None)
    """
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Use provided threshold or config default
    if threshold is None:
        threshold = THRESHOLDS["cross_camera_reid"] if mode == "target" else THRESHOLDS["video_tracking"]
        
    return {
        "status": "success", 
        "stream_url": f"/api/video/stream?filename={file.filename}&mode={mode}&camera_id={camera_id}&threshold={threshold}"
    }

@app.get("/api/video/stream")
async def stream_video(filename: str, mode: str, camera_id: str = "camera_1", threshold: float = None):
    """Stream video processing in real-time.
    
    Args:
        filename: Video file to process
        mode: 'source', 'target', or 'continuous'
        camera_id: Camera identifier for tracking
        threshold: Similarity threshold (uses config if None)
    """
    from fastapi.responses import StreamingResponse
    from reid_pipeline import PersonReIDPipeline
    
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Use provided threshold or config default based on mode
    if threshold is None:
        threshold = THRESHOLDS["cross_camera_reid"] if mode == "target" else THRESHOLDS["video_tracking"]
        
    # Initialize pipeline with the VideoGalleryManager (strictly DINOv2)
    pipeline = PersonReIDPipeline(
        model_weights=None,
        device=video_manager.device,
        gallery_manager=video_manager,
        use_yolo=True,
        similarity_threshold=threshold,
        camera_id=camera_id
    )
    
    def frame_generator():
        # Track last frame where we sampled exemplar per person (for interval-based sampling)
        last_exemplar_frame = {}
        added_persons = set()  # Track which persons have been added to gallery
        
        frame_count = 0
        for frame_bytes in pipeline.process_video_stream(input_path, max_frames=250):
            # Exemplar-based gallery population with frame interval sampling
            if mode in ["source", "continuous"]:
                for person_id, person in pipeline.tracker.tracked_persons.items():
                    # Only add if track is stable enough
                    if person.track_len >= MIN_TRACK_LENGTH_FOR_GALLERY and getattr(person, 'best_crop', None) is not None:
                        
                        # Check if enough frames have passed since last exemplar
                        last_frame = last_exemplar_frame.get(person_id, -EXEMPLAR_SAMPLING_INTERVAL)
                        if frame_count - last_frame >= EXEMPLAR_SAMPLING_INTERVAL:
                            embeddings = np.array(person.embedding_history)
                            median_emb = np.median(embeddings, axis=0).astype(np.float32)
                            
                            # Fix: FAISS IndexFlatIP requires L2 normalized vectors for cosine similarity
                            if len(median_emb.shape) == 1:
                                median_emb = np.array([median_emb])
                            faiss.normalize_L2(median_emb)
                            
                            formatted_id = pipeline.tracker_to_gallery_map.get(person_id, f"PERSON_{person_id:04d}")
                            
                            crop_filename = f"{formatted_id}_f{frame_count:05d}.jpg"
                            crop_path = os.path.join(UPLOAD_DIR, crop_filename)
                            cv2.imwrite(crop_path, person.best_crop)
                            
                            # Add to Video Gallery with exemplar deduplication and centroid updates
                            video_manager.add_identity_embedding(
                                median_emb[0],
                                formatted_id,
                                formatted_id,
                                image_path=crop_path,
                                camera_id=camera_id,
                                match_threshold=CREATE_IDENTITY_THRESHOLD,
                                max_exemplars=MAX_EXEMPLARS_PER_IDENTITY
                            )
                            
                            last_exemplar_frame[person_id] = frame_count

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            frame_count += 1
        # Update Stats
        pipeline_stats = pipeline.get_performance_stats()
        current_stats = load_stats()
        current_stats["videos_processed"] += 1
        current_stats["total_detections"] += pipeline_stats.get("total_persons_tracked", 0)
        current_stats["reid_matches"] += len([k for k, v in pipeline.tracker_to_gallery_map.items() if not v.startswith("PERSON_")])
        save_stats(current_stats)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/video/{filename}")
async def get_video(filename: str):
    """Serve a processed video file."""
    from fastapi.responses import FileResponse
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path)


frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"status": "ok", "message": "Percepta-ReID API is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10100))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)



