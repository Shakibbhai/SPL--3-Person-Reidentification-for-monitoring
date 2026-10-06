"""
End-to-End Real-Time Person Re-ID Pipeline
Integrates detection, feature extraction, and tracking
"""

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

import cv2
import torch
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict
import time
from collections import deque

from model_loader import load_pretrained_model
from faiss_tracker import FAISSPersonTracker
from utils import (
    resize_and_normalize, crop_person_roi, draw_bbox_with_id,
    generate_unique_color, AverageMeter
)


class PersonReIDPipeline:
    """
    End-to-end person re-identification pipeline for video.
    
    Pipeline:
    1. Read frame from video/camera
    2. Detect persons using YOLO
    3. Extract features using DINOv2 model
    4. Track persons using FAISS
    5. Handle re-identification when person returns
    6. Visualize results on frame
    """
    
    def __init__(self,
                 model_weights: str,
                 num_classes: int = 751,
                 device: str = 'cuda',
                 similarity_threshold: float = 0.44,
                 max_age: int = 120,
                 use_yolo: bool = True,
                 yolo_model: str = 'yolov8x.pt',
                 gallery_manager=None,
                 camera_id: str = None):
        """
        Initialize Re-ID pipeline.
        
        Args:
            model_weights: Path to trained weights
            num_classes: Number of identity classes
            device: 'cuda' or 'cpu'
            similarity_threshold: Threshold for person re-matching
            max_age: Max frames before removing track
            use_yolo: Whether to use YOLO for detection
            yolo_model: YOLO model name or path
            gallery_manager: Gallery manager for cross-camera re-id
            camera_id: Camera identifier for tracking provenance
        """
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.camera_id = camera_id or "unknown"
        
        print(f"Initializing Re-ID Pipeline for {self.camera_id}...")
        
        if gallery_manager:
            print("[1/3] Using model from GalleryManager...")
            self.model = gallery_manager.model
        else:
            print("[1/3] Loading DINOv2 model...")
            self.model = load_pretrained_model(
                model_weights,
                num_classes=num_classes,
                device=device
            )
        
        self.gallery_manager = gallery_manager
        self.tracker_to_gallery_map = {}
        
        # Initialize tracker
        print("[2/3] Initializing FAISS tracker...")
        emb_dim = gallery_manager.embedding_dim if gallery_manager else 768
        self.tracker = FAISSPersonTracker(
            embedding_dim=emb_dim,
            max_age=max_age,
            similarity_threshold=0.55,
            reid_threshold=similarity_threshold
        )
        
        # Initialize YOLO detector
        print("[3/3] Loading YOLO detector...")
        self.use_yolo = use_yolo
        if use_yolo:
            try:
                from ultralytics import YOLO
                yolo_path = yolo_model
                if not os.path.exists(yolo_path):
                    root_yolo = os.path.join(os.path.dirname(__file__), "..", yolo_model)
                    if os.path.exists(root_yolo):
                        yolo_path = root_yolo
                self.detector = YOLO(yolo_path)
                self.detector.to(device)
                print(f"✓ YOLO loaded: {yolo_path}")
            except Exception as e:
                print(f"⚠ YOLO loading error: {e}, will use mock detections")
                self.detector = None
        else:
            self.detector = None
        
        # Performance metrics
        self.fps_meter = AverageMeter()
        self.detection_time = AverageMeter()
        self.feature_extraction_time = AverageMeter()
        self.tracking_time = AverageMeter()
        
        print("✓ Pipeline initialized successfully!\n")
    
    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> Dict:
        """
        Process a single frame for re-identification.
        
        Args:
            frame: Input frame (H, W, 3) in BGR
            frame_id: Frame number for logging
            
        Returns:
            Dict with results:
            {
                'frame': annotated frame,
                'detections': list of (bbox, person_id, confidence),
                'num_active_persons': number of active person tracks,
                'metrics': performance metrics
            }
        """
        start_time = time.time()
        
        h, w = frame.shape[:2]
        
        # Step 1: Detect persons
        det_start = time.time()
        person_detections = self._detect_persons(frame)
        det_time = time.time() - det_start
        self.detection_time.update(det_time)
        
        # Step 2: Extract features
        feat_start = time.time()
        detections_with_features = self._extract_features(frame, person_detections)
        feat_time = time.time() - feat_start
        self.feature_extraction_time.update(feat_time)
        
        # Step 3: Track and assign IDs
        track_start = time.time()
        bbox_to_id = self.tracker.update(detections_with_features, frame_id=frame_id)
        track_time = time.time() - track_start
        self.tracking_time.update(track_time)
        
        # Step 4: Visualize
        annotated_frame = frame.copy()
        person_results = []
        
        # Keep track of which gallery IDs have been assigned THIS frame to prevent feature pollution (duplicate IDs)
        assigned_gallery_ids = set()
        
        # Iterate through detections and match with assigned IDs
        for embedding, bbox in detections_with_features:
            if bbox not in bbox_to_id:
                continue
            
            person_id = bbox_to_id[bbox]
            
            # Map tracker ID to Gallery Identity if available
            if self.gallery_manager and self.gallery_manager.index.ntotal > 0:
                query_emb = np.array([embedding]).astype(np.float32)
                query_emb = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8)
                faiss.normalize_L2(query_emb)
                distances, indices = self.gallery_manager.index.search(query_emb, 1)
                dist = float(distances[0][0])
                idx = int(indices[0][0])
                
                if idx in self.gallery_manager.metadata:
                    gallery_person_id = self.gallery_manager.metadata[idx]['person_id']
                    
                    # Mutual Exclusion Rule: Assign if confidence is high AND nobody else in this frame stole the ID
                    if dist >= self.similarity_threshold and gallery_person_id not in assigned_gallery_ids:
                        self.tracker_to_gallery_map[person_id] = gallery_person_id
                        assigned_gallery_ids.add(gallery_person_id)
                    elif person_id in self.tracker_to_gallery_map and self.tracker_to_gallery_map[person_id] not in assigned_gallery_ids:
                        # STICKY: Maintain previously confirmed gallery identity across temporary frame fluctuations
                        assigned_gallery_ids.add(self.tracker_to_gallery_map[person_id])
                    else:
                        if person_id not in self.tracker_to_gallery_map:
                            self.tracker_to_gallery_map[person_id] = f"PERSON_{person_id:04d}"
                else:
                    if person_id not in self.tracker_to_gallery_map:
                        self.tracker_to_gallery_map[person_id] = f"PERSON_{person_id:04d}"
            else:
                if person_id not in self.tracker_to_gallery_map:
                    self.tracker_to_gallery_map[person_id] = f"PERSON_{person_id:04d}"
            
            display_text = self.tracker_to_gallery_map[person_id]
            if display_text in assigned_gallery_ids and display_text != f"PERSON_{person_id:04d}":
                display_text = f"PERSON_{person_id:04d}"
            assigned_gallery_ids.add(display_text)
            
            # Get confidence from tracker
            person = self.tracker.tracked_persons.get(person_id)
            confidence = person.confidence if person else 1.0
            
            # Save crop for gallery population
            if person:
                crop = crop_person_roi(frame, bbox)
                if crop is not None:
                    # Save the crop (simplistic logic: always overwrite with latest, could be improved to save highest res)
                    person.best_crop = crop
            
            # Draw on frame
            color = generate_unique_color(person_id)
            annotated_frame = draw_bbox_with_id(
                annotated_frame, bbox, display_text, confidence, color
            )
            
            person_results.append((bbox, display_text, confidence))
        
        # Calculate FPS
        total_time = time.time() - start_time
        fps = 1.0 / (total_time + 1e-6)
        self.fps_meter.update(fps)
        
        # Add FPS and stats to frame
        stats_text = [
            f"FPS: {self.fps_meter.avg:.1f}",
            f"Active: {len(self.tracker.tracked_persons)}",
            f"Total: {self.tracker.next_person_id}"
        ]
        
        y_offset = 30
        for i, text in enumerate(stats_text):
            cv2.putText(
                annotated_frame, text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
        
        results = {
            'frame': annotated_frame,
            'detections': person_results,
            'num_active_persons': len(self.tracker.tracked_persons),
            'num_new_persons': self.tracker.next_person_id,
            'metrics': {
                'fps': fps,
                'detection_time': det_time,
                'feature_extraction_time': feat_time,
                'tracking_time': track_time,
                'total_time': total_time
            }
        }
        
        return results
    
    def _detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect persons in frame using YOLO.
        
        Args:
            frame: Input frame
            
        Returns:
            List of bounding boxes (x1, y1, x2, y2)
        """
        if not self.detector:
            return []
        
        try:
            # YOLO inference with standard confidence threshold (0.25) to detect all people
            results = self.detector(frame, conf=0.25, imgsz=640, classes=0, verbose=False)  # class 0 = person
            
            bboxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    bboxes.append((x1, y1, x2, y2))
            
            return bboxes
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _extract_features(self, 
                         frame: np.ndarray,
                         bboxes: List[Tuple]) -> List[Tuple[np.ndarray, Tuple]]:
        """
        Extract features for detected persons.
        
        Args:
            frame: Input frame
            bboxes: List of bounding boxes
            
        Returns:
            List of (feature_embedding, bbox) tuples
        """
        detections_with_features = []
        
        if not bboxes or self.model is None:
            return detections_with_features
        
        # Crop and prepare batch
        person_crops = []
        valid_bboxes = []
        
        for bbox in bboxes:
            roi = crop_person_roi(frame, bbox)
            if roi is not None:
                person_crops.append(roi)
                valid_bboxes.append(bbox)
        
        if not person_crops:
            return detections_with_features
        
        # Convert crops to tensors
        input_tensors = []
        target_sz = (252, 126) if hasattr(self.model, 'session') else (256, 128)
        input_tensors = []
        for crop in person_crops:
            tensor = resize_and_normalize(crop, target_size=target_sz)
            input_tensors.append(tensor)
        
        # Extract features in batch
        try:
            batch = torch.cat(input_tensors, dim=0)
            if hasattr(self.model, 'session'):
                features = self.model.extract_features(batch)
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
            else:
                with torch.no_grad():
                    if batch.is_cuda or self.device == 'cuda':
                        batch = batch.to(self.device)
                    features = self.model.extract_features(batch)
                    features = features.cpu().numpy()
            
            # Pair features with bboxes
            for feature, bbox in zip(features, valid_bboxes):
                detections_with_features.append((feature, bbox))
        
        except Exception as e:
            print(f"Feature extraction error: {e}")
        
        return detections_with_features
    
    def process_video(self, 
                      video_path: str,
                      output_path: Optional[str] = None,
                      max_frames: int = None,
                      display: bool = True):
        """
        Process entire video and optionally save output.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            max_frames: Max frames to process (None = all)
            display: Whether to display frames in real-time
        """
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {frame_width}x{frame_height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Setup video writer
        out = None
        if output_path:
            # Determine codec based on extension
            if output_path.endswith('.webm'):
                fourcc = cv2.VideoWriter_fourcc(*'vp80')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if max_frames and frame_count >= max_frames:
                    break
                
                # Process frame
                results = self.process_frame(frame, frame_id=frame_count)
                
                # Write output
                if out:
                    out.write(results['frame'])
                
                # Display
                if display:
                    cv2.imshow('Person Re-ID Pipeline', results['frame'])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print stats every 30 frames
                if (frame_count + 1) % 30 == 0:
                    print(f"Frame {frame_count + 1}/{total_frames} | "
                          f"FPS: {self.fps_meter.avg:.1f} | "
                          f"Active persons: {results['num_active_persons']} | "
                          f"Total tracked: {results['num_new_persons']}")
                
                frame_count += 1
        
        finally:
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"\n✓ Processing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {self.fps_meter.avg:.2f}")

    def process_video_stream(self, video_path: str, max_frames: int = None):
        """
        Process video and yield JPEG frames continuously with paced real-time streaming.
        """
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # Target pacing interval (clamp between 20 and 30 FPS for smooth browser rendering)
        target_fps = max(20.0, min(30.0, video_fps))
        frame_interval = 1.0 / target_fps
        frame_count = 0
        
        try:
            while True:
                t_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_count >= max_frames:
                    break
                    
                results = self.process_frame(frame, frame_id=frame_count)
                annotated_frame = results['frame']
                
                # Optimize resolution for web streaming if larger than 720p to eliminate network lag
                h, w = annotated_frame.shape[:2]
                if w > 1280:
                    stream_img = cv2.resize(annotated_frame, (1280, int(h * 1280 / w)))
                else:
                    stream_img = annotated_frame
                
                # Encode frame to JPEG with optimal streaming compression
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                ret, buffer = cv2.imencode('.jpg', stream_img, encode_param)
                if ret:
                    yield buffer.tobytes()
                    
                frame_count += 1

                # Frame pacing to ensure butter-smooth real-time playback
                t_elapsed = time.time() - t_start
                sleep_sec = frame_interval - t_elapsed
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
        finally:
            cap.release()
        print(f"Total unique persons tracked: {self.tracker.next_person_id}")
    
    def process_camera(self, camera_id: int = 0, display: bool = True):
        """
        Process real-time camera stream.
        
        Args:
            camera_id: Camera device ID (0 for default)
            display: Whether to display frames
        """
        print(f"Opening camera {camera_id}...")
        
        cap = cv2.VideoCapture(camera_id)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera info: {frame_width}x{frame_height} @ {fps:.1f} FPS")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Cannot read frame")
                    break
                
                # Process frame
                results = self.process_frame(frame, frame_id=frame_count)
                
                # Display
                if display:
                    cv2.imshow('Person Re-ID Pipeline', results['frame'])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print stats every 30 frames
                if (frame_count + 1) % 30 == 0:
                    print(f"Frame {frame_count + 1} | "
                          f"FPS: {self.fps_meter.avg:.1f} | "
                          f"Active: {results['num_active_persons']} | "
                          f"Total: {results['num_new_persons']}")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
        
        print(f"✓ Camera stream ended after {frame_count} frames")
    
    def get_performance_stats(self) -> Dict:
        """Get pipeline performance statistics."""
        return {
            'avg_fps': self.fps_meter.avg,
            'avg_detection_time': self.detection_time.avg,
            'avg_feature_extraction_time': self.feature_extraction_time.avg,
            'avg_tracking_time': self.tracking_time.avg,
            'total_persons_tracked': self.tracker.next_person_id,
            'active_persons': len(self.tracker.tracked_persons)
        }
    
    def save_tracker_state(self, output_path: str):
        """Save tracker embeddings and state."""
        self.tracker.save_embeddings(output_path)
        
        # Also save statistics
        stats_path = output_path.replace('.pkl', '_stats.txt')
        with open(stats_path, 'w') as f:
            stats = self.get_performance_stats()
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ Tracker state saved to {output_path}")


if __name__ == '__main__':
    import sys
    
    # Example usage
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Initialize pipeline
        pipeline = PersonReIDPipeline(
            model_weights='net_last.pth',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            similarity_threshold=0.65
        )
        
        # Process video
        pipeline.process_video(
            video_path,
            output_path=output_path,
            display=True
        )
    else:
        print("Usage: python reid_pipeline.py <video_path> [output_path]")
