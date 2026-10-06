"""
Real-time Person Tracking & Re-ID using FAISS
Manages person embeddings and handles re-identification across frames
"""

import numpy as np
import faiss
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
from pathlib import Path


@dataclass
class TrackedPerson:
    """Stores information about a tracked person."""
    person_id: int
    embedding: np.ndarray  # Latest embedding
    last_frame_id: int
    track_len: int  # Number of frames this person has been tracked
    confidence: float  # Confidence of latest match
    bbox_history: deque  # Store last N bounding boxes for trajectory
    embedding_history: deque  # Store all embeddings for this person
    best_crop: np.ndarray = None  # Store the best visual crop of this person
    
    def __post_init__(self):
        if self.bbox_history is None:
            self.bbox_history = deque(maxlen=30)
        if self.embedding_history is None:
            self.embedding_history = deque(maxlen=50)


class FAISSPersonTracker:
    """
    FAISS-based person tracker for video re-identification.
    
    Features:
    - Real-time embedding storage using FAISS
    - Automatic person ID assignment
    - Track continuity across frames
    - Re-identification when person leaves and returns
    - Trajectory history for each person
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 max_age: int = 120,
                 min_matches: int = 2,
                 similarity_threshold: float = 0.55,
                 reid_threshold: float = 0.44):
        """
        Initialize FAISS tracker.
        
        Args:
            embedding_dim: Dimension of person embeddings
            max_age: Max frames a person can be absent before removal
            min_matches: Min matches required before confirming re-ID
            similarity_threshold: Similarity threshold for active track matching (cosine)
            reid_threshold: Similarity threshold for re-ID memory matching (more lenient)
        """
        self.embedding_dim = embedding_dim
        self.max_age = max_age
        self.min_matches = min_matches
        self.similarity_threshold = similarity_threshold
        self.reid_threshold = reid_threshold
        
        # FAISS index for fast nearest neighbor search
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine)
        
        # Store all embeddings added to index
        self.embedding_database = []  # List of (embedding, person_id) tuples
        
        # Track current persons in frame
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        
        # Global person ID counter
        self.next_person_id = 0
        
        # Frame counter
        self.frame_count = 0
        
        # Re-ID memory - store historical embeddings for disappeared persons
        self.reid_memory: Dict[int, List[np.ndarray]] = defaultdict(list)
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit sphere."""
        return embedding / (np.linalg.norm(embedding) + 1e-8)
        
    def _calculate_iou(self, boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def update(self, 
               detections: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
               frame_id: Optional[int] = None) -> Dict[Tuple[int, int, int, int], int]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (embedding, bbox) tuples
                       embedding: (512,) numpy array
                       bbox: (x1, y1, x2, y2) tuple
            frame_id: Optional frame ID for logging
            
        Returns:
            Dict mapping bbox to assigned person_id
        """
        if frame_id is not None:
            self.frame_count = frame_id
        else:
            self.frame_count += 1
        
        # Normalize all detections
        detections = [
            (self._normalize_embedding(emb), bbox) 
            for emb, bbox in detections
        ]
        
        # Stage 1: Spatial Tracking (IoU)
        assignments = {}
        matched_detections = set()
        matched_persons = set()
        
        for detection_idx, (_, bbox) in enumerate(detections):
            best_iou = 0.3  # Minimum IoU threshold for spatial match
            best_person_id = None
            
            for person_id, person in self.tracked_persons.items():
                if person_id in matched_persons:
                    continue
                if person.bbox_history:
                    last_bbox = person.bbox_history[-1]
                    iou = self._calculate_iou(bbox, last_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_person_id = person_id
            
            if best_person_id is not None:
                assignments[detection_idx] = best_person_id
                matched_detections.add(detection_idx)
                matched_persons.add(best_person_id)
                self.tracked_persons[best_person_id].confidence = 2.0  # Flag as Spatial Match
        
        # Build list of unmatched detections for Stage 2
        unmatched_detections_idx = [i for i in range(len(detections)) if i not in matched_detections]
        unmatched_detections = [(i, detections[i]) for i in unmatched_detections_idx]
        
        # Stage 2: Appearance Tracking (FAISS) for remaining unassigned tracks
        appearance_assignments = self._match_detections_appearance(unmatched_detections, matched_persons)
        assignments.update(appearance_assignments)
        
        # Update results
        bbox_to_id = {}
        matched_person_ids = set()
        
        # Process matched detections
        for detection_idx, person_id in assignments.items():
            embedding, bbox = detections[detection_idx]
            bbox_to_id[bbox] = person_id
            matched_person_ids.add(person_id)
            
            # Update existing track
            self.tracked_persons[person_id].embedding = embedding
            self.tracked_persons[person_id].last_frame_id = self.frame_count
            self.tracked_persons[person_id].track_len += 1
            self.tracked_persons[person_id].bbox_history.append(bbox)
            self.tracked_persons[person_id].embedding_history.append(embedding)
            
            # Update re-ID memory
            self.reid_memory[person_id].append(embedding)
        
        # For completely unmatched detections: try re-ID memory (disappeared persons), then create new track
        completely_unmatched = [
            (detection_idx, detections[detection_idx])
            for detection_idx in range(len(detections))
            if detection_idx not in assignments
        ]
        
        # Try to match against re-ID memory (disappeared persons)
        reid_assignments = self._match_reid_memory(completely_unmatched)
        
        for detection_idx, person_id in reid_assignments.items():
            embedding, bbox = detections[detection_idx]
            bbox_to_id[bbox] = person_id
            matched_person_ids.add(person_id)
            
            # Re-activate the track by recreating TrackedPerson
            self.tracked_persons[person_id] = TrackedPerson(
                person_id=person_id,
                embedding=embedding,
                last_frame_id=self.frame_count,
                track_len=1,  # Could restore old track len, but resetting is safer
                confidence=1.0,
                bbox_history=deque(maxlen=30),
                embedding_history=deque(maxlen=50)
            )
            
            self.tracked_persons[person_id].bbox_history.append(bbox)
            self.tracked_persons[person_id].embedding_history.append(embedding)
            
            # Update re-ID memory
            self.reid_memory[person_id].append(embedding)
        
        # Create new tracks for still-unmatched detections
        for detection_idx, (embedding, bbox) in enumerate(detections):
            if detection_idx not in assignments and detection_idx not in reid_assignments:
                person_id = self._create_new_track(embedding, bbox)
                bbox_to_id[bbox] = person_id
                matched_person_ids.add(person_id)
        
        # Stage 4: Deferred Re-Association for recently created tracks
        # If an active track is newly spawned (track_len between 3 and 25 frames) with a high ID,
        # verify if its accumulated embedding history matches an earlier person from reid_memory!
        for person_id in list(self.tracked_persons.keys()):
            person = self.tracked_persons.get(person_id)
            if person and 3 <= person.track_len <= 25 and len(person.embedding_history) >= 3:
                recent_embs = np.array(list(person.embedding_history)[-10:]).astype(np.float32)
                recent_med = np.median(recent_embs, axis=0)
                recent_med = recent_med / (np.linalg.norm(recent_med) + 1e-8)
                
                best_match_id = None
                best_sim = self.reid_threshold
                
                for past_id, past_embs in self.reid_memory.items():
                    if past_id in self.tracked_persons or past_id >= person_id:
                        continue
                    for pe in past_embs:
                        sim = float(np.dot(recent_med, pe))
                        if sim > best_sim:
                            best_sim = sim
                            best_match_id = past_id
                
                if best_match_id is not None and best_match_id not in matched_person_ids:
                    # MERGE track back to original identity
                    old_p = self.tracked_persons.pop(person_id)
                    old_p.person_id = best_match_id
                    self.tracked_persons[best_match_id] = old_p
                    matched_person_ids.add(best_match_id)
                    
                    for b, pid in list(bbox_to_id.items()):
                        if pid == person_id:
                            bbox_to_id[b] = best_match_id
                    
                    # Merge historical memory
                    if person_id in self.reid_memory:
                        self.reid_memory[best_match_id].extend(self.reid_memory.pop(person_id))

        # Remove tracks that are too old
        self._remove_old_tracks()
        
        return bbox_to_id
    
    def _match_detections_appearance(self, 
                                     unmatched_detections: List[Tuple[int, Tuple[np.ndarray, Tuple]]],
                                     already_matched_persons: set) -> Dict[int, int]:
        """
        Match detections to existing active tracks using FAISS visual appearance.
        
        Returns:
            Dict mapping detection_idx to person_id
        """
        assignments = {}
        
        if not self.tracked_persons or not unmatched_detections:
            return assignments
        
        # Prepare embeddings for search
        query_embeddings = np.array([emb for _, (emb, _) in unmatched_detections]).astype(np.float32)
        # Already normalized in update()
        
        # Prepare database embeddings of active tracks NOT ALREADY MATCHED spatially
        database_embeddings = []
        person_ids_in_db = []
        
        for person_id, person in self.tracked_persons.items():
            if person_id not in already_matched_persons:
                database_embeddings.append(person.embedding)
                person_ids_in_db.append(person_id)
        
        if not database_embeddings:
            return assignments
        
        database_embeddings = np.array(database_embeddings).astype(np.float32)
        database_embeddings = database_embeddings / (np.linalg.norm(database_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Search for nearest neighbors
        distances, indices = self._search_faiss(query_embeddings, database_embeddings, k=1)
        
        # Process matches
        matched_detections = set()
        matched_persons = set()
        
        for i, (dist, idx) in enumerate(zip(distances[:, 0], indices[:, 0])):
            similarity = float(dist)
            detection_idx = unmatched_detections[i][0]
            
            if similarity >= self.similarity_threshold:
                person_id = person_ids_in_db[idx]
                
                if detection_idx not in matched_detections and person_id not in matched_persons:
                    assignments[detection_idx] = person_id
                    matched_detections.add(detection_idx)
                    matched_persons.add(person_id)
                    
                    self.tracked_persons[person_id].confidence = similarity
        
        return assignments
    
    def _match_reid_memory(self, 
                           unmatched_detections: List[Tuple[int, Tuple[np.ndarray, Tuple]]]) -> Dict[int, int]:
        """
        Try to re-identify disappeared persons by matching against re-ID memory.
        
        Args:
            unmatched_detections: List of (detection_idx, (embedding, bbox))
            
        Returns:
            Dict mapping detection_idx to original person_id (from disappeared tracks)
        """
        assignments = {}
        
        if not unmatched_detections or not self.reid_memory:
            return assignments
        
        # Build database from re-ID memory with diverse exemplar templates per disappeared person
        database_embeddings = []
        person_ids_in_memory = []
        
        for person_id, embeddings in self.reid_memory.items():
            if person_id in self.tracked_persons:
                # Skip active persons (already matched in spatial or appearance stage)
                continue
            
            if not embeddings:
                continue
            
            embs_arr = np.array(embeddings).astype(np.float32)
            embs_arr = embs_arr / (np.linalg.norm(embs_arr, axis=1, keepdims=True) + 1e-8)
            
            # Extract diverse view exemplars (first view, recent view, median, and distinct angle views)
            chosen_embs = [embs_arr[0]]
            if len(embs_arr) > 1:
                chosen_embs.append(embs_arr[-1])
            if len(embs_arr) >= 4:
                median_emb = np.median(embs_arr, axis=0)
                median_emb = median_emb / (np.linalg.norm(median_emb) + 1e-8)
                chosen_embs.append(median_emb)
            
            # Add up to 5 additional distinct view samples (cosine distance > 0.10)
            for e in embs_arr:
                if len(chosen_embs) >= 8:
                    break
                if all(float(np.dot(e, ce)) < 0.90 for ce in chosen_embs):
                    chosen_embs.append(e)
            
            for ce in chosen_embs:
                database_embeddings.append(ce)
                person_ids_in_memory.append(person_id)
        
        if not database_embeddings:
            return assignments
        
        # Prepare query embeddings
        query_embeddings = np.array([emb for idx, (emb, _) in unmatched_detections]).astype(np.float32)
        database_embeddings = np.array(database_embeddings).astype(np.float32)
        
        # Search for nearest neighbor in re-ID memory
        distances, indices = self._search_faiss(query_embeddings, database_embeddings, k=1)
        
        matched_detections = set()
        matched_persons = set()
        
        for i, (unmatched_idx, unmatched_det) in enumerate(unmatched_detections):
            if distances.shape[0] <= i:
                break
            
            similarity = float(distances[i, 0])
            idx = int(indices[i, 0])
            
            # Use more lenient threshold for re-identification across longer gaps
            if similarity >= self.reid_threshold:
                person_id = person_ids_in_memory[idx]
                
                # Avoid duplicate assignments
                if unmatched_idx not in matched_detections and person_id not in matched_persons:
                    assignments[unmatched_idx] = person_id
                    matched_detections.add(unmatched_idx)
                    matched_persons.add(person_id)
        
        return assignments
    
    def _search_faiss(self, query: np.ndarray, 
                      database: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using FAISS index built on-the-fly.
        
        Args:
            query: Query embeddings (N, D)
            database: Database embeddings (M, D)
            k: Number of nearest neighbors
            
        Returns:
            (distances, indices) where distances are similarities
        """
        # Create temporary index
        index = faiss.IndexFlatIP(database.shape[1])
        index.add(database)
        
        # Search
        distances, indices = index.search(query, k)
        
        return distances, indices
    
    def _create_new_track(self, embedding: np.ndarray, bbox: Tuple) -> int:
        """Create a new person track."""
        person_id = self.next_person_id
        self.next_person_id += 1
        
        person = TrackedPerson(
            person_id=person_id,
            embedding=embedding,
            last_frame_id=self.frame_count,
            track_len=1,
            confidence=1.0,
            bbox_history=deque(maxlen=30),
            embedding_history=deque(maxlen=50)
        )
        person.bbox_history.append(bbox)
        person.embedding_history.append(embedding)
        
        self.tracked_persons[person_id] = person
        self.reid_memory[person_id].append(embedding)
        
        return person_id
    
    def _remove_old_tracks(self):
        """Remove tracks that haven't been updated recently."""
        person_ids_to_remove = []
        
        for person_id, person in self.tracked_persons.items():
            age = self.frame_count - person.last_frame_id
            
            if age > self.max_age:
                person_ids_to_remove.append(person_id)
        
        for person_id in person_ids_to_remove:
            del self.tracked_persons[person_id]
    
    def get_tracker_state(self) -> Dict:
        """Get current tracker state."""
        return {
            'frame_count': self.frame_count,
            'num_active_tracks': len(self.tracked_persons),
            'total_persons_tracked': self.next_person_id,
            'tracked_persons': {
                pid: {
                    'id': p.person_id,
                    'track_len': p.track_len,
                    'confidence': p.confidence,
                    'last_bbox': p.bbox_history[-1] if p.bbox_history else None
                }
                for pid, p in self.tracked_persons.items()
            }
        }
    
    def save_embeddings(self, save_path: str):
        """Save all tracked embeddings for offline analysis."""
        data = {
            'reid_memory': {
                pid: np.array(embs).astype(np.float32)
                for pid, embs in self.reid_memory.items()
            },
            'total_persons': self.next_person_id,
            'embedding_dim': self.embedding_dim
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Embeddings saved to {save_path}")
    
    def load_embeddings(self, load_path: str):
        """Load previously saved embeddings."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.reid_memory = defaultdict(list)
        for pid, embs in data['reid_memory'].items():
            self.reid_memory[pid] = [emb for emb in embs]
        
        self.next_person_id = data['total_persons']
        print(f"✓ Loaded embeddings for {len(self.reid_memory)} persons")
    
    def get_person_embedding_history(self, person_id: int) -> Optional[np.ndarray]:
        """
        Get average embedding for a person from history.
        Useful for improving re-ID accuracy.
        """
        if person_id not in self.reid_memory or not self.reid_memory[person_id]:
            return None
        
        embeddings = np.array(self.reid_memory[person_id])
        return np.mean(embeddings, axis=0)


class MultiCameraTracker:
    """Handle re-identification across multiple camera views."""
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize multi-camera tracker.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.camera_trackers = {}
        self.global_reid_memory = defaultdict(list)
        self.embedding_dim = embedding_dim
        self.camera_to_global_id = {}  # Map (camera_id, local_id) -> global_id
        self.global_id_counter = 0
    
    def add_camera(self, camera_id: str):
        """Add a new camera stream."""
        if camera_id not in self.camera_trackers:
            self.camera_trackers[camera_id] = FAISSPersonTracker(
                embedding_dim=self.embedding_dim
            )
    
    def process_camera_frame(self, camera_id: str, 
                            detections: List[Tuple[np.ndarray, Tuple]]):
        """Process detections from a camera and get global IDs."""
        self.add_camera(camera_id)
        tracker = self.camera_trackers[camera_id]
        
        # Get local assignments
        bbox_to_local_id = tracker.update(detections)
        
        # Convert to global IDs
        bbox_to_global_id = {}
        for bbox, local_id in bbox_to_local_id.items():
            global_key = (camera_id, local_id)
            
            if global_key not in self.camera_to_global_id:
                self.camera_to_global_id[global_key] = self.global_id_counter
                self.global_id_counter += 1
            
            global_id = self.camera_to_global_id[global_key]
            bbox_to_global_id[bbox] = global_id
        
        return bbox_to_global_id
