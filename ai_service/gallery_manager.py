import os
import cv2
import faiss
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
import uuid
from datetime import datetime

from model_loader import load_pretrained_model as load_v2
from model_loader_v3 import load_pretrained_model as load_v3
from utils import resize_and_normalize

class BaseGalleryManager:
    def __init__(self, index_path: str, meta_path: str, embedding_dim: int, device: str, camera_id: str = None):
        self.device = device
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedding_dim = embedding_dim
        self.camera_id = camera_id  # Track which camera/source added this entry
        self.metadata = {}  # Maps FAISS index ID -> {uuid, person_id, exemplars, centroid, ...}
        self.next_id = 0
        self.next_uuid = 0  # For generating unique identity UUIDs
        self.uuid_to_faiss_id = {}  # Map UUID -> FAISS index ID (for lookups)
        self.index = self._load_or_create_index()
        
        # Try connecting to Redis Vector DB if server is running
        self.redis_manager = None
        try:
            from redis_gallery_manager import RedisGalleryManager
            idx_name = "video_reid_idx" if "video" in self.index_path else "image_reid_idx"
            self.redis_manager = RedisGalleryManager(index_name=idx_name, dim=self.embedding_dim)
            print(f"[+] Active Redis Vector DB connected for index '{idx_name}'!")
        except Exception as e:
            print(f"Notice: Redis Vector DB not active ({e}). Using FAISS index.")
        
    def _load_or_create_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            print(f"Loading FAISS index from {self.index_path}")
            index = faiss.read_index(self.index_path)
            
            # Auto-reset if dimension changed (e.g., upgrading DINOv2 -> DINOv3)
            if index.d != self.embedding_dim:
                print(f"Dimension mismatch (Found {index.d}, Expected {self.embedding_dim}). Creating new FAISS index.")
                return faiss.IndexFlatIP(self.embedding_dim)
                
            with open(self.meta_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get('metadata', {})
                self.next_id = data.get('next_id', 0)
                self.next_uuid = data.get('next_uuid', 0)
                self.uuid_to_faiss_id = data.get('uuid_to_faiss_id', {})
            return index
        else:
            print(f"Creating new FAISS index for {self.index_path}")
            return faiss.IndexFlatIP(self.embedding_dim)
            
    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'next_id': self.next_id,
                'next_uuid': self.next_uuid,
                'uuid_to_faiss_id': self.uuid_to_faiss_id
            }, f)

    def add_identity_embedding(self, feature: np.ndarray, person_id: str, name: str = "", image_path: str = "", camera_id: str = None, match_threshold: float = 0.50, max_exemplars: int = 5):
        """
        Add or update identity with exemplar-based deduplication.
        
        Args:
            feature: Embedding vector (normalized)
            person_id: Human-readable person identifier (e.g., "PERSON_0001")
            name: Optional person name
            image_path: Path to exemplar crop image
            camera_id: Which camera captured this
            match_threshold: Confidence threshold to match existing identity (else create new)
            max_exemplars: Max exemplar crops to store per identity
            
        Returns:
            UUID of the identity (new or matched)
        """
        # Ensure numpy array of float32 with shape (N, D)
        if isinstance(feature, list):
            feature = np.array(feature)
        feature = feature.astype(np.float32)

        if feature.ndim == 1:
            feature = feature.reshape(1, -1)

        # Defensive check: embedding dimension must match the index
        if feature.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dim mismatch: expected {self.embedding_dim}, got {feature.shape}")

        # Normalize vectors for IndexFlatIP (cosine via dot product)
        faiss.normalize_L2(feature)
        
        entry_camera_id = camera_id or self.camera_id or "unknown"
        
        # Step 1: Try to match with existing identities
        matched_identity_uuid = None
        if self.index.ntotal > 0:
            distances, indices = self.index.search(feature, k=1)
            dist = float(distances[0][0])
            idx = int(indices[0][0])
            
            if dist >= match_threshold and idx in self.metadata:
                matched_identity_uuid = self.metadata[idx].get('uuid')
        
        # Step 2: Create new identity or update existing
        if matched_identity_uuid is None:
            # Create new identity
            identity_uuid = str(uuid.uuid4())
            self.next_uuid += 1
            faiss_id = self.next_id
            self.next_id += 1
            
            self.metadata[faiss_id] = {
                'uuid': identity_uuid,
                'person_id': person_id,
                'name': name,
                'model_type': getattr(self, 'model_type', None),
                'dim': self.embedding_dim,
                'camera_id': entry_camera_id,
                'created_at': datetime.now().isoformat(),
                'exemplars': [],  # List of (image_path, embedding, confidence, timestamp)
                'centroid': None,  # Running mean of embeddings
                'num_exemplars': 0
            }
            self.uuid_to_faiss_id[identity_uuid] = faiss_id
            
            # Add first exemplar
            self.metadata[faiss_id]['exemplars'].append({
                'image_path': image_path,
                'embedding': feature[0].copy(),
                'timestamp': datetime.now().isoformat()
            })
            self.metadata[faiss_id]['centroid'] = feature[0].copy()
            self.metadata[faiss_id]['num_exemplars'] = 1
            
            # Add to FAISS index
            self.index.add(feature)
            result_uuid = identity_uuid
        else:
            # Update existing identity
            faiss_id = self.uuid_to_faiss_id[matched_identity_uuid]
            meta = self.metadata[faiss_id]
            
            # Update centroid using incremental mean
            if meta['centroid'] is not None:
                n = meta['num_exemplars']
                meta['centroid'] = (n * meta['centroid'] + feature[0]) / (n + 1)
            else:
                meta['centroid'] = feature[0].copy()
            
            # Check if embedding is sufficiently different from last exemplar
            # (to avoid storing nearly-duplicate crops taken milliseconds apart)
            should_add_exemplar = True
            if meta['exemplars']:
                last_embedding = meta['exemplars'][-1]['embedding']
                drift = np.linalg.norm(feature[0] - last_embedding)
                if drift < 0.12:  # EXEMPLAR_DRIFT_THRESHOLD from config
                    should_add_exemplar = False
            
            # Add exemplar if space available or overwrite oldest
            if should_add_exemplar:
                meta['exemplars'].append({
                    'image_path': image_path,
                    'embedding': feature[0].copy(),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only max_exemplars
                if len(meta['exemplars']) > max_exemplars:
                    meta['exemplars'] = meta['exemplars'][-max_exemplars:]
                
                meta['num_exemplars'] = len(meta['exemplars'])
            
            # Update FAISS index with new centroid
            self.index.reset()  # Clear and rebuild with updated centroids
            for idx, entry in self.metadata.items():
                centroid = entry['centroid']
                if centroid is not None:
                    centroid_norm = centroid.reshape(1, -1).copy().astype(np.float32)
                    faiss.normalize_L2(centroid_norm)
                    self.index.add(centroid_norm)
            
            result_uuid = matched_identity_uuid

        # Sync vector embedding and metadata to Redis Vector DB
        if getattr(self, 'redis_manager', None):
            try:
                feat_vec = feature[0] if feature.ndim > 1 else feature
                self.redis_manager.add_identity(person_id, feat_vec, {
                    'uuid': result_uuid,
                    'name': name,
                    'camera_id': camera_id or self.camera_id,
                    'exemplars': [{'image_path': image_path, 'timestamp': datetime.now().isoformat()}]
                })
            except Exception as e:
                print(f"Redis Vector DB sync notice: {e}")

        self.save_index()
        return result_uuid

    def get_identity_exemplars(self, identity_uuid: str) -> Optional[Dict]:
        """
        Retrieve all exemplars for a given identity UUID.
        
        Returns:
            Dict with exemplar list and metadata, or None if not found
        """
        if identity_uuid not in self.uuid_to_faiss_id:
            return None
        
        faiss_id = self.uuid_to_faiss_id[identity_uuid]
        meta = self.metadata[faiss_id].copy()
        
        clean_exemplars = []
        for ex in meta['exemplars']:
            clean_exemplars.append({
                'image_path': ex['image_path'],
                'timestamp': ex['timestamp']
            })
            
        # Return public-facing info
        return {
            'uuid': identity_uuid,
            'person_id': meta['person_id'],
            'camera_id': meta['camera_id'],
            'created_at': meta['created_at'],
            'exemplars': clean_exemplars,
            'num_exemplars': meta['num_exemplars']
        }

    def get_all_identities(self) -> List[Dict]:
        """
        Get summary of all identities with exemplar counts.
        """
        identities = []
        for faiss_id, meta in self.metadata.items():
            identities.append({
                'uuid': meta['uuid'],
                'person_id': meta['person_id'],
                'camera_id': meta['camera_id'],
                'created_at': meta['created_at'],
                'num_exemplars': meta['num_exemplars']
            })
        return identities

    def add_identity(self, image_path: str, person_id: str, name: str = ""):
        from PIL import Image
        try:
            pil_img = Image.open(image_path).convert('RGB')
            img = np.array(pil_img)
            # PIL is already RGB, so no need for cvtColor(BGR2RGB)
        except Exception as e:
            raise ValueError(f"Could not read image {image_path}: {e}")
        
        feature = self.extract_feature(img)
        return self.add_identity_embedding(feature[0], person_id, name, image_path, match_threshold=0.50, max_exemplars=5)

class VideoGalleryManager(BaseGalleryManager):
    """Uses Trained DINOv3 or ONNX DINOv2 for robust tracking with exemplar-based identity storage."""
    def __init__(self, index_path: str = "video_gallery.faiss", meta_path: str = "video_meta.pkl", device: str = None, camera_id: str = None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(index_path, meta_path, 384 if device == 'cpu' else 512, device, camera_id=camera_id)
        # provenance
        self.match_threshold = 0.65  # Confidence for matching existing identities
        self.max_exemplars = 5       # Max exemplar crops per identity
        
        if device == 'cpu':
            print(f"Loading ONNX Runtime DINOv2 for Video Tracking on {device}...")
            from model_loader_onnx import load_onnx_reid_model
            self.model = load_onnx_reid_model()
            self.model_type = "dinov2_onnx"
            self.embedding_dim = 384
            # Update index dimension if needed
            self.index = faiss.IndexFlatIP(384)
        else:
            print(f"Loading Trained DINOv3 for Video Tracking on {device}...")
            self.model_type = "dinov3"
            pth_path = os.path.join(os.path.dirname(__file__), "net_last.pth")
            if not os.path.exists(pth_path):
                pth_path = os.path.join(os.path.dirname(__file__), "..", "oracle_best.pth")
            self.model = load_v3(model_path=pth_path, device=device)
            self.model.eval()

    def extract_feature(self, image: np.ndarray) -> np.ndarray:
        target_sz = (252, 126) if getattr(self, 'model_type', '') == 'dinov2_onnx' else (256, 128)
        tensor = resize_and_normalize(image, target_size=target_sz)
        if hasattr(self.model, 'session'):
            feature = self.model.extract_features(tensor)
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
        else:
            tensor = tensor.to(self.device)
            with torch.no_grad():
                feature = self.model.extract_features(tensor)
                feature = feature.cpu().numpy().astype(np.float32)
        if not feature.flags.c_contiguous:
            feature = np.ascontiguousarray(feature)
        faiss.normalize_L2(feature)
        return feature

class ImageGalleryManager(BaseGalleryManager):
    """Uses ONNX / PyTorch DINO for Top-K static image search."""
    def __init__(self, model_weights: str, index_path: str = "image_gallery.faiss", meta_path: str = "image_meta.pkl", device: str = None, camera_id: str = None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(index_path, meta_path, 384 if device == 'cpu' else 512, device, camera_id=camera_id)
        # provenance
        self.match_threshold = 0.55  # Stricter threshold for image search
        self.max_exemplars = 3       # Fewer exemplars for static images
        
        if device == 'cpu':
            print(f"Loading ONNX Runtime DINOv2 for Image Search on {device}...")
            from model_loader_onnx import load_onnx_reid_model
            self.model = load_onnx_reid_model()
            self.model_type = "dinov2_onnx"
            self.embedding_dim = 384
            self.index = faiss.IndexFlatIP(384)
        else:
            print(f"Loading DINOv3 for Image Search on {device}...")
            self.model_type = "dinov3"
            self.model = load_v3(model_weights, device=device)
            self.model.eval()

    def extract_feature(self, image: np.ndarray) -> np.ndarray:
        target_sz = (252, 126) if getattr(self, 'model_type', '') == 'dinov2_onnx' else (256, 128)
        tensor = resize_and_normalize(image, target_size=target_sz)
        if hasattr(self.model, 'session'):
            feature = self.model.extract_features(tensor)
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
        else:
            tensor = tensor.to(self.device)
            with torch.no_grad():
                feature = self.model.extract_features(tensor)
                feature = feature.cpu().numpy().astype(np.float32)
        if not feature.flags.c_contiguous:
            feature = np.ascontiguousarray(feature)
        faiss.normalize_L2(feature)
        return feature
        
    def add_identity(self, image_path: str, person_id: str, name: str = ""):
        from PIL import Image
        import uuid
        from datetime import datetime
        try:
            pil_img = Image.open(image_path).convert('RGB')
            img = np.array(pil_img)
        except Exception as e:
            raise ValueError(f"Could not read image {image_path}: {e}")
        
        feature = self.extract_feature(img)
        
        # MARKET-1501 STYLE (FLAT GALLERY)
        # Every uploaded image is an independent entry in the FAISS index.
        # No centroid grouping or drift deduplication for static image ranking.
        
        identity_uuid = str(uuid.uuid4())
        faiss_id = self.next_id
        self.next_id += 1
        
        self.metadata[faiss_id] = {
            'uuid': identity_uuid,
            'person_id': person_id,
            'name': name,
            'model_type': self.model_type,
            'dim': self.embedding_dim,
            'camera_id': self.camera_id,
            'created_at': datetime.now().isoformat(),
            'image_path': image_path  # Saved directly to root for UI
        }
        self.uuid_to_faiss_id[identity_uuid] = faiss_id
        
        # Add to FAISS index
        feat_to_add = feature[0].copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(feat_to_add)
        self.index.add(feat_to_add)
        
        self.save_index()
        return identity_uuid
        
    def search(self, image: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        feature = self.extract_feature(image)
        feature = feature.astype(np.float32)
        if not feature.flags.c_contiguous:
            feature = np.ascontiguousarray(feature)
        distances, indices = self.index.search(feature, min(top_k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.metadata:
                meta = self.metadata[idx].copy()
                meta['similarity'] = float(dist)
                # Strip unserializable numpy arrays (legacy data)
                meta.pop('exemplars', None)
                meta.pop('centroid', None)
                
                # Ensure image_path is at the root for frontend (legacy data fallback)
                if 'image_path' not in meta and 'exemplars' in self.metadata[idx] and len(self.metadata[idx]['exemplars']) > 0:
                    meta['image_path'] = self.metadata[idx]['exemplars'][0].get('image_path')
                    
                results.append(meta)
        return results
