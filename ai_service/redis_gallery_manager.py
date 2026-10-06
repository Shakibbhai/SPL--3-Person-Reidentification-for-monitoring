"""
Redis Vector Database Manager for Person Re-ID.
Uses RedisVL / RediSearch Vector Similarity Search (VSS) with HNSW index.
Stores 384-D / 512-D float32 vector embeddings alongside JSON metadata directly inside Redis.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    import redis
    from redis.commands.search.field import VectorField, TextField, TagField, NumericField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_AVAILABLE = True
except Exception as e:
    print(f"Redis import notice: {e}")
    REDIS_AVAILABLE = False


class RedisGalleryManager:
    """Redis-backed Vector Search & Identity Metadata Manager."""
    def __init__(self, index_name: str = "video_reid_idx", dim: int = 384, host: str = 'localhost', port: int = 6379, password: str = None):
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis package is not installed. Run: pip install redis redisvl")
            
        self.index_name = index_name
        self.dim = dim
        self.prefix = f"{index_name}:person:"
        
        self.client = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=False,
            socket_timeout=3.0
        )
        
        # Test connection
        self.client.ping()
        self._init_vector_index()

    def _init_vector_index(self):
        """Create RediSearch Vector Similarity Index (HNSW Cosine)."""
        try:
            self.client.ft(self.index_name).info()
            print(f"[+] Connected to existing Redis Vector Index: '{self.index_name}'")
        except Exception:
            print(f"[+] Creating new Redis RediSearch HNSW Index: '{self.index_name}' (dim={self.dim})...")
            schema = (
                TagField("person_id"),
                TagField("uuid"),
                TagField("camera_id"),
                NumericField("created_at"),
                VectorField(
                    "vector",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.dim,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                        "M": 16,
                        "EF_CONSTRUCTION": 200
                    }
                )
            )
            definition = IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH)
            self.client.ft(self.index_name).create_index(fields=schema, definition=definition)
            print(f"[+] Redis Vector Index '{self.index_name}' created successfully!")

    def add_identity(self, person_id: str, vector: np.ndarray, metadata: Dict) -> str:
        """Add or update an identity embedding in Redis."""
        vector = vector.astype(np.float32)
        # Normalize
        norm = np.linalg.norm(vector) + 1e-8
        vector = vector / norm
        
        key = f"{self.prefix}{person_id}"
        
        doc = {
            "person_id": person_id,
            "uuid": metadata.get("uuid", ""),
            "name": metadata.get("name", person_id),
            "camera_id": metadata.get("camera_id", "camera_1"),
            "created_at": time.time(),
            "vector": vector.tobytes(),
            "exemplars_json": json.dumps(metadata.get("exemplars", []))
        }
        
        self.client.hset(key, mapping=doc)
        return key

    def search_similar(self, query_vector: np.ndarray, top_k: int = 5, distance_threshold: float = 0.35) -> List[Tuple[Dict, float]]:
        """
        Search nearest neighbor vectors in Redis.
        Returns list of (metadata_dict, similarity_score).
        """
        query_vector = query_vector.astype(np.float32)
        norm = np.linalg.norm(query_vector) + 1e-8
        query_vector = query_vector / norm
        
        q = Query(f"*=>[KNN {top_k} @vector $vec AS score]") \
            .sort_by("score") \
            .return_fields("person_id", "uuid", "name", "camera_id", "exemplars_json", "score") \
            .paging(0, top_k) \
            .dialect(2)
            
        params = {"vec": query_vector.tobytes()}
        res = self.client.ft(self.index_name).search(q, query_params=params)
        
        results = []
        for doc in res.documents:
            # RediSearch COSINE distance range is [0, 2], Similarity = 1 - distance
            cosine_dist = float(doc.score)
            sim_score = float(1.0 - cosine_dist)
            
            meta = {
                "person_id": getattr(doc, "person_id", ""),
                "uuid": getattr(doc, "uuid", ""),
                "name": getattr(doc, "name", ""),
                "camera_id": getattr(doc, "camera_id", ""),
                "exemplars": json.loads(getattr(doc, "exemplars_json", "[]"))
            }
            results.append((meta, sim_score))
            
        return results
