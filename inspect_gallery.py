"""
Inspector Tool for FAISS Vector Indexes and Metadata PKL Files.
Run: python inspect_gallery.py
"""

import os
import pickle
import faiss
import numpy as np

def inspect_faiss_and_pkl(index_path: str, meta_path: str, name: str):
    print("=" * 65)
    print(f" [INSPECTING] {name.upper()}")
    print("=" * 65)
    
    # 1. Inspect FAISS Index
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
            print(f"[+] FAISS Index File : {index_path}")
            print(f"    - Total Vector Embeddings Stored : {index.ntotal}")
            print(f"    - Vector Dimension                : {index.d}D")
            
            if index.ntotal > 0:
                try:
                    vec = index.reconstruct(0)
                    print(f"    - Sample Vector (first 8 values)  : {np.round(vec[:8], 4)} ...")
                except Exception:
                    pass
        except Exception as e:
            print(f"[!] Could not read FAISS index {index_path}: {e}")
    else:
        print(f"[-] FAISS Index File not found at: {index_path}")

    print("-" * 65)

    # 2. Inspect Pickle Metadata
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
                metadata = data.get('metadata', {})
                next_id = data.get('next_id', 0)
                next_uuid = data.get('next_uuid', 0)
                
            print(f"[+] Pickle Metadata File: {meta_path}")
            print(f"    - Total Stored Identities : {len(metadata)}")
            print(f"    - Next Identity Counter   : {next_id}")
            print("-" * 65)
            print(" [STORED IDENTITIES & METADATA DETAILS]")
            
            if metadata:
                for faiss_id, meta in metadata.items():
                    print(f"\n    * [FAISS Index #{faiss_id}]")
                    print(f"      - Person ID  : {meta.get('person_id', 'N/A')}")
                    print(f"      - Name       : {meta.get('name', 'None')}")
                    print(f"      - UUID       : {meta.get('uuid', 'N/A')}")
                    print(f"      - Camera ID  : {meta.get('camera_id', 'N/A')}")
                    print(f"      - Created At : {meta.get('created_at', 'N/A')}")
                    
                    exemplars = meta.get('exemplars', [])
                    print(f"      - Exemplar Crops ({len(exemplars)} saved):")
                    for idx, ex in enumerate(exemplars[:3], 1):
                        print(f"        {idx}. Path: {ex.get('image_path')} | Time: {ex.get('timestamp')}")
                    if len(exemplars) > 3:
                        print(f"        ... and {len(exemplars) - 3} more crops")
            else:
                print("    (No identities registered yet. Process a video to add entries!)")
        except Exception as e:
            print(f"[!] Could not read Metadata file {meta_path}: {e}")
    else:
        print(f"[-] Pickle Metadata File not found at: {meta_path}")
        
    print("=" * 65 + "\n")


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "ai_service")
    
    # Inspect Video Gallery
    inspect_faiss_and_pkl(
        os.path.join(base_dir, "video_gallery.faiss"),
        os.path.join(base_dir, "video_meta.pkl"),
        "Video Gallery (Dynamic Surveillance Tracking)"
    )
    
    # Inspect Image Gallery
    inspect_faiss_and_pkl(
        os.path.join(base_dir, "image_gallery.faiss"),
        os.path.join(base_dir, "image_meta.pkl"),
        "Image Gallery (Static Top-K Search)"
    )
