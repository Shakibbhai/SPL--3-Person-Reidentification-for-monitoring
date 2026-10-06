"""
Redis Vector DB Connection & Data Inspector Tool.
Run: python inspect_redis.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "ai_service"))

def test_and_inspect_redis():
    print("=" * 65)
    print(" [INSPECTING REDIS VECTOR DB CONNECTION]")
    print("=" * 65)
    
    try:
        import redis
        from redis_gallery_manager import RedisGalleryManager
        
        r = redis.Redis(
            host='localhost',
            port=6379,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            retry_on_timeout=False
        )
        r.ping()
        print("[+] SUCCESS: Redis Server is RUNNING on localhost:6379!")
        
        manager = RedisGalleryManager(index_name="video_reid_idx", dim=384)
        info = r.ft("video_reid_idx").info()
        num_docs = info.get("num_docs", 0)
        print(f"[+] RediSearch Index Name: 'video_reid_idx'")
        print(f"    - Total Stored Vectors/Keys : {num_docs}")
        print("=" * 65)
        
    except Exception as e:
        print("[-] REDIS SERVER IS NOT RUNNING CURRENTLY ON LOCALHOST:6379")
        print(f"    Notice: {e}")
        print("-" * 65)
        print("    [HOW TO START REDIS VECTOR SERVER ON YOUR PC]")
        print("    Option 1 (Docker):")
        print("      docker run -d -p 6379:6379 --name redis-vector redis/redis-stack-server")
        print()
        print("    Option 2 (Memurai / Windows Binary):")
        print("      Download Memurai Developer (Redis for Windows) and run service on port 6379")
        print("=" * 65)

if __name__ == "__main__":
    test_and_inspect_redis()
