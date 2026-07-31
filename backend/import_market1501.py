import os
import requests
import glob
from tqdm import tqdm
import time

def import_market1501(data_dir: str, api_url: str = "http://127.0.0.1:8000/api/identities", max_images: int = 200, max_per_id: int = 5):
    """
    Imports Market-1501 dataset images to the Re-ID backend via the REST API.
    
    Args:
        data_dir: Path to the Market-1501 bounding_box_test directory
        api_url: URL to the backend's add_identity endpoint
        max_images: Maximum total images to upload
        max_per_id: Maximum images to upload per person ID (to avoid overwhelming the gallery)
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return

    # Find all jpg images
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    if not image_paths:
        print(f"No .jpg images found in {data_dir}")
        return

    print(f"Found {len(image_paths)} images. Preparing to import (limited to {max_images} total, {max_per_id} per ID)...")
    
    id_counts = {}
    uploaded = 0
    errors = 0
    
    # Sort paths for consistent importing
    image_paths.sort()
    
    for path in tqdm(image_paths, desc="Uploading Images"):
        if uploaded >= max_images:
            break
            
        filename = os.path.basename(path)
        # Market-1501 format: 0001_c1s1_001051_00.jpg -> ID is '0001'
        person_id = filename.split("_")[0]
        
        # Skip junk/distractors if desired
        if person_id in ["-1", "0000"]:
            continue
            
        # Enforce max images per ID
        if id_counts.get(person_id, 0) >= max_per_id:
            continue
            
        name = f"Market_{person_id}"
        
        try:
            with open(path, "rb") as f:
                files = {"file": (filename, f, "image/jpeg")}
                data = {"person_id": person_id, "name": name}
                
                response = requests.post(api_url, data=data, files=files)
                
                if response.status_code == 200:
                    uploaded += 1
                    id_counts[person_id] = id_counts.get(person_id, 0) + 1
                else:
                    print(f"\nFailed to upload {filename}: {response.text}")
                    errors += 1
        except Exception as e:
            print(f"\nError uploading {filename}: {e}")
            errors += 1
            
        # Small delay to not overwhelm the API
        time.sleep(0.05)
            
    print(f"\nImport Complete!")
    print(f"Successfully uploaded: {uploaded} images")
    print(f"Errors: {errors}")
    print(f"Unique identities added: {len(id_counts)}")

if __name__ == "__main__":
    MARKET_TEST_DIR = r"D:\Major Project\percepta_reid_01\data\market1501\market1501\bounding_box_test"
    # Note: Running on CPU without limits could take a very long time.
    # Adjust max_images and max_per_id as needed for your testing.
    import_market1501(MARKET_TEST_DIR, max_images=250, max_per_id=5)
