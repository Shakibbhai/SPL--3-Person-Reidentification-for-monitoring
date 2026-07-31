import os
import sys
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.manifold import TSNE

# Add ai_service to path so we can use the GalleryManager to extract features
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(project_root, "ai_service"))

from gallery_manager import ImageGalleryManager

def generate_tsne(data_dir, num_identities=10, images_per_identity=5):
    print("Loading Trained DINOv3 model...")
    pth_path = os.path.join(project_root, "ai_service", "net_last.pth")
    # if not os.path.exists(pth_path)   pth_path = os.path.join(project_root, "oracle_best.pth")
        
    # Instantiate the gallery manager just to use its feature extraction logic
    gallery = ImageGalleryManager(model_weights=pth_path, index_path="temp_tsne.faiss", meta_path="temp_tsne.pkl")

    print(f"Scanning Market-1501 dataset in {data_dir}...")
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    
    # Group by person ID
    identities = {}
    for path in image_paths:
        filename = os.path.basename(path)
        person_id = filename.split("_")[0]
        if person_id in ["-1", "0000"]:
            continue
        if person_id not in identities:
            identities[person_id] = []
        identities[person_id].append(path)
        
    # Filter to identities with at least `images_per_identity` images
    valid_ids = [pid for pid, paths in identities.items() if len(paths) >= images_per_identity]
    
    if len(valid_ids) < num_identities:
        raise ValueError(f"Not enough identities with {images_per_identity} images. Found {len(valid_ids)}.")
        
    # Select a stable random subset for reproducibility
    random.seed(42)
    selected_ids = random.sample(valid_ids, num_identities)
    
    embeddings = []
    labels = []
    
    print("Extracting 512-D features using Circle Loss optimized weights...")
    for pid in selected_ids:
        selected_paths = random.sample(identities[pid], images_per_identity)
        for path in selected_paths:
            pil_img = Image.open(path).convert('RGB')
            img = np.array(pil_img)
            feature = gallery.extract_feature(img)
            embeddings.append(feature[0])
            labels.append(pid)
            
    embeddings = np.array(embeddings)
    
    print("Computing t-SNE dimensionality reduction...")
    # Perplexity set lower (e.g. 10) because we only have 50 data points
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=1500)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    print("Plotting results for Paper 2...")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams["font.family"] = "serif"
    
    plt.figure(figsize=(9, 7), dpi=300)
    
    unique_labels = list(set(labels))
    colors = sns.color_palette("tab10", n_colors=len(unique_labels))
    
    for i, pid in enumerate(unique_labels):
        idx = [j for j, label in enumerate(labels) if label == pid]
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], 
                    c=[colors[i]], label=f"ID: {pid}", s=120, edgecolors='black', linewidth=1.5, alpha=0.9)
        
    plt.xlabel("t-SNE Dimension 1", fontsize=14, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=14, fontweight='bold')
    
    # Place legend outside the plot so it doesn't cover data points
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=12, frameon=True, shadow=True)
    sns.despine(left=False, bottom=False)
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "tsne_visualization.png")
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Done! Graph saved directly to {output_path}")

if __name__ == "__main__":
    MARKET_TEST_DIR = r"D:\Major Project\percepta_reid_01\data\market1501\market1501\bounding_box_test"
    generate_tsne(MARKET_TEST_DIR)
