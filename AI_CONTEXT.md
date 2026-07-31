# Percepta Re-ID System: Project Documentation Context

## 1. Project Aim & Overview
The **Percepta Re-ID** project is an advanced, real-time Person Re-Identification (Re-ID) system designed to track and re-identify individuals across multiple camera feeds. 
The primary objective of the project is to overcome traditional surveillance limitations by maintaining robust tracking performance even under **adversarial environmental conditions** such as severe occlusion, low lighting, and motion blur. 

Instead of relying merely on spatial bounding box overlap (which fails when subjects leave the camera frame), the system extracts deep semantic feature vectors (embeddings) to act as a "mathematical fingerprint" for each person. This allows the system to seamlessly recognize a person when they reappear in a completely different video feed.

## 2. Core Architecture & Technologies
*   **Spatial Detection & Tracking:** RT-DETR (Real-Time DEtection TRansformer). Used to detect human subjects in video frames and extract highly accurate bounding box crops.
*   **Semantic Feature Extraction:** DINOv3 Foundation Model. Processes the crops and generates highly discriminative 512-Dimensional feature embeddings.
*   **Global Memory & Retrieval:** FAISS (Facebook AI Similarity Search). Acts as the vector database to store embeddings and perform instantaneous similarity matching.
*   **Backend:** FastAPI (Python). Provides high-performance REST APIs to bridge the AI engine and the frontend.
*   **Frontend:** React.js. A premium, dark-mode aesthetic dashboard displaying live video tracking, bounding boxes, and an "Identity Gallery" where operators can assign custom tags (e.g., "James Bond") to detected profiles.

## 3. Key Performance Metrics (Market-1501 Evaluation)
Based on rigorous evaluation (documented in the project presentations):
*   **Clean Baseline:** 89.88% Rank-1 Accuracy, 76.66% mAP
*   **Adversarial Robustness:** Even under severe occlusion, the model retains a 78.80% Rank-1 and 90.20% Rank-5 accuracy, mathematically proving its resilience for real-world deployment.

---

## 4. Technical Design Choices & Advanced Specifications

To provide absolute precision for documentation generation, here are the deeper technical mechanics governing the project:

### A. Distance Metrics & Memory Retention
*   **Vector Distance:** The FAISS index maps the 512-D embeddings using **L2 (Euclidean) Distance** or **Cosine Similarity** to match subjects. 
*   **Spatial Exit Handling:** The FAISS database acts as a persistent memory bank. When a subject physically leaves the camera's field of view, their tracking ID is not lost; their 512-D embedding signature remains retained in the FAISS gallery. When they re-enter, the system recovers their original ID rather than assigning a duplicate.

### B. Training & Fine-Tuning Phase
*   **Loss Functions:** During the 60-epoch training phase on the Market-1501 dataset, the model was optimized utilizing **Circle Loss**. This advanced metric learning loss function was specifically chosen because it offers a unified perspective on learning with class-level labels and pair-wise labels, effectively maximizing the within-class similarity and minimizing between-class similarity more robustly than standard triplet loss.
*   **Classification Head Removal:** Post-training, the classification head was discarded. Only the semantic feature extraction backbone is deployed during inference, utilizing the extracted vectors for similarity search rather than raw classification.

### C. Hardware Constraints & Efficiency
*   **Real-Time Priority:** **RT-DETR** was specifically selected as the spatial tracker over traditional CNN-based trackers. As a transformer-based detector, RT-DETR guarantees high FPS (Frames Per Second) processing while maintaining superior accuracy. Given GPU constraints, this highly efficient, end-to-end spatial tracker ensures the computational heavy lifting is strictly reserved for the DINOv3 semantic extraction, allowing the pipeline to maintain real-time performance without bottlenecking.

## 5. Detailed File & Component Breakdown

### A. The AI Engine (`ai_service/`)
This is the "brain" of the project where computer vision and deep learning happen.
*   **`ai_service/reid_pipeline.py`**
    *   **Purpose:** The central orchestrator for video inference.
    *   **How it works:** It reads video frames, passes them to RT-DETR for spatial detection, crops the detected humans, passes those crops to DINOv3 for semantic embedding, and then queries `gallery_manager.py` to either assign a new tracking ID or recognize an existing one.
*   **`ai_service/model_loader_v3.py`**
    *   **Purpose:** The model weight handler.
    *   **How it works:** Safely loads the heavily fine-tuned PyTorch DINOv3 model checkpoint (e.g., `net_last.pth`) into GPU/CPU memory and prepares the model for inference.
*   **`ai_service/gallery_manager.py`**
    *   **Purpose:** The memory system and vector database interface.
    *   **How it works:** Manages the FAISS index. It stores the 512-D embeddings of every tracked person. It supports querying the database for matches and allows operators to assign custom alias strings (like specific names) to auto-generated IDs.

### B. The Backend Server (`backend/`)
*   **`backend/main.py`**
    *   **Purpose:** The FastAPI web server.
    *   **How it works:** Exposes endpoints (like `/upload`, `/stream`, `/identities`, `/update_identity`) so the React frontend can interact with the Python AI engine. It manages WebSocket connections for streaming processed video frames with bounding boxes back to the browser.
*   **`backend/import_market1501.py`**
    *   **Purpose:** Data ingestion script.
    *   **How it works:** Used to import the standard Market-1501 academic dataset images directly into the FAISS gallery for baseline testing.

### C. The Frontend Dashboard (`frontend/src/`)
*   **`frontend/src/App.jsx`**
    *   **Purpose:** The main React application file.
    *   **How it works:** Contains the layout and state management for the entire UI. It fetches video streams from the backend and renders the "Video Tracker" cards. It includes premium, glassmorphism UI elements to display the FAISS database subjects (Identity Gallery) and allows users to type in custom names to update the backend database in real-time.
*   **`frontend/src/App.css` & `index.css`**
    *   **Purpose:** Styling.
    *   **How it works:** Enforces a sleek, professional, high-contrast dark mode aesthetic tailored for enterprise security applications.

### D. Analytics & Evaluation (`archive/scripts/`)
*   **`archive/scripts/evaluate_models.py`**
    *   **Purpose:** Performance testing script.
    *   **How it works:** Runs the DINOv3 model against the Market-1501 test dataset. It applies synthetic noise, motion blur, and simulated occlusion to calculate the exact degradation in Rank-1, Rank-5, Rank-10, and mAP, generating the `evaluation_results.json` file.

---
**How to Use This Context:**
Whenever you start a new AI chat (in ChatGPT, Gemini, Claude, etc.) to ask for help writing the project report, generating diagrams, or expanding documentation, simply copy and paste the contents of this entire document into your first prompt. The AI will instantly understand the exact architecture, the aim, the specific files, and the tech stack of the Percepta Re-ID system.
