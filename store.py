import faiss
import numpy as np
import uuid
import json
import os
import torch
import subprocess
from pathlib import Path
from typing import Dict, Tuple, List

# --- Custom Modules (From previous steps) ---
from load_db import load_db
from load_videomae_model import load_model  # Assumes this returns your VideoMAEv2 model
from isc_extractor import ISCFeatureExtractor, save_isc_features, get_isc_features
from shot_detection import detect_shots_pyscenedetect
from frame_extraction import extract_frames_per_shot
from shot_embeddings import process_shots_to_embeddings

# --- Helper: Get Duration for Metadata ---
def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        return float(subprocess.check_output(cmd).decode().strip())
    except Exception:
        return 0.0

# --- Helper: Save DB ---
def save_faiss_db(index: faiss.IndexIDMap, metadata: Dict[int, dict], index_path: str, metadata_path: str):
    print(f"Saving index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, "w") as f:
        # JSON keys must be strings
        json.dump({str(k): v for k, v in metadata.items()}, f, indent=2)

# --- Core Logic ---
def process_and_store_video(
    # DB State
    index: faiss.IndexIDMap,
    metadata: Dict[int, dict],
    next_id: int,
    # Models
    videomae_model,
    isc_extractor: ISCFeatureExtractor,
    # Config
    video_path: str,
    isc_output_dir: str,
    device: str = "cuda"
) -> int:
    """
    Runs the full pipeline for a single video.
    Returns the new next_id.
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"Error: {video_path} not found.")
        return next_id

    # Generate Identifiers
    video_uuid = str(uuid.uuid4())
    video_title = video_path_obj.stem # "my_video" from "path/my_video.mp4"
    
    print(f"\n=== Processing: {video_title} ===")
    
    # ---------------------------------------------------------
    # 1. ISC Features (Save to file)
    # ---------------------------------------------------------
    isc_save_path = os.path.join(isc_output_dir, f"{video_title}.npy")
    if os.path.exists(isc_save_path):
        print(f"[ISC] Skipping, already exists: {isc_save_path}")
    else:
        print("[ISC] Extracting features...")
        isc_feats = get_isc_features(
            str(video_path), 
            extractor=isc_extractor, 
            fps=1.0, 
            device=device
        )
        save_isc_features(isc_feats, isc_save_path)

    # ---------------------------------------------------------
    # 2. Shot Detection
    # ---------------------------------------------------------
    print("[Shots] Detecting boundaries...")
    # Returns [(timestamp, frame, fps), ...] for every CUT
    cuts = detect_shots_pyscenedetect(str(video_path), threshold=3.0)
    
    # Calculate Intervals for Metadata (Start, End)
    duration = get_video_duration(str(video_path))
    
    # Logic: 
    #   Shot 0: 0.0 -> Cut 0
    #   Shot 1: Cut 0 -> Cut 1
    #   ...
    #   Last Shot: Cut N -> Duration
    
    cut_times = sorted([c[0] for c in cuts])
    intervals = []
    current_time = 0.0
    
    for c_time in cut_times:
        if c_time > current_time:
            intervals.append((current_time, c_time))
        current_time = c_time
    
    # Add final segment
    if current_time < duration:
        intervals.append((current_time, duration))
        
    print(f"[Shots] Found {len(intervals)} shots.")

    # ---------------------------------------------------------
    # 3. Frame Extraction (16 frames per shot)
    # ---------------------------------------------------------
    # Pass the raw cuts to the extractor (it handles the seek logic)
    print("[Frames] Extracting 16 frames per shot...")
    shot_tensors = extract_frames_per_shot(str(video_path), cuts, num_frames_per_shot=16)

    # Sanity Check
    if len(shot_tensors) != len(intervals):
        print(f"Warning: Extracted {len(shot_tensors)} tensors but calculated {len(intervals)} intervals.")
        # We trim intervals to match tensors if extraction failed for some tiny shots
        intervals = intervals[:len(shot_tensors)]

    if not shot_tensors:
        print("No valid shots extracted. Skipping video.")
        return next_id

    # ---------------------------------------------------------
    # 4. Generate Embeddings (VideoMAEv2)
    # ---------------------------------------------------------
    print("[Embeddings] Running VideoMAEv2 Inference...")
    embeddings = process_shots_to_embeddings(
        shot_tensors, 
        model=videomae_model, 
        device=device,
        batch_size=8 # Adjust based on VRAM
    )

    # ---------------------------------------------------------
    # 5. Store in FAISS
    # ---------------------------------------------------------
    if embeddings.shape[0] == 0:
        return next_id
        
    print(f"[DB] Adding {embeddings.shape[0]} vectors to FAISS...")
    
    # Ensure float32
    embeddings = embeddings.astype(np.float32)
    
    # Generate IDs for this batch
    num_vectors = embeddings.shape[0]
    ids = np.arange(next_id, next_id + num_vectors, dtype=np.int64)
    
    # Add to FAISS
    index.add_with_ids(embeddings, ids)
    
    # Update Metadata
    for i, db_id in enumerate(ids):
        start_t, end_t = intervals[i]
        
        metadata[int(db_id)] = {
            "video_id": video_uuid,
            "video_title": video_title,
            "shot_start_sec": float(f"{start_t:.2f}"), # Round for cleanliness
            "shot_end_sec": float(f"{end_t:.2f}"),
            "source_path": str(video_path)
        }

    return next_id + num_vectors


def main():
    # --- CONFIGURATION ---
    # List of videos or a directory walk
    video_paths = [
        '/home/sam/reverse-video-search/videos/bigbuckbunny/bigbuckbunny.mp4'
    ]
    
    index_path = "db/shots.faiss"
    metadata_path = "db/shots.json"
    isc_folder = "db/isc_features"
    
    vector_dim = 768 # VideoMAEv2 Base
    # ---------------------

    # 1. Setup Directories
    os.makedirs("db", exist_ok=True)
    os.makedirs(isc_folder, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"System Device: {device}")

    # 2. Load DB
    index, meta, next_id = load_db(index_path, metadata_path, vector_dim)
    print(f"DB Loaded. Vectors: {index.ntotal}")

    # 3. Load Models (Load once, reuse)
    print("Loading VideoMAEv2...")
    videomae_model = load_model() # From your existing load_model.py
    
    print("Loading ISC Extractor...")
    isc_extractor = ISCFeatureExtractor(device=device)

    # 4. Process Loop
    for v_path in video_paths:
        try:
            next_id = process_and_store_video(
                index, meta, next_id,
                videomae_model,
                isc_extractor,
                v_path,
                isc_folder,
                device
            )
        except Exception as e:
            print(f"CRITICAL FAIL on {v_path}: {e}")
            import traceback
            traceback.print_exc()

    # 5. Save Final DB
    save_faiss_db(index, meta, index_path, metadata_path)
    print("All done.")

if __name__ == '__main__':
    main()