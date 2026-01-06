import faiss
import numpy as np
import torch
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# --- Custom Modules ---
from load_db import load_db
from load_videomae_model import load_model
from isc_extractor import ISCFeatureExtractor, get_isc_features
from shot_detection import detect_shots_pyscenedetect
from frame_extraction import extract_frames_per_shot
from shot_embeddings import process_shots_to_embeddings
from transvcl_wrapper import TransVCLWrapper, get_transvcl_alignments

# ==========================================
# 1. Query Processing Logic
# ==========================================

def process_query_video(
    video_path: str,
    videomae_model,
    isc_extractor: ISCFeatureExtractor,
    device: str = "cuda"
):
    """
    Runs the pipeline on the query video (without saving to DB).
    Returns:
        embeddings: (N_shots, 768) - VideoMAE embeddings for retrieval
        isc_feats: (Total_Frames, 256) - ISC features for TransVCL
        shot_intervals: List of (start, end) tuples
    """
    print(f"\n=== Processing Query: {video_path} ===")
    
    # A. Extract ISC Features (for fine-grained TransVCL later)
    print("[ISC] Extracting features...")
    isc_feats = get_isc_features(video_path, extractor=isc_extractor, fps=1.0, device=device)
    print("here2")

    # B. Shot Detection
    print("[Shots] Detecting boundaries...")
    cuts = detect_shots_pyscenedetect(video_path, threshold=3.0)

    # C. Frame Extraction
    print("[Frames] Extracting 16 frames per shot...")
    shot_tensors = extract_frames_per_shot(video_path, cuts, num_frames_per_shot=16)
    if not shot_tensors:
        raise ValueError("No valid shots extracted from query video.")

    # D. VideoMAE Embeddings (for Coarse Retrieval)
    print("[Embeddings] Running VideoMAEv2 Inference...")
    embeddings = process_shots_to_embeddings(
        shot_tensors, 
        model=videomae_model, 
        device=device,
        batch_size=8
    )
    
    return embeddings.astype(np.float32), isc_feats


# ==========================================
# 2. Retrieval Logic
# ==========================================

def get_candidate_videos(
    query_embeddings: np.ndarray,
    index: faiss.Index,
    metadata: dict,
    top_k_shots: int = 5
) -> set:
    """
    Searches FAISS for every shot in the query.
    Returns a set of unique 'video_title' strings that appear in the top K results.
    """
    print(f"\n[Retrieval] Searching FAISS for {query_embeddings.shape[0]} query shots...")
    
    # Search: distances (D) and indices (I)
    D, I = index.search(query_embeddings, top_k_shots)
    
    candidates = set()
    
    for shot_idx in range(len(I)):
        for rank in range(top_k_shots):
            db_id = int(I[shot_idx][rank])
            
            # -1 indicates no match found (e.g., empty DB)
            if db_id == -1: 
                continue
                
            if db_id in metadata:
                # We identify candidates by their title (filename) to find their ISC .npy files later
                vid_title = metadata[db_id].get("video_title")
                if vid_title:
                    candidates.add(vid_title)

    print(f"[Candidates] Found {len(candidates)} unique video candidates.")
    return candidates


def rerank_candidates(
    query_isc: np.ndarray,
    candidates: set,
    transvcl_wrapper: TransVCLWrapper,
    isc_db_folder: str
) -> list:
    """
    Runs TransVCL on the Query vs Each Candidate.
    Returns list of dicts: {video, score, alignment} sorted by score.
    """
    print(f"\n[Re-ranking] Running TransVCL on {len(candidates)} candidates...")
    
    results = []
    
    for vid_title in tqdm(candidates, unit="pair"):
        # 1. Load Reference ISC Features
        ref_path = os.path.join(isc_db_folder, f"{vid_title}.npy")
        
        if not os.path.exists(ref_path):
            print(f"Warning: ISC features missing for {vid_title} at {ref_path}")
            continue
            
        try:
            ref_isc = np.load(ref_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {ref_path}: {e}")
            continue

        # 2. Run TransVCL
        # Returns list of [q_start, r_start, q_end, r_end, score]
        alignments = get_transvcl_alignments(
            feat1=query_isc, 
            feat2=ref_isc, 
            model_wrapper=transvcl_wrapper, 
            topk=1 # We only care about the best alignment segment per video
        )
        
        if not alignments:
            continue
            
        # 3. Store Result
        best_align = alignments[0] # The one with highest confidence
        score = best_align[4]
        
        results.append({
            "video_title": vid_title,
            "score": float(score),
            "alignment": {
                "query_start_frame": int(best_align[0]),
                "ref_start_frame": int(best_align[1]),
                "query_end_frame": int(best_align[2]),
                "ref_end_frame": int(best_align[3])
            }
        })
        
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ==========================================
# 3. Main Execution
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Video Retrieval Query Pipeline")
    parser.add_argument("--query", required=True, help="Path to query video file")
    parser.add_argument("--db_index", default="db/shots.faiss")
    parser.add_argument("--db_meta", default="db/shots.json")
    parser.add_argument("--isc_dir", default="db/isc_features", help="Folder containing reference .npy files")
    parser.add_argument("--transvcl_model", default="TransVCL/transvcl/weights/model_1.pth")
    parser.add_argument("--top_k", type=int, default=5, help="Number of final results to show")
    parser.add_argument("--threshold", type=float, default=0, help="Min TransVCL score to display")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Resources
    print("--- Loading Resources ---")
    
    # DB
    index, metadata, _ = load_db(args.db_index, args.db_meta, 768)
    if index.ntotal == 0:
        print("Error: Database is empty. Please run store.py first.")
        return

    # Models
    print("Loading VideoMAEv2...")
    videomae_model = load_model()
    
    print("Loading ISC Extractor...")
    isc_extractor = ISCFeatureExtractor(device=device)
    
    print("Loading TransVCL...")
    try:
        transvcl = TransVCLWrapper(model_path=args.transvcl_model, device=device)
    except FileNotFoundError:
        print(f"Error: TransVCL model not found at {args.transvcl_model}")
        return

    # 2. Process Query
    try:
        q_embs, q_isc = process_query_video(
            args.query, videomae_model, isc_extractor, device
        )
    except Exception as e:
        print(f"Failed to process query video: {e}")
        return

    # 3. Coarse Retrieval (Filter Candidates)
    candidates = get_candidate_videos(q_embs, index, metadata, top_k_shots=5)
    
    if not candidates:
        print("No candidates found in database.")
        return

    # 4. Fine-Grained Re-ranking (TransVCL)
    results = rerank_candidates(q_isc, candidates, transvcl, args.isc_dir)

    # 5. Output Results
    print(f"\n=== Top Results (Threshold > {args.threshold}) ===")
    
    count = 0
    for res in results:
        if res['score'] < args.threshold:
            continue
            
        print(f"[{count+1}] Video: {res['video_title']}")
        print(f"    Confidence: {res['score']:.4f}")
        print(f"    Alignment:  Query Frames {res['alignment']['query_start_frame']}-{res['alignment']['query_end_frame']} "
              f"<--> Ref Frames {res['alignment']['ref_start_frame']}-{res['alignment']['ref_end_frame']}")
        print("-" * 40)
        
        count += 1
        if count >= args.top_k:
            break

    if count == 0:
        print("No matches passed the threshold.")

if __name__ == "__main__":
    main()