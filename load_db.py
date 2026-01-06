import faiss
import json
import os

def load_db(index_path: str, metadata_path: str, dim: int):
    """
    Load (index, metadata) if both exist; else create fresh.
    metadata maps int_id -> {"video_id": str, "offset_sec": float}
    """
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(index_path)
        with open(metadata_path, "r") as f:
            raw = json.load(f)
            metadata = {int(k): v for k, v in raw.items()}
        next_id = max(metadata.keys(), default=-1) + 1
    else:
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        metadata = {}
        next_id = 0

    return index, metadata, next_id
