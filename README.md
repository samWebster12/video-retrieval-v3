# Video Retrieval v3

A two-stage video retrieval system combining:
- **Coarse retrieval** using VideoMAEv2 embeddings + FAISS
- **Fine-grained re-ranking** using TransVCL for temporal alignment

---

## 1. Environment Setup

### Create and activate Conda environment
```bash
conda create -n video_retrieval_v3 python=3.10
conda activate video_retrieval_v3
```

### Install dependencies
```bash
conda env update -f environment.yml
```

---

## 2. Initialize Git Submodules

This repository depends on external submodules (TransVCL).

```bash
git submodule update --init --recursive
```

---

## 3. Download TransVCL Weights

Navigate to the TransVCL weights directory:
```bash
cd TransVCL/transvcl/weights
```

Download the pretrained model:
```bash
gdown 19H45GO_mVpwVpcPQIOyseCBmjpdYtYv4
```

Return to the project root:
```bash
cd ../../..
```

---

## 4. Storing Videos in the Database

Edit the `main()` function in **`store.py`** to point to your videos.

### Example `store.py` configuration
```python
def main():
    # --- CONFIGURATION ---
    video_paths = [
        '/home/sam/reverse-video-search/videos/bigbuckbunny/bigbuckbunny.mp4'
    ]
    
    index_path = "db/shots.faiss"
    metadata_path = "db/shots.json"
    isc_folder = "db/isc_features"
    
    vector_dim = 768  # VideoMAEv2 Base
    # ---------------------

    # 1. Setup Directories
    os.makedirs("db", exist_ok=True)
    os.makedirs(isc_folder, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"System Device: {device}")

    # 2. Load DB
    index, meta, next_id = load_db(index_path, metadata_path, vector_dim)
    print(f"DB Loaded. Vectors: {index.ntotal}")

    # 3. Load Models
    print("Loading VideoMAEv2...")
    videomae_model = load_model()
    
    print("Loading ISC Extractor...")
    isc_extractor = ISCFeatureExtractor(device=device)

    # 4. Process Videos
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

    # 5. Save DB
    save_faiss_db(index, meta, index_path, metadata_path)
    print("All done.")
```

### Run the store pipeline
```bash
python store.py
```

---

## 5. Querying the Database

Once the database is built, you can query it using `query.py`.

### Basic query command
```bash
python query.py --query <path_to_query_video>
```

### Example
```bash
python query.py --query assets/query/example.mp4
```

### Key arguments
- `--query` : Path to query video (required)
- `--top_k` : Number of final results to display (default: 5)
- `--threshold` : Minimum TransVCL score to display (default: 0)

---

## Notes

- **You must run `store.py` before querying**
- GPU with CUDA 12.1 compatibility is required
- TransVCL weights must be downloaded manually

---

## License / Attribution

This project builds on:
- VideoMAEv2
- TransVCL (AAAI 2023)
