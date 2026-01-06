import argparse
import json
import torch
import sys
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any


# TransVCL Dependencies (Must be in PYTHONPATH)
current_dir = os.path.dirname(os.path.abspath(__file__))
transvcl_repo_path = os.path.join(current_dir, "TransVCL")

# Add it to sys.path so Python can find 'exps' and 'transvcl' modules
if transvcl_repo_path not in sys.path:
    sys.path.append(transvcl_repo_path)

from exps.exp import Exp
from transvcl.utils import postprocess

# ==========================================
# 1. The Clean New Function (API)
# ==========================================

def get_transvcl_alignments(
    feat1: np.ndarray, 
    feat2: np.ndarray, 
    model_wrapper: "TransVCLWrapper",
    topk: int = 20
) -> List[List[float]]:
    """
    Takes two sets of ISC features and returns alignments.
    
    Args:
        feat1: Query features (N, 256)
        feat2: Reference features (M, 256)
        model_wrapper: Loaded instance of TransVCLWrapper
        topk: Number of top alignments to return.
        
    Returns:
        List of [q_start, r_start, q_end, r_end, score]
        Coordinates are frame indices (floats).
    """
    return model_wrapper.predict(feat1, feat2, topk=topk)


# ==========================================
# 2. The Engine (Class)
# ==========================================

class TransVCLWrapper:
    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda", 
        conf_thre: float = 0.1, 
        nms_thre: float = 0.3,
        feat_length: int = 1200,
        img_size: int = 640
    ):
        self.device = device if torch.cuda.is_available() and "cuda" in device else "cpu"
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        self.feat_length = feat_length
        self.img_size = img_size
        
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        print(f"[TransVCL] Loading model from {model_path} on {self.device}...")
        exp = Exp()
        model = exp.get_model()
        model.eval()
        
        # Load weights
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = model.to(self.device)
        return model

    @torch.no_grad()
    def predict(self, feat1_np: np.ndarray, feat2_np: np.ndarray, topk: int = 20) -> List[List[float]]:
        # 1. Validation
        if feat1_np.ndim != 2 or feat1_np.shape[1] != 256:
            raise ValueError(f"Feat1 shape mismatch: {feat1_np.shape} (Expected N,256)")
        if feat2_np.ndim != 2 or feat2_np.shape[1] != 256:
            raise ValueError(f"Feat2 shape mismatch: {feat2_np.shape} (Expected M,256)")

        # 2. Pre-processing (Chunking)
        batches = self._make_batches(feat1_np, feat2_np)
        
        # 3. Inference
        batch_feat_result = {}
        bs = 256 # Inference batch size
        
        for start in range(0, len(batches), bs):
            chunk = batches[start : start + bs]
            
            # Prepare Tensors
            feat1 = torch.stack([x[0] for x in chunk]).to(self.device)
            feat2 = torch.stack([x[1] for x in chunk]).to(self.device)
            mask1 = torch.stack([x[2] for x in chunk]).to(self.device)
            mask2 = torch.stack([x[3] for x in chunk]).to(self.device)
            
            # img_info format: List[Tensor(Batch), Tensor(Batch)]
            info_q = torch.stack([x[4][0] for x in chunk]).to(self.device) # Len Q
            info_r = torch.stack([x[4][1] for x in chunk]).to(self.device) # Len R
            img_info = [info_q, info_r]
            
            file_names = [x[5] for x in chunk] # Just IDs for mapping back

            # Forward Pass
            model_outputs = self.model(feat1, feat2, mask1, mask2, file_names, img_info)

            # Post-Process (NMS)
            outputs = postprocess(
                model_outputs[1], 1, self.conf_thre, self.nms_thre, class_agnostic=True
            )

            # 4. Rescale Coordinates back to Chunk Size
            for bi, output in enumerate(outputs):
                name = file_names[bi]
                if output is None:
                    continue

                bboxes = output[:, :5].detach().cpu() # [x1, y1, x2, y2, score]

                # TransVCL internal logic:
                # x-axis (0,2) is Reference, y-axis (1,3) is Query
                # We need to scale them back based on img_size (640) vs actual len
                scale_q = (info_q[bi] / self.img_size).cpu()
                scale_r = (info_r[bi] / self.img_size).cpu()

                # Scale Reference (cols 0, 2)
                bboxes[:, 0] *= scale_r
                bboxes[:, 2] *= scale_r
                # Scale Query (cols 1, 3)
                bboxes[:, 1] *= scale_q
                bboxes[:, 3] *= scale_q

                # Reorder to standard [q1, r1, q2, r2, score]
                # Current: [r1, q1, r2, q2, score]
                batch_feat_result[name] = bboxes[:, (1, 0, 3, 2, 4)].tolist()

        # 5. Aggregate Global Coordinates
        global_boxes = self._aggregate_results(batch_feat_result)
        
        # Sort desc by score and take top-k
        global_boxes.sort(key=lambda x: x[4], reverse=True)
        return global_boxes[:topk]

    def _make_batches(self, feat1_np, feat2_np):
        """Chunks features into overlapping windows (naive non-overlapping for now per original code)."""
        L = self.feat_length
        f1_chunks = [feat1_np[i:i+L] for i in range(0, len(feat1_np), L)]
        f2_chunks = [feat2_np[j:j+L] for j in range(0, len(feat2_np), L)]

        batch_list = []
        for i, f1 in enumerate(f1_chunks):
            for j, f2 in enumerate(f2_chunks):
                # Padding
                f1_pad = self._pad_feat(f1, L)
                f2_pad = self._pad_feat(f2, L)
                
                # Masks
                m1 = torch.zeros(L, dtype=torch.bool)
                m2 = torch.zeros(L, dtype=torch.bool)
                m1[:len(f1)] = True
                m2[:len(f2)] = True
                
                # Length Info
                info = [torch.tensor([len(f1)], dtype=torch.float32), 
                        torch.tensor([len(f2)], dtype=torch.float32)]
                
                # ID for reconstruction
                idx_id = f"{i}_{j}"
                
                batch_list.append((f1_pad, f2_pad, m1, m2, info, idx_id))
        return batch_list

    def _pad_feat(self, feat_np, target_len):
        tensor = torch.from_numpy(feat_np).float()
        curr = tensor.shape[0]
        if curr >= target_len:
            return tensor[:target_len]
        padding = torch.zeros((target_len - curr, 256), dtype=torch.float32)
        return torch.cat([tensor, padding], dim=0)

    def _aggregate_results(self, batch_results):
        out = []
        L = self.feat_length
        
        for key, boxes in batch_results.items():
            i, j = map(int, key.split("_"))
            offset_q = i * L
            offset_r = j * L
            
            for box in boxes:
                # box is [q1, r1, q2, r2, score] local
                global_box = [
                    box[0] + offset_q, # q1
                    box[1] + offset_r, # r1
                    box[2] + offset_q, # q2
                    box[3] + offset_r, # r2
                    box[4]             # score
                ]
                out.append(global_box)
        return out


# ==========================================
# 3. Helpers & CLI
# ==========================================

def load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)

def main():
    ap = argparse.ArgumentParser(description="Clean TransVCL Pair Predictor")
    ap.add_argument("--feat1", required=True, help="Path to Query .npy")
    ap.add_argument("--feat2", required=True, help="Path to Reference .npy")
    ap.add_argument("--model", required=True, help="Path to model checkpoint")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    # 1. Load Model
    wrapper = TransVCLWrapper(model_path=args.model)

    # 2. Load Data
    f1 = load_npy(args.feat1)
    f2 = load_npy(args.feat2)

    # 3. Predict
    results = get_transvcl_alignments(f1, f2, wrapper, topk=args.topk)

    # 4. Output
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()